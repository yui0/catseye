#!/usr/bin/env python3
"""Independent PyTorch reference of the Inkling-2-Loop text model, written from the Hugging Face modeling code
(transformers inkling2_loop). It reads the file written by inkling_quantize (so both implementations see exactly the
same quantized weights), processes the whole sequence at once (no incremental caches) and writes the float32 logits
of every position.  usage: ref_inkling.py MODEL.safetensors "1,2,3" OUT.f32"""
import json, math, struct, sys
import numpy as np
import torch

def load(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    T = {}
    for k, v in hdr.items():
        if k == "__metadata__": continue
        b, e = v["data_offsets"]; buf = raw[base + b: base + e]; shape = v["shape"]; dt = v["dtype"]
        if dt == "F32": x = np.frombuffer(buf, np.float32).reshape(shape)
        elif dt == "F16": x = np.frombuffer(buf, np.float16).astype(np.float32).reshape(shape)
        elif dt == "Q8_0":
            blk = np.frombuffer(buf, np.uint8).reshape(-1, 34); d = blk[:, :2].copy().view(np.float16).astype(np.float32)
            x = (blk[:, 2:].view(np.int8).astype(np.float32) * d).reshape(shape)
        elif dt == "Q4_0":
            blk = np.frombuffer(buf, np.uint8).reshape(-1, 18); d = blk[:, :2].copy().view(np.float16).astype(np.float32)
            lo = (blk[:, 2:] & 15).astype(np.float32) - 8; hi = (blk[:, 2:] >> 4).astype(np.float32) - 8
            x = (np.concatenate([lo, hi], 1) * d).reshape(shape)
        else: x = np.frombuffer(buf, np.uint8).copy()
        T[k] = x
    return T

path, ids, outp = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
T = load(path)
cfg = json.loads(bytes(T["config"]).decode())["text_config"]
t = lambda k: torch.from_numpy(np.array(T[k], dtype=np.float32))
H, NL, EPS = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["rms_norm_eps"]
LS, LE, PASSES = cfg["loop_config"]["core_start"], cfg["loop_config"]["core_end"], cfg["loop_config"]["num_passes"]
import os
PASSES = int(os.environ.get("PASSES", PASSES))
BLOCK = cfg["n_layers_per_attn_res_block"]; LOCAL = set(cfg["local_layer_ids"]); E, TOPK, I = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["intermediate_size"]
S = len(ids)

def rms(x, w=None, eps=EPS):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y
def l2clip(x, thr): return x * (thr / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
def softcap(x, c): return torch.tanh(x / c) * c

def sconv(x, w):             # x [S,C], w [C,K]: causal depthwise conv, (conv + x) / sqrt(K + 1)
    K = w.shape[1]; xp = torch.nn.functional.pad(x, (0, 0, K - 1, 0)); out = torch.zeros_like(x)
    for j in range(K): out += w[:, j] * xp[j: j + x.shape[0]]
    return (out + x) / math.sqrt(K + 1)
def sconv_dyn(x, static, wproj, dyn, nh):   # static [K,C], taps indexed by lag
    K, C = static.shape; hd = C // nh; dw = (dyn @ wproj.T).view(S, nh, K)
    xp = torch.nn.functional.pad(x, (0, 0, K - 1, 0)); out = torch.zeros_like(x)
    for lag in range(K):
        wt = static[lag].view(nh, hd)[None] + dw[:, :, lag, None]
        out += (wt * xp[K - 1 - lag: K - 1 - lag + S].view(S, nh, hd)).reshape(S, C)
    return (out + x) / math.sqrt(2 * K + 1)

def block_res(blocks, partial, q):     # blocks [S,nb,H] (nb may be 0), partial [S,H], q [nq,H] -> [S,nq,H]
    st = rms(torch.cat([blocks, partial[:, None]], 1), eps=1e-6)
    sc = torch.einsum("qh,snh->sqn", q, st) * H ** -0.5
    return torch.einsum("sqn,snh->sqh", sc.softmax(-1), st)

def msg_gate(x, msg, c):
    g = torch.cat([rms(x), rms(msg)], -1) @ c["down"].T
    return (1 - torch.tanh(g @ c["forget"].T)) * x + torch.tanh(g @ c["gain"].T) * msg

pos = torch.arange(S)
def attention(l, q_in, k_in, v_in, full):
    P = f"blk.{l}."
    nq, nkv, hd = (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]) if full else (cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"])
    q = (q_in @ t(P + "wq").T).view(S, nq, hd) * t(P + "qscale"); q = l2clip(q, cfg["qk_l2_clip_threshold"])
    rs = (q_in @ t(P + "wr").T).view(S, nq, cfg["d_rel"]) * t(P + "rscale"); rs = l2clip(rs, cfg["rel_l2_clip_threshold"])
    if full:
        tau = (1 + cfg["log_scaling_alpha"] * torch.log(((pos + 1).float() / cfg["log_scaling_n_floor"]).clamp(min=1.0)))[:, None, None]
        q, rs = q * tau, rs * tau
    k = k_in @ t(P + "wk").T; v = v_in @ t(P + "wv").T
    if full: k = sconv_dyn(k, t(P + "k_static"), t(P + "k_wproj"), k_in, nkv); v = sconv_dyn(v, t(P + "v_static"), t(P + "v_wproj"), v_in, nkv)
    else: k = sconv(k, t(P + "k_conv")); v = sconv(v, t(P + "v_conv"))
    k = l2clip(k.view(S, nkv, hd), cfg["qk_l2_clip_threshold"]); v = rms(v.view(S, nkv, hd))
    k = k.repeat_interleave(nq // nkv, 1); v = v.repeat_interleave(nq // nkv, 1)
    relproj = t(P + "relproj"); relw = relproj.shape[1]
    rel = torch.einsum("shr,rd->shd", rs, relproj)                       # [S,nh,relw]
    dist = pos[:, None] - pos[None, :]                                    # [S,S]
    bias = rel.gather(-1, dist.clamp(0, relw - 1)[:, None, :].expand(S, nq, S))
    bias = bias.masked_fill(((dist < 0) | (dist >= relw))[:, None, :], 0.0)
    logit = torch.einsum("shd,thd->hst", q, k) / math.sqrt(hd) + bias.permute(1, 0, 2)
    allowed = dist >= 0
    if not full: allowed = allowed & (dist < cfg["sliding_window_size"])
    logit = logit.masked_fill(~allowed[None], float("-inf"))
    sink = (q_in @ t(P + "sink").T).T[:, :, None]                         # [nh,S,1]
    p = torch.cat([logit, sink], -1).softmax(-1)[..., :-1]
    o = torch.einsum("hst,thd->shd", p, v) * torch.sigmoid((q_in @ t(P + "wgate").T).view(S, nq, hd))
    return o.reshape(S, nq * hd) @ t(P + "wo").T

def moe(l, x):
    P = f"blk.{l}."
    logits = x @ t(P + "router").T; sc = logits.sigmoid()
    idx = (sc + t(P + "router_bias")).topk(TOPK, -1).indices
    w = sc.gather(1, idx); w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg["route_scale"]
    w13 = t(P + "w13").view(E, 2 * I, H); w2 = t(P + "w2").view(E, H, I); out = torch.zeros_like(x)
    for s in range(S):
        for j in range(TOPK):
            e = idx[s, j]; gu = w13[e] @ x[s]; g, u = gu[:I], gu[I:]
            a = softcap(g, cfg["mlp_gate_softcap"]) * torch.sigmoid(g) * softcap(u, cfg["mlp_up_softcap"])
            out[s] += w[s, j] * (w2[e] @ a)
    return out

def layer(l, h, blocks, msg):
    P = f"blk.{l}."; full = l not in LOCAL; boundary = l % BLOCK == 0; resid = h
    if l > 0: x = block_res(blocks, h, t(P + "attn_res_q").view(-1, H))
    else: x = h[:, None]
    ctrl = lambda c: {k: t(f"{P}ctrl_{c}_{k}") for k in ("down", "forget", "gain")}
    if full:
        ins = []
        for a, c, nm in ((0, "q", "q_in_norm"), (1, "k", "k_in_norm"), (2, "v", "v_in_norm")):
            xi = x[:, a]
            if msg is not None: xi = msg_gate(xi, msg[:, a], ctrl(c))
            ins.append(rms(xi, t(P + nm)))
    else:
        xs = x[:, 0]; ins = []
        for a, c in enumerate("qkv"):
            ins.append(rms(msg_gate(xs, msg[:, a], ctrl(c)) if msg is not None else xs, t(P + "attn_norm")))
    a = sconv(attention(l, ins[0], ins[1], ins[2], full), t(P + "attn_sconv"))
    if boundary: blocks = torch.cat([blocks, resid[:, None]], 1); part = a
    else: part = resid + a
    mi = block_res(blocks, part, t(P + "mlp_res_q").view(-1, H))[:, 0]
    if msg is not None: mi = msg_gate(mi, msg[:, 3], ctrl("m"))
    y = sconv(moe(l, rms(mi, t(P + "post_norm"))), t(P + "mlp_sconv"))
    return part + y, blocks

with torch.no_grad():
    emb = t("embed")[torch.tensor(ids)]
    h = rms(emb, t("embed_norm")); blocks = h.new_zeros(S, 0, H)
    for l in range(LS):
        h, blocks = layer(l, h, blocks, None)
    pre_h, pre_blocks = h, blocks; npre = pre_blocks.shape[1]
    for l in range(LS, LE): h, blocks = layer(l, h, blocks, None)
    mq = t("msg_query")
    for p in range(1, PASSES):
        prev_blocks, prev_h = blocks[:, npre + 1:], h
        h, blocks = pre_h, pre_blocks
        for li, l in enumerate(range(LS, LE)):
            msg = block_res(prev_blocks, prev_h, mq[li])
            h, blocks = layer(l, h, blocks, msg)
    for l in range(LE, NL): h, blocks = layer(l, h, blocks, None)
    h = rms(block_res(blocks, h, t("out_query").view(1, -1))[:, 0], t("final_norm"))
    logits = softcap(h @ t("unembed").T, cfg["final_logit_softcapping"])
logits.numpy().astype(np.float32).tofile(outp)
print("reference logits", tuple(logits.shape), "argmax of last:", int(logits[-1].argmax()))
