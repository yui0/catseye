#!/usr/bin/env python3
"""Writes a tiny random Inkling-2-Loop checkpoint in the Hugging Face layout (same tensor names, NVFP4 experts in
the first layers, MXFP8 in the last two) so that inkling_quantize / inkling can be tested without the 20 GB model.
usage: make_tiny.py OUT_DIR"""
import json, os, sys
import numpy as np
import torch
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
g = torch.Generator().manual_seed(1234)

H, NL, E, TOPK, I = 64, 8, 8, 2, 32
NQ, NKV, HD = 4, 2, 32            # full attention
SNQ, SNKV, SHD = 4, 2, 16         # sliding attention
WINDOW, REL_EXT, DREL = 8, 12, 4
BLOCK, LS, LE, PASSES, RANK = 2, 2, 6, 3, 8
V, VPAD = 100, 104
FULL = {3, 7}
K = 4
cfg = {
    "dtype": "bfloat16", "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop", "eos_token_id": 5,
    "text_config": {
        "model_type": "inkling2_loop_text", "hidden_size": H, "num_hidden_layers": NL, "vocab_size": VPAD, "unpadded_vocab_size": V,
        "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD, "d_rel": DREL, "rel_extent": REL_EXT,
        "log_scaling_n_floor": 4, "log_scaling_alpha": 0.5, "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 6.0, "rel_l2_clip_threshold": 2.0,
        "local_layer_ids": [i for i in range(NL) if i not in FULL], "sconv_kernel_size": K, "final_logit_softcapping": 30.0,
        "swa_head_dim": SHD, "swa_num_attention_heads": SNQ, "swa_num_key_value_heads": SNKV, "sliding_window_size": WINDOW,
        "n_layers_per_attn_res_block": BLOCK, "n_routed_experts": E, "num_experts_per_tok": TOPK, "n_shared_experts": 0,
        "intermediate_size": I, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.5,
        "loop_config": {"core_start": LS, "core_end": LE, "num_passes": PASSES, "gate_rank": RANK, "is_varloop": False},
    },
}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

T = {}
def rnd(*shape, std=0.2): return (torch.randn(*shape, generator=g) * std)
def bf(x): return x.to(torch.bfloat16).float()   # the checkpoint is bf16: keep the reference exact
def put(name, x, dtype=torch.bfloat16): T[name] = x.to(dtype).contiguous()

P = "model.llm."
put(P + "embed.weight", rnd(VPAD, H, std=1.0))
put(P + "embed_norm.weight", 1 + rnd(H, std=0.1))
put(P + "unembed.weight", rnd(VPAD, H, std=0.5))
put(P + "norm.weight", 1 + rnd(H, std=0.1))
put(P + "attn_res.output_query", rnd(H, std=0.5), torch.float32)
for l in range(NL):
    nq, nkv, hd = (NQ, NKV, HD) if l in FULL else (SNQ, SNKV, SHD)
    relw = REL_EXT if l in FULL else WINDOW
    a = f"{P}layers.{l}.attn."
    put(a + "wq_du.weight", rnd(nq * hd, H)); put(a + "wk_dv.weight", rnd(nkv * hd, H)); put(a + "wv_dv.weight", rnd(nkv * hd, H))
    put(a + "wo_ud.weight", rnd(H, nq * hd)); put(a + "gate_attn.weight", rnd(nq * hd, H)); put(a + "wr_du.weight", rnd(nq * DREL, H))
    put(a + "attn_sink_proj.weight", rnd(nq, H)); put(a + "q_l2_clip.scale", 1 + rnd(nq, hd, std=0.2)); put(a + "rel_l2_clip.scale", 1 + rnd(nq, DREL, std=0.2))
    put(a + "rel_logits_proj.proj", rnd(DREL, relw, std=0.6))
    if l in FULL:
        for kv in "kv":
            put(a + f"{kv}_sconv.static_w", rnd(K, nkv * hd, std=0.3)); put(a + f"{kv}_sconv.weight_proj", rnd(nkv * K, H, std=0.2))
            put(f"{P}layers.{l}.{kv}_input_norm.weight", 1 + rnd(H, std=0.1))
        put(f"{P}layers.{l}.q_input_norm.weight", 1 + rnd(H, std=0.1))
    else:
        put(a + "k_sconv.weight", rnd(nkv * hd, 1, K, std=0.4)); put(a + "v_sconv.weight", rnd(nkv * hd, 1, K, std=0.4))
        put(f"{P}layers.{l}.attn_norm.weight", 1 + rnd(H, std=0.1))
    put(f"{P}layers.{l}.attn_sconv.weight", rnd(H, 1, K, std=0.4)); put(f"{P}layers.{l}.mlp_sconv.weight", rnd(H, 1, K, std=0.4))
    put(f"{P}layers.{l}.mlp_norm.weight", 1 + rnd(H, std=0.1))
    put(f"{P}layers.{l}.mlp.gate.weight", rnd(E, H, std=0.6)); put(f"{P}layers.{l}.mlp.gate.bias", rnd(E, std=0.1), torch.float32)
    if l > 0: put(f"{P}attn_res.attn_queries.{l - 1}", rnd(3, H, std=0.5) if l in FULL else rnd(H, std=0.5), torch.float32)
    put(f"{P}attn_res.mlp_queries.{l}", rnd(H, std=0.5), torch.float32)
    if LS <= l < LE:
        for c in ("attn", "k", "v", "mlp"):
            c0 = f"{P}layers.{l}.controllers.{c}."
            put(c0 + "down_weight", rnd(RANK, 2 * H, std=0.2)); put(c0 + "forget_weight", rnd(H, RANK, std=0.4)); put(c0 + "gain_weight", rnd(H, RANK, std=0.4))
    # experts: NVFP4 (U8 codes, E4M3 block scales, per-expert amax / denominator) or MXFP8 (E4M3 values, E8M0 scales)
    e0 = f"{P}layers.{l}.mlp.experts."
    for w, rows, cols in (("w13", 2 * I, H), ("w2", H, I)):
        if l < NL - 2:
            codes = torch.randint(0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8)
            scale = (torch.rand(E, rows, cols // 16, generator=g) * 0.04 + 0.01).to(torch.float8_e4m3fn)
            T[e0 + w + "_weight"] = codes; T[e0 + w + "_weight_scale"] = scale
            T[e0 + w + "_weight_amax"] = torch.rand(E, generator=g) + 0.5
            T[e0 + w + "_weight_outer_scale_denominator"] = torch.tensor([1.5])
        else:
            T[e0 + w + "_weight"] = (torch.randn(E, rows, cols, generator=g) * 0.3).to(torch.float8_e4m3fn)
            T[e0 + w + "_weight_scale"] = (127 + torch.randint(-3, 0, (E, rows, cols // 32), generator=g)).to(torch.uint8)
nl = LE - LS
T[P + "middle_cycle_cross_pass_queries"] = rnd(2 * nl, H, std=0.5)
T[P + "middle_cycle_cross_pass_kv_queries"] = rnd(2 * nl, H, std=0.5)
for k in list(T):
    if k.endswith("_queries") or "attn_res" in k or "cross_pass" in k: T[k] = T[k].float()
save_file(T, f"{out}/model-00001-of-00001.safetensors", metadata={"format": "pt"})
print("tiny checkpoint written to", out, f"({sum(v.numel() * v.element_size() for v in T.values()) / 1e6:.1f} MB)")
