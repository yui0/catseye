#!/usr/bin/env python3
"""Generate tiny structurally valid mock ONNX graphs (NO speech weights!).
Only Python's standard library, so CI can verify the C pipeline offline.
"""
import struct, sys, pathlib, wave, math

def vi(x):
    o = bytearray()
    while x > 127:
        o.append((x & 0x7f) | 0x80); x >>= 7
    o.append(x)
    return bytes(o)

def v(field, x):return vi((field << 3)) + vi(x)
def b(field, x):return vi((field << 3) | 2) + vi(len(x)) + x

def typ(shape):
    t = b(1,b(1,v(1,1)+b(2,b''.join(b(1,v(1,d)) for d in shape))))
    return t

def val(name, shape):return b(1,name.encode())+b(2,typ(shape))

def node(op, ins, outs):
    return b''.join(b(1,x.encode()) for x in ins) + b''.join(b(2,x.encode()) for x in outs)+b(4,op.encode())

def tensor(name, dims, dtype, raw):
    return b''.join(v(1,dim) for dim in dims)+v(2,dtype)+b(8,name.encode())+b(9,raw)

def graph(nodes, init, ins, outs):
    return b''.join(b(1,n) for n in nodes)+b(2,b'eg2_mock')+b''.join(b(5,i) for i in init)+b''.join(b(11,val(*x)) for x in ins)+b''.join(b(12,val(*x)) for x in outs)

def model(g):return v(1,9)+b(2,b'cats-eg2-mock')+b(7,g)+b(8,v(2,13))

out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parent.parent/'fixtures');out.mkdir(parents=True,exist_ok=True)
enc=graph([node('Reshape',['features','reshape_shape'],['soft_tokens'])],
          [tensor('reshape_shape',[3],7,struct.pack('<qqq',1,750,512))],
          [('features',[1,3000,128]),('half_mask',[1,1500])],
          [('soft_tokens',[1,750,512])])
(out/'eg2_mock_encoder.onnx').write_bytes(model(enc))
zero=struct.pack('<750f',*([0.]*750))
one=struct.pack('<750f',*([1.]*750))
reader=graph([node('Identity',['word_indices'],['top1_index']),node('Identity',['word_scores'],['top1_logit']),node('Identity',['blank_scores'],['blank_logit'])],
             [tensor('word_indices',[1,750],1,zero),tensor('word_scores',[1,750],1,one),tensor('blank_scores',[1,750],1,zero)],
             [('soft_tokens',[1,750,512]),('token_mask',[1,750])],
             [('top1_index',[1,750]),('top1_logit',[1,750]),('blank_logit',[1,750])])
(out/'eg2_mock_reader.onnx').write_bytes(model(reader))
(out/'eg2_mock_vocab.txt').write_text('hello\nworld\n',encoding='utf8')
with wave.open(str(out/'eg2_mock_one_second.wav'),'wb') as f:
    f.setnchannels(1); f.setsampwidth(2);f.setframerate(16000)
    f.writeframes(b''.join(struct.pack('<h',round(math.sin(i*2*math.pi*440/16000)*5000)) for i in range(16000)))

# Synthetic reader that returns a constant projected 768-vector per 40 ms
# token. Exercises native cosine ranking without pretending the model is trained.
projected=struct.pack('<768f',1.,*([0.]*767))*750
mock=graph([node('Identity',['projected'],['projected_tokens']),node('Identity',['blank_scores'],['blank_logit'])],
    [tensor('projected',[1,750,768],1,projected),tensor('blank_scores',[1,750],1,zero)],
    [('soft_tokens',[1,750,512]),('token_mask',[1,750])],
    [('projected_tokens',[1,750,768]),('blank_logit',[1,750])])
(out/'eg2_mock_reader_q8.onnx').write_bytes(model(mock))
# The .EG2V vectors are actual float32 data for two artificial word entries.
(out/'eg2_mock_word_table.eg2v').write_bytes(b'EG2V'+struct.pack('<II',1,2)+
    struct.pack('<768f',1.,*([0.]*767))+struct.pack('<768f',0.,1.,*([0.]*766)))
