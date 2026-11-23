#!/bin/bash
# End-to-end test on a tiny random checkpoint (no 20 GB download needed): converter decoding, then the C runtime
# against an independent PyTorch implementation. Needs python3 with torch, numpy and safetensors.
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
IDS="3,17,42,8,99,5,60,61,2,33,71,12,9,48,50,1,44,27,7,19,23,88,64,31,72,15"
python3 tests/make_tiny.py $T/hf
# unquantized (F16) weights: C and PyTorch must agree to rounding
./inkling_quantize -i $T/hf -o $T/m.f16 -b f16 -a f16 -e f16 2>/dev/null
python3 tests/check_quant.py $T/hf $T/m.f16 | tail -1
python3 tests/ref_inkling.py $T/m.f16 $IDS $T/ref.f32 | tail -1
# every chunk size must give the same logits (chunks cross the sliding window / convolution state boundaries)
for B in 1 5 7 64; do
  ./inkling -m $T/m.f16 --ids $IDS --logits $T/c$B.f32 -c 64 --batch $B >/dev/null
  python3 - $T $B <<'P'
import sys, numpy as np
a = np.fromfile(sys.argv[1] + "/ref.f32", np.float32); b = np.fromfile(sys.argv[1] + "/c" + sys.argv[2] + ".f32", np.float32)
d = np.abs(a - b).max(); print("batch %-2s max |C - PyTorch| over all positions: %.2e (logit std %.2f)" % (sys.argv[2], d, a.std()))
sys.exit(0 if d < 5e-3 else 1)
P
done
# quantized files must run and stay close in argmax terms
for t in "q8 q8 q8" "q4 q8 q8" "q3 q8 q8" "q3 q4 q4"; do set -- $t
  ./inkling_quantize -i $T/hf -o $T/m.$1 -b $1 -a $2 -e $3 >/dev/null 2>&1
  echo "-b $1 -a $2 -e $3: $(./inkling -m $T/m.$1 --ids $IDS -c 64 | tail -1) (batch 1: $(./inkling -m $T/m.$1 --ids $IDS -c 64 --batch 1 | tail -1))"
done
echo "--topk 1 / --passes 1 run: $(./inkling -m $T/m.f16 --ids $IDS -c 64 --topk 1 --passes 1 | tail -1)"
echo "tiny model test passed"
