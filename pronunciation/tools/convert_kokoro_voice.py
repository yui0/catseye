#!/usr/bin/env python3
"""Kokoro voice pack (.pt from hexgrad/Kokoro-82M/voices) -> raw float32 [510][256] (.bin) for tts_kokoro.h.

  convert_kokoro_voice.py af_heart.pt af_heart.bin
"""
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=True).float().reshape(-1, 256)
p.numpy().astype("<f4").tofile(sys.argv[2])
print(sys.argv[2], tuple(p.shape))
