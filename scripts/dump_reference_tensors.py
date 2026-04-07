"""
Dump intermediate tensors from the MLX parakeet reference implementation.

Usage:
    uv run --with 'parakeet-mlx' --with numpy scripts/dump_reference_tensors.py

Saves .npy files to scripts/reference_tensors/ for comparison with Rust impl.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Ensure parakeet-mlx is importable
sys.path.insert(0, "/tmp/parakeet-mlx")

import mlx.core as mx
from parakeet_mlx.utils import from_pretrained
from parakeet_mlx.audio import get_logmel, load_audio

MODEL_DIR = Path("models/parakeet-tdt-0.6b-v3")
WAV_PATH = Path("samples/dots.wav")
OUT_DIR = Path("scripts/reference_tensors")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(name: str, arr: mx.array):
    """Save MLX array as .npy (via numpy, C-contiguous for simple Rust loading)."""
    data = np.ascontiguousarray(np.array(arr, dtype=np.float32))
    path = OUT_DIR / f"{name}.npy"
    np.save(path, data)
    print(f"  {name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")


def main():
    print(f"Loading model from {MODEL_DIR}...")
    model = from_pretrained(str(MODEL_DIR), dtype=mx.float32)
    mx.eval(model.parameters())

    print(f"Loading audio from {WAV_PATH}...")
    audio = load_audio(WAV_PATH, model.preprocessor_config.sample_rate, dtype=mx.float32)
    print(f"  Audio samples: {audio.shape[0]}, duration: {audio.shape[0]/16000:.2f}s")

    # --- Stage 1: Mel features ---
    print("\n=== Stage 1: Mel features ===")
    mel = get_logmel(audio, model.preprocessor_config)
    mx.eval(mel)
    save("mel", mel)

    # --- Stage 2: Pre-encode (subsampling) ---
    print("\n=== Stage 2: Pre-encode ===")
    lengths = mx.array([mel.shape[1]])
    pre_enc_out, out_lengths = model.encoder.pre_encode(mel, lengths)
    mx.eval(pre_enc_out, out_lengths)
    save("pre_encode", pre_enc_out)
    print(f"  out_lengths: {int(out_lengths[0])}")

    # --- Stage 3: Positional encoding ---
    print("\n=== Stage 3: Positional encoding ===")
    x_scaled, pos_emb = model.encoder.pos_enc(pre_enc_out, offset=0)
    mx.eval(x_scaled, pos_emb)
    save("pos_enc_x", x_scaled)
    save("pos_emb", pos_emb)

    # --- Stage 4: Conformer layer 0 (sub-stages) ---
    print("\n=== Stage 4: Conformer layer 0 ===")
    layer = model.encoder.layers[0]

    # 4a: FF1
    x_l = x_scaled
    x_l = x_l + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(x_l))
    mx.eval(x_l)
    save("layer0_after_ff1", x_l)

    # 4b: Self-attention
    x_norm = layer.norm_self_att(x_l)
    attn_out = layer.self_attn(x_norm, x_norm, x_norm, mask=None, pos_emb=pos_emb)
    mx.eval(attn_out)
    save("layer0_attn_out", attn_out)
    x_l = x_l + attn_out
    mx.eval(x_l)
    save("layer0_after_attn", x_l)

    # 4c: Conv module (sub-stages)
    conv = layer.conv
    cx = layer.norm_conv(x_l)

    cx_pw1 = conv.pointwise_conv1(cx)
    mx.eval(cx_pw1)
    save("layer0_conv_after_pw1", cx_pw1)

    import mlx.nn as nn_mod
    cx_glu = nn_mod.glu(cx_pw1, axis=2)
    mx.eval(cx_glu)
    save("layer0_conv_after_glu", cx_glu)

    cx_padded = mx.pad(cx_glu, ((0, 0), (conv.padding, conv.padding), (0, 0)))
    cx_dw = conv.depthwise_conv(cx_padded)
    mx.eval(cx_dw)
    save("layer0_conv_after_dw", cx_dw)

    cx_bn = conv.batch_norm(cx_dw)
    mx.eval(cx_bn)
    save("layer0_conv_after_bn", cx_bn)

    cx_act = nn_mod.SiLU()(cx_bn)
    mx.eval(cx_act)
    save("layer0_conv_after_silu", cx_act)

    conv_out = conv.pointwise_conv2(cx_act)
    mx.eval(conv_out)
    save("layer0_conv_out", conv_out)
    x_l = x_l + conv_out
    mx.eval(x_l)
    save("layer0_after_conv", x_l)

    # 4d: FF2
    x_l = x_l + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(x_l))
    mx.eval(x_l)
    save("layer0_after_ff2", x_l)

    # 4e: Final norm
    layer0_out = layer.norm_out(x_l)
    mx.eval(layer0_out)
    save("layer0", layer0_out)

    # --- Stage 4.5: More layers ---
    print("\n=== Stage 4.5: Layers 1-5 ===")
    x_running = layer0_out
    for li in range(1, min(6, len(model.encoder.layers))):
        x_running = model.encoder.layers[li](x_running, pos_emb=pos_emb)
        mx.eval(x_running)
        save(f"layer{li}", x_running)

    # --- Stage 5: Full encoder output ---
    print("\n=== Stage 5: Full encoder ===")
    encoder_out, enc_lengths = model.encoder(mel)
    mx.eval(encoder_out, enc_lengths)
    save("encoder_out", encoder_out)

    # --- Stage 6: Predictor (blank input) ---
    print("\n=== Stage 6: Predictor (blank input) ===")
    pred_out, pred_state = model.decoder(None, None)
    mx.eval(pred_out)
    save("pred_blank", pred_out)

    # --- Stage 7: Joint logits at t=0 ---
    print("\n=== Stage 7: Joint logits at t=0 ===")
    enc_t0 = encoder_out[:, 0:1, :]
    joint_out = model.joint(enc_t0, pred_out)
    mx.eval(joint_out)
    # joint_out shape: [1, 1, 1, output_dim]
    logits = joint_out[0, 0, 0, :]
    mx.eval(logits)
    save("joint_logits_t0", logits)

    # Print top-5 token logits for comparison
    vocab_size = len(model.vocabulary)
    token_logits = np.array(logits[:vocab_size + 1], dtype=np.float32)
    top5_idx = np.argsort(token_logits)[::-1][:5]
    print(f"\n  Top-5 token logits (vocab_size={vocab_size}, blank={vocab_size}):")
    for idx in top5_idx:
        tok = model.vocabulary[idx] if idx < vocab_size else "<blank>"
        print(f"    [{idx}] {tok!r}: {token_logits[idx]:.4f}")

    dur_logits = np.array(logits[vocab_size + 1:], dtype=np.float32)
    print(f"  Duration logits: {dur_logits.tolist()}")

    print(f"\nDone! Reference tensors saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
