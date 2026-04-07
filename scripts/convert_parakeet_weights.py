#!/usr/bin/env python3
"""Convert Parakeet ONNX weights to safetensors with clean names.

Usage:
    uv run --with onnx --with safetensors python scripts/convert_parakeet_weights.py \
        models/parakeet-v3-fp32 models/parakeet-v3-fp32

Reads encoder-model.onnx (+ .data) and decoder_joint-model.onnx, maps weights
to clean names, and writes encoder.safetensors + decoder.safetensors.
"""

import sys
import os
import numpy as np
import onnx
from safetensors.numpy import save_file


def clean_layer_name(name):
    """Apply consistent renaming to a weight name."""
    name = name.replace("norm_feed_forward1", "norm_ff1")
    name = name.replace("norm_feed_forward2", "norm_ff2")
    name = name.replace("norm_self_att", "norm_attn")
    name = name.replace("self_attn.", "attn.")
    name = name.replace("pointwise_conv1", "pw1")
    name = name.replace("pointwise_conv2", "pw2")
    return name


def build_weight_map(model):
    """Build a complete mapping: clean_name → onnx_initializer_name."""
    init_names = {init.name for init in model.graph.initializer if list(init.dims)}
    mapped_onnx_names = set()
    result = {}

    # 1. Map MatMul weights using graph node names (most reliable)
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for inp in node.input:
            if inp not in init_names or inp in mapped_onnx_names:
                continue
            clean = _matmul_to_name(node.name)
            if clean:
                result[clean] = inp
                mapped_onnx_names.add(inp)

    # 2. Map Conv weights using graph node names
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        prefix = _conv_to_name(node.name)
        if not prefix:
            continue
        # input[1] = weight, input[2] = bias
        for i, suffix in [(1, ".weight"), (2, ".bias")]:
            if i < len(node.input) and node.input[i] in init_names:
                onnx_name = node.input[i]
                if onnx_name not in mapped_onnx_names:
                    result[prefix + suffix] = onnx_name
                    mapped_onnx_names.add(onnx_name)

    # 3. Map remaining named weights (norms, biases, embeddings)
    for init in model.graph.initializer:
        if not list(init.dims) or init.name in mapped_onnx_names:
            continue
        if init.name.startswith("onnx::"):
            continue  # skip unmapped anonymous weights
        if init.name == "tile_repeats_const":
            continue
        clean = clean_layer_name(init.name)
        if clean not in result:
            result[clean] = init.name
            mapped_onnx_names.add(init.name)

    return result


def _matmul_to_name(node_name):
    """Convert MatMul node name → clean weight name."""
    name = node_name.lstrip("/")
    if not name.endswith("/MatMul"):
        return None
    name = name.removesuffix("/MatMul").replace("/", ".")
    name = name.replace("feed_forward1.", "ff1.")
    name = name.replace("feed_forward2.", "ff2.")
    name = name.replace("self_attn.", "attn.")
    name = name.replace("linear_q", "q")
    name = name.replace("linear_k", "k")
    name = name.replace("linear_v", "v")
    name = name.replace("linear_pos", "pos_proj")
    name = name.replace("linear_out", "out")
    return f"{name}.weight"


def _conv_to_name(node_name):
    """Convert Conv node name → clean component name."""
    name = node_name.lstrip("/")
    if not name.endswith("/Conv"):
        return None
    name = name.removesuffix("/Conv").replace("/", ".")
    name = name.replace("depthwise_conv", "dw")
    name = clean_layer_name(name)
    return name


def load_tensor(init, model_dir):
    """Load tensor data, handling external data references."""
    if init.data_location == onnx.TensorProto.EXTERNAL:
        ext = {e.key: e.value for e in init.external_data}
        path = os.path.join(model_dir, ext["location"])
        offset = int(ext.get("offset", 0))
        length = int(ext.get("length", 0))
        dtype_map = {1: np.float32, 7: np.int64, 6: np.int32}
        dtype = dtype_map.get(init.data_type, np.float32)
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(length), dtype=dtype).reshape(list(init.dims)).copy()
    else:
        return onnx.numpy_helper.to_array(init)


def convert_encoder(model_dir, output_dir):
    print("Loading encoder model...")
    model = onnx.load(os.path.join(model_dir, "encoder-model.onnx"), load_external_data=False)
    weight_map = build_weight_map(model)
    init_lookup = {init.name: init for init in model.graph.initializer}

    tensors = {}
    for clean_name, onnx_name in sorted(weight_map.items()):
        init = init_lookup.get(onnx_name)
        if init is None:
            print(f"  WARNING: {onnx_name} not found")
            continue
        arr = load_tensor(init, model_dir)
        tensors[clean_name] = arr

    # Print summary per layer
    layer_counts = {}
    for k in tensors:
        parts = k.split(".")
        if parts[0] == "layers":
            layer_counts[int(parts[1])] = layer_counts.get(int(parts[1]), 0) + 1

    for i in range(24):
        count = layer_counts.get(i, 0)
        if count < 17:
            print(f"  WARNING: layer {i} has {count} weights (expected ~17)")

    other = [k for k in tensors if not k.startswith("layers.")]
    print(f"  Pre-encode weights: {len(other)}")
    for k in sorted(other):
        print(f"    {k}: {list(tensors[k].shape)}")

    print(f"  Layer weights: {sum(layer_counts.values())} across {len(layer_counts)} layers")
    print(f"  Per layer: ~{sum(layer_counts.values()) // max(len(layer_counts), 1)}")

    output_path = os.path.join(output_dir, "encoder.safetensors")
    save_file(tensors, output_path)
    print(f"Saved {len(tensors)} encoder tensors to {output_path}")
    return tensors


def convert_decoder(model_dir, output_dir):
    print("\nLoading decoder model...")
    model = onnx.load(os.path.join(model_dir, "decoder_joint-model.onnx"))
    init_lookup = {init.name: init for init in model.graph.initializer}

    tensors = {}

    # Embedding
    emb = init_lookup.get("decoder.prediction.embed.weight")
    if emb:
        tensors["decoder.embed.weight"] = onnx.numpy_helper.to_array(emb)

    # LSTM weights (2 layers)
    lstm_map = {
        "onnx::LSTM_205": ("decoder.lstm.0.ih.weight", True),
        "onnx::LSTM_206": ("decoder.lstm.0.hh.weight", True),
        "onnx::LSTM_207": ("decoder.lstm.0.bias", True),
        "onnx::LSTM_225": ("decoder.lstm.1.ih.weight", True),
        "onnx::LSTM_226": ("decoder.lstm.1.hh.weight", True),
        "onnx::LSTM_227": ("decoder.lstm.1.bias", True),
    }
    for onnx_name, (clean_name, squeeze) in lstm_map.items():
        init = init_lookup.get(onnx_name)
        if init:
            arr = onnx.numpy_helper.to_array(init)
            if squeeze and arr.ndim > 2:
                arr = arr.squeeze(0)  # Remove num_directions=1 dim
            tensors[clean_name] = arr

    # Joint network
    # Map by shape since names are anonymous
    joint_shapes = {
        (1024, 640): "joint.enc.weight",
        (640, 640): "joint.pred.weight",
        (640, 8198): "joint.out.weight",
    }
    for node in model.graph.node:
        if node.op_type == "MatMul":
            for inp in node.input:
                init = init_lookup.get(inp)
                if init and list(init.dims):
                    shape = tuple(init.dims)
                    if shape in joint_shapes:
                        name = joint_shapes.pop(shape)
                        tensors[name] = onnx.numpy_helper.to_array(init)

    # Joint biases (already named)
    bias_map = {
        "joint.pred.bias": "joint.pred.bias",
        "joint.enc.bias": "joint.enc.bias",
        "joint.joint_net.2.bias": "joint.out.bias",
    }
    for onnx_name, clean_name in bias_map.items():
        init = init_lookup.get(onnx_name)
        if init:
            tensors[clean_name] = onnx.numpy_helper.to_array(init)

    for name, arr in sorted(tensors.items()):
        print(f"  {name}: {list(arr.shape)} ({arr.dtype})")

    output_path = os.path.join(output_dir, "decoder.safetensors")
    save_file(tensors, output_path)
    print(f"Saved {len(tensors)} decoder tensors to {output_path}")
    return tensors


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    enc = convert_encoder(model_dir, output_dir)
    dec = convert_decoder(model_dir, output_dir)

    print(f"\n=== Summary ===")
    print(f"Encoder: {len(enc)} tensors")
    print(f"Decoder: {len(dec)} tensors")
    print(f"Total: {len(enc) + len(dec)} tensors")


if __name__ == "__main__":
    main()
