#!/usr/bin/env python3
"""Import the trained SSD-Slim TFLite kernels into ComparableSsdSlim.

The public SSD-Slim artifact contains folded, per-channel int8 weights.  This
utility dequantizes those constants into a common PyTorch checkpoint.  Both
the TFLite and ExecuTorch comparison artifacts are then generated from that
checkpoint, avoiding random detector weights or unrelated model recipes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from torch import nn

from comparable_models import ComparableSsdSlim, ConvBnRelu


def dequantize_tensor(interpreter: tf.lite.Interpreter, index: int) -> np.ndarray:
    detail = next(item for item in interpreter.get_tensor_details() if item["index"] == index)
    value = interpreter.get_tensor(index).astype(np.float32)
    params = detail["quantization_parameters"]
    scales = params["scales"].astype(np.float32)
    zero_points = params["zero_points"].astype(np.float32)
    if not len(scales):
        return value
    axis = int(params["quantized_dimension"])
    shape = [1] * value.ndim
    shape[axis] = len(scales)
    return (value - zero_points.reshape(shape)) * scales.reshape(shape)


def pytorch_weight(value: np.ndarray, depthwise: bool) -> torch.Tensor:
    if depthwise:
        # TFLite depthwise filter: [1, H, W, C]; PyTorch: [C, 1, H, W].
        value = value.transpose(3, 0, 1, 2)
    else:
        # TFLite Conv2D filter: [O, H, W, I]; PyTorch: [O, I, H, W].
        value = value.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(value))


def load_conv_bn(
    interpreter: tf.lite.Interpreter,
    block: ConvBnRelu,
    weight_index: int,
    bias_index: int,
) -> None:
    conv: nn.Conv2d = block[0]
    bn: nn.BatchNorm2d = block[1]
    weight = pytorch_weight(
        dequantize_tensor(interpreter, weight_index),
        depthwise=conv.groups == conv.in_channels and conv.in_channels > 1,
    )
    bias = torch.from_numpy(dequantize_tensor(interpreter, bias_index))
    if tuple(weight.shape) != tuple(conv.weight.shape):
        raise RuntimeError(
            f"tensor {weight_index} shape {tuple(weight.shape)} does not match "
            f"{tuple(conv.weight.shape)}"
        )
    with torch.no_grad():
        conv.weight.copy_(weight)
        # Recreate the already-folded TFLite bias while leaving a conventional
        # Conv-BN-ReLU training topology in the shared checkpoint.
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.weight.fill_((1.0 + bn.eps) ** 0.5)
        bn.bias.copy_(bias)


def load_conv(
    interpreter: tf.lite.Interpreter,
    conv: nn.Conv2d,
    weight_index: int,
    bias_index: int,
) -> None:
    weight = pytorch_weight(dequantize_tensor(interpreter, weight_index), False)
    bias = torch.from_numpy(dequantize_tensor(interpreter, bias_index))
    if tuple(weight.shape) != tuple(conv.weight.shape):
        raise RuntimeError(
            f"tensor {weight_index} shape {tuple(weight.shape)} does not match "
            f"{tuple(conv.weight.shape)}"
        )
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)


def backbone_blocks(model: ComparableSsdSlim) -> list[ConvBnRelu]:
    blocks: list[ConvBnRelu] = [model.stem]
    for stage in (model.stage0,):
        for layer in stage:
            blocks.extend((layer.depthwise, layer.pointwise))
    blocks.extend((model.stage1_down.depthwise, model.stage1_down.pointwise))
    for layer in model.stage1:
        blocks.extend((layer.depthwise, layer.pointwise))
    blocks.extend((model.stage2_down.depthwise, model.stage2_down.pointwise))
    blocks.extend((model.stage2.depthwise, model.stage2.pointwise))
    return blocks


def import_model(interpreter: tf.lite.Interpreter) -> ComparableSsdSlim:
    model = ComparableSsdSlim().eval()
    blocks = backbone_blocks(model)
    weight_indices = list(range(4, 54, 2))
    if len(blocks) != len(weight_indices):
        raise RuntimeError(f"backbone mapping mismatch: {len(blocks)} != {len(weight_indices)}")
    for block, weight_index in zip(blocks, weight_indices):
        load_conv_bn(interpreter, block, weight_index, weight_index + 1)

    # Extra 4x5 -> 2x3 feature stage.
    load_conv_bn(interpreter, model.extra[0], 66, 67)
    load_conv_bn(interpreter, model.extra[1].depthwise, 68, 69)
    load_conv_bn(interpreter, model.extra[1].pointwise, 70, 71)

    class_pairs = ((54, 55, 56, 57), (58, 59, 60, 61), (62, 63, 64, 65))
    box_pairs = ((74, 75, 76, 77), (78, 79, 80, 81), (82, 83, 84, 85))
    for head, (dw, dw_bias, out, out_bias) in zip(model.class_heads, class_pairs):
        load_conv_bn(interpreter, head.depthwise, dw, dw_bias)
        load_conv(interpreter, head.output, out, out_bias)
    for head, (dw, dw_bias, out, out_bias) in zip(model.box_heads, box_pairs):
        load_conv_bn(interpreter, head.depthwise, dw, dw_bias)
        load_conv(interpreter, head.output, out, out_bias)
    load_conv(interpreter, model.class_extra.output, 72, 73)
    load_conv(interpreter, model.box_extra.output, 86, 87)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(
        model_path=str(args.source), experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    model = import_model(interpreter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    # Confirm the common checkpoint preserves the source model's trained
    # behavior. The original artifact exposes softmax scores, whereas the
    # shared model intentionally exposes raw logits for firmware postprocess.
    generator = np.random.default_rng(29)
    source_input = generator.integers(-128, 128, (1, 120, 160, 1), dtype=np.int8)
    input_detail = interpreter.get_input_details()[0]
    scale, zero_point = input_detail["quantization"]
    float_input = (source_input.astype(np.float32) - zero_point) * scale
    with torch.no_grad():
        boxes, logits = model(torch.from_numpy(float_input).permute(0, 3, 1, 2))
        probabilities = torch.softmax(logits, dim=-1).numpy()
    interpreter.set_tensor(input_detail["index"], source_input)
    interpreter.invoke()
    outputs = interpreter.get_output_details()
    source_boxes = interpreter.get_tensor(outputs[0]["index"])
    source_scores = interpreter.get_tensor(outputs[1]["index"])
    box_scale, box_zero = outputs[0]["quantization"]
    score_scale, score_zero = outputs[1]["quantization"]
    source_boxes = (source_boxes.astype(np.float32) - box_zero) * box_scale
    source_scores = (source_scores.astype(np.float32) - score_zero) * score_scale
    print(f"checkpoint: {args.output} ({args.output.stat().st_size} bytes)")
    print(f"validation max-box-error={np.max(np.abs(boxes.numpy() - source_boxes)):.6f}")
    print(
        "validation max-score-error="
        f"{np.max(np.abs(probabilities - source_scores)):.6f}"
    )


if __name__ == "__main__":
    main()
