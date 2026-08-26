#!/usr/bin/env python3
"""Export full-int8 TFLite models from the shared comparison checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import ai_edge_torch
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from comparable_models import ComparableSsdSlim


class CheckpointMobileNetV2(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.model = mobilenet_v2(weights=None)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Match the deployed Arm ML-zoo MobileNetV2 contract. Keeping Softmax
        # in the graph gives both Vela/TFLM and ExecuTorch a quantized
        # probability output instead of backend-specific raw-logit handling.
        return torch.softmax(self.model(image), dim=1)


def load_model(kind: str, checkpoint: Path) -> tuple[nn.Module, tuple[int, ...]]:
    if kind == "ssd-slim":
        model = ComparableSsdSlim()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        return model.eval(), ComparableSsdSlim.input_shape
    return CheckpointMobileNetV2(checkpoint).eval(), (1, 3, 224, 224)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("ssd-slim", "mobilenet-v2"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(29)
    model, nchw_shape = load_model(args.model, args.weights)
    public_shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
    example = (torch.zeros(nchw_shape),)

    # AI Edge Torch creates an equivalent SavedModel. Use TensorFlow Lite's
    # representative-dataset PTQ on that graph so the public model boundary is
    # int8 rather than float with boundary Quantize/Dequantize operators.
    with tempfile.TemporaryDirectory(prefix=f"{args.model}-saved-model-") as saved:
        ai_edge_torch.convert(model, example, _saved_model_dir=saved)
        loaded = tf.saved_model.load(saved)
        signature = loaded.signatures["serving_default"]
        input_name = next(iter(signature.structured_input_signature[1]))

        class CameraInputWrapper(tf.Module):
            def __init__(self):
                super().__init__()
                self.base = loaded

            @tf.function
            def serve(self, image):
                image = tf.transpose(image, (0, 3, 1, 2))
                if args.model == "ssd-slim":
                    image = image * 255.0 - 128.0
                return signature(**{input_name: image})

        wrapper = CameraInputWrapper()
        concrete = wrapper.serve.get_concrete_function(
            tf.TensorSpec(public_shape, tf.float32, name="image")
        )
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete], wrapper
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        generator = np.random.default_rng(29)

        def representative_dataset():
            for _ in range(args.calibration_samples):
                sample = generator.random(public_shape, dtype=np.float32)
                if args.model == "mobilenet-v2":
                    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
                    sample = (sample - mean) / std
                yield [sample]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        data = converter.convert()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    interpreter = tf.lite.Interpreter(model_content=data)
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if any(item["dtype"] != np.int8 for item in inputs + outputs):
        raise RuntimeError("model boundary is not fully int8")
    print(f"model={args.model} output={args.output} bytes={len(data)}")
    print(
        "inputs="
        + ", ".join(
            f"{tuple(item['shape'])}/{item['dtype'].__name__}/{item['quantization']}"
            for item in inputs
        )
    )
    print(
        "outputs="
        + ", ".join(
            f"{tuple(item['shape'])}/{item['dtype'].__name__}/{item['quantization']}"
            for item in outputs
        )
    )


if __name__ == "__main__":
    main()
