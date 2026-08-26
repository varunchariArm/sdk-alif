#!/usr/bin/env python3
"""Export torchvision MobileNetV2 weights through a native NHWC Keras graph.

AI Edge Torch preserves the PyTorch network's internal NCHW layout.  Although
that TFLite graph is valid on the CPU, older embedded Vela releases can lower
it incorrectly.  This exporter transfers the identical checkpoint weights to
the equivalent Keras MobileNetV2 topology before full-int8 conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_torchvision_keras_mobilenet_v2() -> tf.keras.Model:
    """Build torchvision's MobileNetV2 topology in native NHWC form."""
    image = tf.keras.Input((224, 224, 3), name="image")
    tensor = image
    sequence = 0

    def conv_bn_relu(
        value, filters: int, kernel: int, stride: int = 1, groups: int = 1
    ):
        nonlocal sequence
        prefix = f"features_{sequence}"
        sequence += 1
        if kernel == 3:
            # torchvision Conv2d uses symmetric padding=1, including stride 2.
            value = tf.keras.layers.ZeroPadding2D(1, name=f"{prefix}_pad")(value)
            padding = "valid"
        else:
            padding = "same"
        if groups == filters:
            value = tf.keras.layers.DepthwiseConv2D(
                kernel, strides=stride, padding=padding, use_bias=False,
                name=f"{prefix}_depthwise",
            )(value)
        else:
            value = tf.keras.layers.Conv2D(
                filters, kernel, strides=stride, padding=padding,
                use_bias=False, name=f"{prefix}_conv",
            )(value)
        value = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, name=f"{prefix}_bn"
        )(value)
        return tf.keras.layers.ReLU(max_value=6.0, name=f"{prefix}_relu6")(value)

    tensor = conv_bn_relu(tensor, 32, 3, stride=2)
    input_channels = 32
    settings = (
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )
    for expansion, output_channels, repeats, first_stride in settings:
        for repeat in range(repeats):
            stride = first_stride if repeat == 0 else 1
            residual = tensor
            hidden_channels = int(round(input_channels * expansion))
            if expansion != 1:
                tensor = conv_bn_relu(tensor, hidden_channels, 1)
            tensor = conv_bn_relu(
                tensor, hidden_channels, 3, stride=stride, groups=hidden_channels
            )
            prefix = f"features_{sequence}"
            sequence += 1
            tensor = tf.keras.layers.Conv2D(
                output_channels, 1, use_bias=False, name=f"{prefix}_project"
            )(tensor)
            tensor = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, name=f"{prefix}_bn"
            )(tensor)
            if stride == 1 and input_channels == output_channels:
                tensor = tf.keras.layers.Add(name=f"{prefix}_add")(
                    [residual, tensor]
                )
            input_channels = output_channels

    tensor = conv_bn_relu(tensor, 1280, 1)
    tensor = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(tensor)
    probabilities = tf.keras.layers.Dense(
        1000, activation="softmax", name="predictions"
    )(tensor)
    return tf.keras.Model(image, probabilities, name="torchvision_mobilenet_v2")


def transfer_weights(torch_model: nn.Module, keras_model: tf.keras.Model) -> None:
    torch_convs = [
        layer for layer in torch_model.modules() if isinstance(layer, nn.Conv2d)
    ]
    keras_convs = [
        layer
        for layer in keras_model.layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
    ]
    if len(torch_convs) != len(keras_convs):
        raise RuntimeError(
            f"convolution count mismatch: torch={len(torch_convs)} "
            f"keras={len(keras_convs)}"
        )

    for source, destination in zip(torch_convs, keras_convs):
        weight = source.weight.detach().cpu().numpy()
        if isinstance(destination, tf.keras.layers.DepthwiseConv2D):
            if source.groups != source.in_channels or source.out_channels != source.in_channels:
                raise RuntimeError(f"unexpected depthwise layer: {source}")
            converted = np.transpose(weight[:, 0, :, :], (1, 2, 0))[:, :, :, None]
        else:
            converted = np.transpose(weight, (2, 3, 1, 0))
        values = [converted]
        if source.bias is not None:
            values.append(source.bias.detach().cpu().numpy())
        destination.set_weights(values)

    torch_bns = [
        layer for layer in torch_model.modules() if isinstance(layer, nn.BatchNorm2d)
    ]
    keras_bns = [
        layer for layer in keras_model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    if len(torch_bns) != len(keras_bns):
        raise RuntimeError(
            f"batch-normalization count mismatch: torch={len(torch_bns)} "
            f"keras={len(keras_bns)}"
        )
    for source, destination in zip(torch_bns, keras_bns):
        destination.epsilon = source.eps
        destination.set_weights(
            [
                source.weight.detach().cpu().numpy(),
                source.bias.detach().cpu().numpy(),
                source.running_mean.detach().cpu().numpy(),
                source.running_var.detach().cpu().numpy(),
            ]
        )

    torch_dense = torch_model.classifier[-1]
    keras_dense = keras_model.get_layer("predictions")
    keras_dense.set_weights(
        [
            torch_dense.weight.detach().cpu().numpy().T,
            torch_dense.bias.detach().cpu().numpy(),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=32)
    args = parser.parse_args()

    torch_model = mobilenet_v2(weights=None).eval()
    torch_model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    keras_model = build_torchvision_keras_mobilenet_v2()
    transfer_weights(torch_model, keras_model)

    # Refuse to export if topology, padding, or weight transfer differs from
    # the PyTorch source. This protects the backend comparison from silently
    # benchmarking two numerically different networks.
    generator = np.random.default_rng(29)
    reference = generator.standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.no_grad():
        torch_output = torch.softmax(torch_model(torch.from_numpy(reference)), 1)
    keras_output = keras_model(np.transpose(reference, (0, 2, 3, 1)), training=False)
    maximum_error = float(
        np.max(np.abs(torch_output.numpy() - np.asarray(keras_output)))
    )
    if maximum_error > 1e-5:
        raise RuntimeError(f"PyTorch/Keras equivalence failed: max error={maximum_error}")
    print(f"PyTorch/Keras max error={maximum_error:.9g}")

    @tf.function(
        input_signature=[tf.TensorSpec((1, 224, 224, 3), tf.float32, name="image")]
    )
    def serve(image):
        return {"probabilities": keras_model(image, training=False)}

    concrete = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    generator = np.random.default_rng(29)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    stddev = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def representative_dataset():
        for _ in range(args.calibration_samples):
            sample = generator.random((1, 224, 224, 3), dtype=np.float32)
            yield [(sample - mean) / stddev]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    if args.labels_output:
        args.labels_output.parent.mkdir(parents=True, exist_ok=True)
        args.labels_output.write_text(
            "".join(
                f"{label}\n"
                for label in MobileNet_V2_Weights.DEFAULT.meta["categories"]
            )
        )
    interpreter = tf.lite.Interpreter(model_content=data)
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    print(f"output={args.output} bytes={len(data)}")
    print(f"input={tuple(input_info['shape'])}/{input_info['quantization']}")
    print(f"output={tuple(output_info['shape'])}/{output_info['quantization']}")


if __name__ == "__main__":
    main()
