#!/usr/bin/env python3
"""Export trained SSD-Slim and torchvision MobileNetV2 to Ethos-U PTEs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from torch import nn

# Some ExecuTorch development environments omit torchvision's host C++ ops.
# Load the extension first when it is usable; otherwise provide only the
# schemas needed while importing torchvision. Model construction never calls
# either host NMS implementation.
schemas = None
try:
    torchvision_spec = importlib.util.find_spec("torchvision")
    if torchvision_spec is None or torchvision_spec.origin is None:
        raise ImportError("torchvision package not found")
    extension_dir = Path(torchvision_spec.origin).parent
    extension = next(extension_dir.glob("_C*.so"))
    torch.ops.load_library(str(extension))
except (ImportError, OSError, StopIteration):
    schemas = torch.library.Library("torchvision", "DEF")
    schemas.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    schemas.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")

from torchvision.models import MobileNet_V2_Weights, mobilenet_v2  # noqa: E402
from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # noqa: E402

from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner  # noqa: E402
from executorch.backends.arm.quantizer import (  # noqa: E402
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes.quantize_io_pass import (  # noqa: E402
    QuantizeInputs,
    QuantizeOutputs,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

from comparable_models import ComparableSsdSlim  # noqa: E402


class SsdHeads(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        features = list(self.backbone((image - self.mean) / self.std).values())
        outputs = self.head(features)
        return outputs["bbox_regression"], outputs["cls_logits"]


class ComparableSsdCameraInput(nn.Module):
    """Expose the same normalized camera boundary as the TFLite artifact."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        return self.model(image * 255.0 - 128.0)


class NormalizedClassifier(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        return self.model((image - self.mean) / self.std)


def make_person_ssd(weights: Path) -> nn.Module:
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    classification = model.head.classification_head
    for block in classification.module_list:
        old = block[1]
        anchors = old.out_channels // 91
        replacement = nn.Conv2d(old.in_channels, anchors * 2, 1)
        with torch.no_grad():
            weight = old.weight.reshape(anchors, 91, old.in_channels, 1, 1)
            bias = old.bias.reshape(anchors, 91)
            replacement.weight.copy_(weight[:, :2].reshape_as(replacement.weight))
            replacement.bias.copy_(bias[:, :2].reshape_as(replacement.bias))
        block[1] = replacement
    classification.num_columns = 2
    return SsdHeads(model).eval()


def make_comparable_ssd(weights: Path) -> nn.Module:
    model = ComparableSsdSlim()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return ComparableSsdCameraInput(model.eval()).eval()


def selected_imagenet_classes(count: int) -> list[int]:
    selected = [round(i * 999 / (count - 1)) for i in range(count)]
    # Grace Hopper's jacket is ImageNet class 653. Keep it in the broadly
    # distributed subset so the startup image remains a useful known answer.
    nearest = min(range(count), key=lambda i: abs(selected[i] - 653))
    selected[nearest] = 653
    return sorted(set(selected))


def make_mv2(weights: Path, classes: list[int]) -> nn.Module:
    model = mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    old = model.classifier[1]
    replacement = nn.Linear(old.in_features, len(classes))
    with torch.no_grad():
        replacement.weight.copy_(old.weight[classes])
        replacement.bias.copy_(old.bias[classes])
    model.classifier[1] = replacement
    return NormalizedClassifier(model).eval()


def remove_unused_export_guards(graph_module: torch.fx.GraphModule) -> None:
    """Remove no-output guard modules emitted by some aarch64 torch wheels.

    These guards have no users and do not contribute to model semantics, but
    ExecuTorch's Arm annotation passes intentionally reject call_module nodes.
    Other torch builds omit the node during export, so normalize the graphs
    before applying the backend quantizer.
    """
    removed = []
    for node in list(graph_module.graph.nodes):
        if (
            node.op == "call_module"
            and str(node.target).startswith("_guards_fn")
            and not node.users
        ):
            removed.append(str(node.target))
            graph_module.graph.erase_node(node)
    for target in removed:
        if hasattr(graph_module, target):
            delattr(graph_module, target)
    if removed:
        graph_module.graph.lint()
        graph_module.recompile()
        print(f"removed unused export guards: {', '.join(removed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", choices=("ssd-person", "comparable-ssd", "mobilenet-v2")
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--mv2-classes", type=int, default=384)
    parser.add_argument("--labels-output", type=Path)
    args = parser.parse_args()

    if args.model == "ssd-person":
        model = make_person_ssd(args.weights)
        shape = (1, 3, 320, 320)
        target = "ethos-u55-256"
        system_config = "RTSS_HP_SRAM_MRAM"
        output_count = 2
    elif args.model == "comparable-ssd":
        model = make_comparable_ssd(args.weights)
        shape = ComparableSsdSlim.input_shape
        target = "ethos-u55-256"
        system_config = "RTSS_HP_SRAM_MRAM"
        output_count = 2
    else:
        classes = selected_imagenet_classes(args.mv2_classes)
        if len(classes) != args.mv2_classes:
            raise RuntimeError("selected ImageNet class list contains duplicates")
        model = make_mv2(args.weights, classes)
        shape = (1, 3, 224, 224)
        target = "ethos-u85-256"
        system_config = "Ethos_U85_SRAM_MRAM"
        output_count = 1
        if args.labels_output:
            categories = MobileNet_V2_Weights.DEFAULT.meta["categories"]
            args.labels_output.parent.mkdir(parents=True, exist_ok=True)
            args.labels_output.write_text(
                "".join(f"{categories[index]}\n" for index in classes)
            )

    compile_spec = EthosUCompileSpec(
        target=target,
        system_config=system_config,
        memory_mode="Shared_Sram",
        config_ini=str(args.config.resolve()),
        extra_flags=["--verbose-operators"],
    )
    example = torch.zeros(shape)
    exported = torch.export.export(model, (example,), strict=True).module()
    remove_unused_export_guards(exported)
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(exported, quantizer)
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for _ in range(args.calibration_samples):
            prepared(torch.rand(shape, generator=generator))
    quantized = convert_pt2e(prepared)
    for node in quantized.graph.nodes:
        if "dequantize_per_tensor" in str(node.target):
            print(f"qparam {node.name}: scale={node.args[1]} zero_point={node.args[2]}")
    quantized_program = torch.export.export(quantized, (example,), strict=True)

    edge = to_edge_transform_and_lower(
        programs=quantized_program,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    graph = edge.exported_program().graph_module.graph
    delegates = sum(
        node.op == "call_function" and "delegate" in str(node.target)
        for node in graph.nodes
    )
    fallbacks = [
        str(node.target)
        for node in graph.nodes
        if node.op == "call_function"
        and "delegate" not in str(node.target)
        and "quantized_decomposed" not in str(node.target)
        and "getitem" not in str(node.target)
    ]
    if delegates != 1 or fallbacks:
        raise RuntimeError(
            f"not fully delegated: delegates={delegates} fallbacks={fallbacks}"
        )

    # Strip the graph-boundary quantize/dequantize operations after lowering.
    # The resulting forward method accepts and returns int8 tensors directly,
    # matching TFLM and avoiding image-sized FP32 conversion on the Cortex-M55.
    edge = edge.transform(
        passes=[
            QuantizeInputs(edge, [0], method_name="forward"),
            QuantizeOutputs(
                edge, list(range(output_count)), method_name="forward"
            ),
        ]
    )

    exported_edge = edge.exported_program()
    placeholders = [
        node for node in exported_edge.graph_module.graph.nodes if node.op == "placeholder"
    ]
    outputs = list(exported_edge.graph_module.graph.output_node().args[0])
    if len(placeholders) != 1 or placeholders[0].meta["val"].dtype != torch.int8:
        raise RuntimeError("QuantizeInputs did not produce an int8 input boundary")
    if len(outputs) != output_count or any(
        output.meta["val"].dtype != torch.int8 for output in outputs
    ):
        raise RuntimeError("QuantizeOutputs did not produce int8 output boundaries")

    executorch_program = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(executorch_program.buffer)

    print(f"model={args.model} target={target} fully_delegated=yes")
    print(
        f"runtime_boundary=input:int8 outputs:{','.join(['int8'] * output_count)} "
        f"quant_config_methods={len(edge._config_methods or {})}"
    )
    print(f"PTE: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
