#!/usr/bin/env python3
"""Backend-neutral PyTorch models for TFLM versus ExecuTorch comparisons.

The modules in this file deliberately return raw tensors.  Anchor decoding,
softmax, NMS, and label selection belong in the common firmware so those
operations do not bias either model container/backend.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBnRelu(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.depthwise = ConvBnRelu(
            in_channels, in_channels, 3, stride=stride, groups=in_channels
        )
        self.pointwise = ConvBnRelu(in_channels, out_channels, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(tensor))


class SeparablePredictionHead(nn.Module):
    def __init__(self, channels: int, anchors: int, columns: int) -> None:
        super().__init__()
        self.depthwise = ConvBnRelu(channels, channels, 3, groups=channels)
        self.output = nn.Conv2d(channels, anchors * columns, 1)
        self.columns = columns

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.output(self.depthwise(tensor))
        # Keep anchor order equivalent to the NHWC SSD-Slim artifact while
        # authoring the trainable network in conventional PyTorch NCHW form.
        return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, self.columns)


class DirectPredictionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        anchors: int,
        columns: int,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.output = nn.Conv2d(
            channels,
            anchors * columns,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.columns = columns

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.output(tensor)
        return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, self.columns)


class ComparableSsdSlim(nn.Module):
    """Compact four-scale SSD matching the deployed SSD-Slim I/O contract.

    Input:  ``[N, 1, 120, 160]`` float during training/export calibration.
    Output: raw box deltas ``[N, 1118, 4]`` and class logits
    ``[N, 1118, 2]``.  The feature-map/anchor counts are:

    * 15x20, three anchors per location (900)
    * 8x10, two anchors per location (160)
    * 4x5, two anchors per location (40)
    * 2x3, three anchors per location (18)
    """

    input_shape = (1, 1, 120, 160)
    output_boxes_shape = (1, 1118, 4)
    output_logits_shape = (1, 1118, 2)

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBnRelu(1, 16, 3, stride=2)
        self.stage0 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            DepthwiseSeparableConv(32, 32, stride=2),
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64),
        )
        self.stage1_down = DepthwiseSeparableConv(64, 128, stride=2)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 128),
        )
        self.stage2_down = DepthwiseSeparableConv(128, 256, stride=2)
        self.stage2 = DepthwiseSeparableConv(256, 256)
        self.extra = nn.Sequential(
            ConvBnRelu(256, 64, 1),
            DepthwiseSeparableConv(64, 256, stride=2),
        )

        channels = (64, 128, 256)
        anchors = (3, 2, 2)
        self.box_heads = nn.ModuleList(
            SeparablePredictionHead(c, a, 4) for c, a in zip(channels, anchors)
        )
        self.class_heads = nn.ModuleList(
            SeparablePredictionHead(c, a, 2) for c, a in zip(channels, anchors)
        )
        # The deployed SSD-Slim model uses conventional 3x3 predictors on the
        # final 2x3 feature map (the earlier predictors are separable).
        self.box_extra = DirectPredictionHead(256, 3, 4, kernel_size=3)
        self.class_extra = DirectPredictionHead(256, 3, 2, kernel_size=3)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature0 = self.stage0(self.stem(image))
        feature1 = self.stage1(self.stage1_down(feature0))
        feature2 = self.stage2(self.stage2_down(feature1))
        feature3 = self.extra(feature2)

        features = (feature0, feature1, feature2)
        boxes = [head(feature) for head, feature in zip(self.box_heads, features)]
        logits = [
            head(feature) for head, feature in zip(self.class_heads, features)
        ]
        boxes.append(self.box_extra(feature3))
        logits.append(self.class_extra(feature3))
        return torch.cat(boxes, dim=1), torch.cat(logits, dim=1)


def validate_comparable_ssd() -> None:
    model = ComparableSsdSlim().eval()
    with torch.no_grad():
        boxes, logits = model(torch.zeros(model.input_shape))
    if tuple(boxes.shape) != model.output_boxes_shape:
        raise RuntimeError(f"unexpected box shape: {tuple(boxes.shape)}")
    if tuple(logits.shape) != model.output_logits_shape:
        raise RuntimeError(f"unexpected logit shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    validate_comparable_ssd()
    model = ComparableSsdSlim()
    print(f"ComparableSsdSlim parameters={sum(p.numel() for p in model.parameters())}")
    print(f"input={model.input_shape}")
    print(f"outputs={model.output_boxes_shape}/{model.output_logits_shape}")
