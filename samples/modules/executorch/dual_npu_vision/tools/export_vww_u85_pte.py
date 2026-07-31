#!/usr/bin/env python3
"""Package a Vela-compiled MicroNet VWW NPZ as an ExecuTorch Ethos-U PTE."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from executorch.backends.arm.ethosu.backend import EthosUBackend
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import PreprocessResult


def pack_io(prefix: str, data: np.lib.npyio.NpzFile) -> bytes:
    shapes = data[f"{prefix}_shape"]
    packed = struct.pack("<i", len(shapes))
    for i, shape in enumerate(shapes):
        if len(shape) != 6:
            raise ValueError(f"{prefix} shape must be 6D: {shape}")
        packed += struct.pack(
            "<iiiiiiiii",
            *[int(v) for v in shape],
            int(data[f"{prefix}_elem_size"][i]),
            int(data[f"{prefix}_offset"][i]),
            int(data[f"{prefix}_region"][i]),
        )
    return packed


def pack_block(name: str, payload: bytes) -> bytes:
    encoded = name.encode()[:15].ljust(16, b"\0")
    header = encoded + struct.pack("<IIII", len(payload), 0, 0, 0)
    return header + payload + b"\0" * ((-len(payload)) % 16)


def make_vela_stream(npz_path: Path) -> bytes:
    with np.load(npz_path, allow_pickle=False) as data:
        blocks = (
            ("vela_bin_stream", b""),
            ("cmd_data", data["cmd_data"].tobytes()),
            ("weight_data", data["weight_data"].tobytes()),
            ("scratch_size", struct.pack("<I", int(data["scratch_shape"][0]))),
            ("inputs", pack_io("input", data)),
            ("outputs", pack_io("output", data)),
            ("vela_end_stream", b""),
        )
        return b"".join(pack_block(name, payload) for name, payload in blocks)


class VwwInterface(nn.Module):
    """Shape/type contract for the precompiled VWW delegate."""

    def forward(self, image: torch.Tensor):
        # Shape/type contract; implementation is replaced by the delegate.
        pooled = image[:, :, :1, :1].reshape(1, 1)
        return pooled.expand(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vela_stream = make_vela_stream(args.npz)
    example = (torch.zeros((1, 1, 128, 128), dtype=torch.int8),)
    edge = to_edge(torch.export.export(VwwInterface().eval(), example, strict=True))

    original_preprocess = EthosUBackend.preprocess
    EthosUBackend.preprocess = staticmethod(
        lambda _program, _specs: PreprocessResult(processed_bytes=vela_stream)
    )
    try:
        lowered = to_backend("EthosUBackend", edge.exported_program(), [])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(lowered.buffer())
    finally:
        EthosUBackend.preprocess = original_preprocess

    print(f"Vela stream: {len(vela_stream)} bytes")
    print(f"PTE: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
