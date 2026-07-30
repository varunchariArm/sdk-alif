#!/usr/bin/env python3
"""Concatenate fixed-layout assets into the SRAM0 image loaded by SETools."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with args.output.open("wb") as output:
        for source in args.inputs:
            data = source.read_bytes()
            output.write(data)
            print(f"{source.name}: offset=0x{offset:x} size={len(data)}")
            offset += len(data)
    print(f"{args.output}: total={offset} (0x{offset:x})")


if __name__ == "__main__":
    main()
