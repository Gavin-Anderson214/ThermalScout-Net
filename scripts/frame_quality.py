"""Inspect grayscale brightness on a saved drone image using display pixel values."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def frame_quality(path):
    with Image.open(path) as source:
        if source.mode != "L":
            raise ValueError(f"Expected an 8-bit grayscale image, found {source.mode}")
        pixels = np.asarray(source)
        width, height = source.size

    low, high = np.percentile(pixels, [5, 95])
    return {
        "width_px": width,
        "height_px": height,
        "brightness_p05": float(low),
        "brightness_p95": float(high),
        "central_brightness_span": float(high - low),
        "percent_at_black": float(100 * np.mean(pixels == 0)),
        "percent_at_white": float(100 * np.mean(pixels == 255)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(frame_quality(args.image), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{args.image}: {error}\n")


if __name__ == "__main__":
    main()
