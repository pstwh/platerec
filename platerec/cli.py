import argparse
import os
from typing import List

from PIL import Image

from platerec import Platerec


def get_args():
    parser = argparse.ArgumentParser(description="Read license plates images.")

    parser.add_argument(
        "image_paths",
        type=str,
        nargs="+",
        help="Path to the image you want to detect license plates in.",
    )

    parser.add_argument(
        "--encoder_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "artifacts", "encoder.onnx"),
        help="Path to the ONNX encoder model (default: artifacts/encoder.onnx).",
    )

    parser.add_argument(
        "--return_type",
        type=str,
        default="word",
        choices=["word", "char"],
        help="Return types for inference.",
    )

    parser.add_argument(
        "--decoder_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "artifacts", "decoder.onnx"),
        help="Path to the ONNX decoder model (default: artifacts/decoder.onnx).",
    )

    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Execution provider for ONNX Runtime (default: CPUExecutionProvider).",
    )

    parser.add_argument(
        "--platedet", action="store_true", help="Use platedet to detect plates first."
    )

    return parser.parse_args()


def main():
    args = get_args()
    platerec = Platerec(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        providers=args.providers,
    )

    fn = platerec.detect_read if args.platedet else platerec.read

    for image_path in args.image_paths:
        output = fn(Image.open(image_path).convert("RGB"), return_type=args.return_type)
        print(output)


if __name__ == "__main__":
    main()
