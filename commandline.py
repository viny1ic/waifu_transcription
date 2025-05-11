#!/usr/bin/env python3
import argparse
from pathlib import Path

from transcription_engine import TranscriptionEngine


def main():
    parser = argparse.ArgumentParser(
        description="CLI driver for TranscriptionEngine"
    )
    parser.add_argument(
        "--waifu", action="store_true",
        help="Print a random ASCII‐waifu at startup"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show per‐language debug output (word counts & phrases)"
    )
    parser.add_argument(
        "--big", action="store_true",
        help="Use the larger, high-accuracy server-grade models"
    )
    args = parser.parse_args()

    # Instantiate and run the engine
    base_dir = Path(__file__).parent
    engine = TranscriptionEngine(
        base_dir=base_dir,
        waifu=args.waifu,
        debug=args.debug,
        big=args.big
    )
    engine.run()


if __name__ == "__main__":
    main()
