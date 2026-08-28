#!/usr/bin/env python3
"""Print Colab-compatible HTML image embeds for a rendered deck."""

import argparse


DEFAULT_BASE_URL = "https://williamtheisen.com/nd-cse-10124-lectures/Lecture_Images"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("lecture", "programming-day"))
    parser.add_argument("number", type=int)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int, help="Last slide number, inclusive")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    if args.number < 0 or args.start < 1 or args.end < args.start:
        parser.error("Use a nonnegative deck number and a valid positive slide range.")

    prefix = "Lecture" if args.kind == "lecture" else "ProgrammingDay"
    deck_url = f"{args.base_url.rstrip('/')}/{prefix}{args.number:02d}"
    for slide_number in range(args.start, args.end + 1):
        print(
            '<div class="thumbnail">\n'
            f'    <img src="{deck_url}/slide-{slide_number:03d}.png" '
            'class="img-responsive"/>\n'
            "</div>"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
