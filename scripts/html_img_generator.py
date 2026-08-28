#!/usr/bin/env python3
"""Print stable Colab image embeds for slides selected by current position."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_BASE_URL = "https://williamtheisen.com/nd-cse-10124-lectures/Lecture_Images"


def load_manifest(base_url: str, manifest_file: Path | None) -> dict:
    if manifest_file:
        return json.loads(manifest_file.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="lecture-manifest-") as temp_dir:
        destination = Path(temp_dir) / "manifest.json"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--output",
                str(destination),
                f"{base_url.rstrip('/')}/manifest.json",
            ],
            check=True,
        )
        return json.loads(destination.read_text(encoding="utf-8"))


def render_embed(url: str) -> str:
    return (
        '<div class="thumbnail">\n'
        f'    <img src="{url}" class="img-responsive"/>\n'
        "</div>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("lecture", "programming-day"))
    parser.add_argument("number", type=int)
    parser.add_argument("positions", type=int, nargs="+", help="Current slide numbers")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--manifest-file", type=Path)
    args = parser.parse_args()

    if args.number < 0 or any(position < 1 for position in args.positions):
        parser.error("Use a nonnegative deck number and positive slide positions.")

    prefix = "Lecture" if args.kind == "lecture" else "ProgrammingDay"
    deck_key = f"{prefix}{args.number:02d}"
    manifest = load_manifest(args.base_url, args.manifest_file)
    try:
        deck = manifest["decks"][deck_key]
        slides = {slide["position"]: slide for slide in deck["slides"]}
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"The deployed manifest has no stable slide data for {deck_key}."
        ) from error

    embeds = []
    for position in args.positions:
        if position not in slides:
            raise ValueError(f"{deck_key} has no slide at position {position}.")
        relative_image = slides[position]["image"]
        embeds.append(render_embed(f"{args.base_url.rstrip('/')}/{relative_image}"))

    print("\n\n---\n\n".join(embeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
