#!/usr/bin/env python3
"""Render public Google Slides and filled-slide PDFs for notebook embeds."""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DECK_RE = re.compile(
    r"^(Lecture|Programming[ _-]?Day)[ _-]?(\d{1,3})(?:[ _-].*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class DeckId:
    kind: str
    number: str

    @property
    def output_name(self) -> str:
        return f"{self.kind}{self.number}"


def deck_id_from_stem(stem: str) -> Optional[DeckId]:
    match = DECK_RE.match(stem.strip())
    if not match:
        return None
    kind = "Lecture" if match.group(1).lower() == "lecture" else "ProgrammingDay"
    return DeckId(kind=kind, number=match.group(2).zfill(2))


def read_presentation_id(pointer: Path) -> str:
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Google Slides pointer {pointer}: {error}") from error

    presentation_id = str(payload.get("doc_id", "")).strip()
    if not presentation_id:
        raise ValueError(f"Google Slides pointer has no doc_id: {pointer}")
    return presentation_id


def index_decks(paths: list[Path], label: str) -> dict[DeckId, Path]:
    indexed: dict[DeckId, Path] = {}
    for path in sorted(paths):
        deck_id = deck_id_from_stem(path.stem)
        if deck_id is None:
            print(f"Ignoring {label} with unrecognized name: {path.name}")
            continue
        if deck_id in indexed:
            raise ValueError(
                f"Multiple {label} files claim {deck_id.output_name}: "
                f"{indexed[deck_id].name}, {path.name}"
            )
        indexed[deck_id] = path
    return indexed


def export_google_slides_pdf(
    presentation_id: str, destination: Path, curl: str
) -> None:
    url = f"https://docs.google.com/presentation/d/{presentation_id}/export/pdf"
    try:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "120",
                "--output",
                str(destination),
                url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Google Slides export failed for {presentation_id}. "
            "Share the deck as 'Anyone with the link' (Viewer), then rerun the workflow."
        ) from error

    with destination.open("rb") as exported:
        if exported.read(5) != b"%PDF-":
            raise RuntimeError(
                f"Google returned something other than a PDF for {presentation_id}. "
                "Confirm that the deck is publicly viewable by link."
            )


def pdf_page_count(pdf: Path, pdfinfo: str) -> int:
    result = subprocess.run(
        [pdfinfo, str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not determine page count for {pdf}")
    return int(match.group(1))


def render_pdf_to_images(pdf: Path, out_dir: Path, pdftoppm: str, dpi: int) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf), str(out_dir / "slide")],
        check=True,
    )

    images = sorted(out_dir.glob("slide-*.png"))
    for image in images:
        match = re.search(r"-(\d+)\.png$", image.name)
        if match:
            image.rename(out_dir / f"slide-{int(match.group(1)):03d}.png")
    return len(images)


def find_binary(explicit: Optional[str], name: str) -> str:
    candidate = explicit or shutil.which(name)
    if not candidate:
        raise FileNotFoundError(f"Required executable not found: {name}")
    return candidate


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-dir", type=Path, default=repo_root / "Slides")
    parser.add_argument("--filled-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "Lecture_Images")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdftoppm")
    parser.add_argument("--pdfinfo")
    parser.add_argument("--curl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdftoppm = find_binary(args.pdftoppm, "pdftoppm")
    pdfinfo = find_binary(args.pdfinfo, "pdfinfo")
    curl = find_binary(args.curl, "curl")

    if not args.slides_dir.is_dir():
        raise FileNotFoundError(f"Slides directory does not exist: {args.slides_dir}")

    slide_decks = index_decks(list(args.slides_dir.glob("*.gslides")), "Google Slides")
    filled_paths = list(args.filled_dir.rglob("*.pdf")) if args.filled_dir else []
    filled_decks = index_decks(filled_paths, "filled PDF")
    all_decks = sorted(set(slide_decks) | set(filled_decks))

    if not all_decks:
        raise RuntimeError("No renderable Google Slides pointers or filled PDFs were found.")

    manifest: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="lecture-images-") as temp_dir:
        temp_root = Path(temp_dir)
        staged_output = temp_root / "Lecture_Images"
        staged_output.mkdir()
        for deck_id in all_decks:
            filled_pdf = filled_decks.get(deck_id)
            pointer = slide_decks.get(deck_id)

            if filled_pdf:
                source_pdf = filled_pdf
                source_type = "filled_pdf"
                source_name = filled_pdf.name
            elif pointer:
                source_pdf = temp_root / f"{deck_id.output_name}.pdf"
                presentation_id = read_presentation_id(pointer)
                print(f"{deck_id.output_name}: exporting {pointer.name}")
                export_google_slides_pdf(presentation_id, source_pdf, curl)
                source_type = "google_slides"
                source_name = pointer.name
            else:
                raise AssertionError(f"No source for {deck_id.output_name}")

            page_count = pdf_page_count(source_pdf, pdfinfo)
            rendered_count = render_pdf_to_images(
                source_pdf,
                staged_output / deck_id.output_name,
                pdftoppm,
                args.dpi,
            )
            if rendered_count != page_count or rendered_count == 0:
                raise RuntimeError(
                    f"{deck_id.output_name}: expected {page_count} images, rendered {rendered_count}"
                )

            print(
                f"{deck_id.output_name}: rendered {rendered_count} slides "
                f"from {source_type} ({source_name})"
            )
            manifest[deck_id.output_name] = {
                "source_type": source_type,
                "source_name": source_name,
                "slide_count": rendered_count,
            }

        total_images = len(list(staged_output.glob("*/slide-*.png")))
        if total_images == 0:
            raise RuntimeError(
                "Rendering completed without producing any PNG files; refusing to deploy."
            )

        (staged_output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Replace the published tree only after every deck has rendered and
        # validated, so a bad share setting cannot erase known-good output.
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_output), str(args.output_dir))

    print(f"Rendered {total_images} images across {len(manifest)} decks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
