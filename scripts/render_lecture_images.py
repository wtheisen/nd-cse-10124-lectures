#!/usr/bin/env python3
"""Render Google Slides and filled PDFs with stable slide-ID image aliases."""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DECK_RE = re.compile(
    r"^(Lecture|Programming[ _-]?Day)[ _-]?(\d{1,3})(?:[ _-].*)?$",
    re.IGNORECASE,
)
SLIDE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


@dataclass(frozen=True, order=True)
class DeckId:
    kind: str
    number: str

    @property
    def output_name(self) -> str:
        return f"{self.kind}{self.number}"


@dataclass(frozen=True)
class SlideInfo:
    position: int
    object_id: str
    title: str


@dataclass(frozen=True)
class SlideMap:
    deck_id: DeckId
    presentation_id: str
    slides: tuple[SlideInfo, ...]
    source: str


def deck_id_from_stem(stem: str) -> Optional[DeckId]:
    match = DECK_RE.match(stem.strip())
    if not match:
        return None
    kind = "Lecture" if match.group(1).lower() == "lecture" else "ProgrammingDay"
    return DeckId(kind=kind, number=match.group(2).zfill(2))


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error


def read_presentation_id(pointer: Path) -> str:
    payload = read_json(pointer)
    if not isinstance(payload, dict):
        raise ValueError(f"Google Slides pointer is not a JSON object: {pointer}")
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


def download_file(url: str, destination: Path, curl: str, label: str) -> None:
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
        raise RuntimeError(f"Could not download {label}: {url}") from error


def export_google_slides_pdf(
    presentation_id: str, destination: Path, curl: str
) -> None:
    url = f"https://docs.google.com/presentation/d/{presentation_id}/export/pdf"
    try:
        download_file(url, destination, curl, f"Google Slides presentation {presentation_id}")
    except RuntimeError as error:
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


def parse_slide_map(
    payload: object,
    source: str,
    expected_deck: Optional[DeckId] = None,
) -> SlideMap:
    if not isinstance(payload, dict):
        raise ValueError(f"Slide map from {source} is not a JSON object.")

    deck_name = str(payload.get("deck", "")).strip()
    deck_id = deck_id_from_stem(deck_name)
    if not deck_id or deck_id.output_name != deck_name:
        raise ValueError(f"Slide map from {source} has an invalid deck key: {deck_name!r}")
    if expected_deck and deck_id != expected_deck:
        raise ValueError(
            f"Slide map from {source} claims {deck_id.output_name}, "
            f"expected {expected_deck.output_name}."
        )

    presentation_id = str(payload.get("presentationId", "")).strip()
    if not presentation_id:
        raise ValueError(f"Slide map for {deck_name} has no presentationId: {source}")

    raw_slides = payload.get("slides")
    if not isinstance(raw_slides, list) or not raw_slides:
        raise ValueError(f"Slide map for {deck_name} has no slides: {source}")

    slides: list[SlideInfo] = []
    seen_ids: set[str] = set()
    for index, raw_slide in enumerate(raw_slides, start=1):
        if not isinstance(raw_slide, dict):
            raise ValueError(f"Slide {index} in {source} is not an object.")
        position = raw_slide.get("position")
        object_id = str(raw_slide.get("id", "")).strip()
        title = str(raw_slide.get("title", "")).strip()
        if position != index:
            raise ValueError(
                f"Slide positions in {source} must be contiguous; expected {index}, got {position}."
            )
        if not SLIDE_ID_RE.fullmatch(object_id):
            raise ValueError(f"Slide {index} in {source} has an unsafe object ID: {object_id!r}")
        if object_id in seen_ids:
            raise ValueError(f"Slide map in {source} repeats object ID {object_id}.")
        seen_ids.add(object_id)
        slides.append(SlideInfo(position=index, object_id=object_id, title=title))

    return SlideMap(
        deck_id=deck_id,
        presentation_id=presentation_id,
        slides=tuple(slides),
        source=source,
    )


def index_live_slide_maps(payload: object, source: str) -> dict[DeckId, SlideMap]:
    if not isinstance(payload, dict) or not isinstance(payload.get("decks"), dict):
        raise ValueError(f"Live slide manifest from {source} has no decks object.")
    indexed: dict[DeckId, SlideMap] = {}
    for deck_name, raw_map in sorted(payload["decks"].items()):
        if not isinstance(raw_map, dict):
            raise ValueError(f"Live slide manifest entry {deck_name} is not an object.")
        expanded = dict(raw_map)
        expanded.setdefault("deck", deck_name)
        slide_map = parse_slide_map(expanded, source)
        if slide_map.deck_id in indexed:
            raise ValueError(f"Live slide manifest repeats {slide_map.deck_id.output_name}.")
        indexed[slide_map.deck_id] = slide_map
    return indexed


def index_filled_slide_maps(filled_dir: Optional[Path]) -> dict[DeckId, SlideMap]:
    indexed: dict[DeckId, SlideMap] = {}
    paths = list(filled_dir.rglob("*.slide-map.json")) if filled_dir else []
    for path in sorted(paths):
        slide_map = parse_slide_map(read_json(path), str(path))
        if slide_map.deck_id in indexed:
            raise ValueError(
                f"Multiple filled slide maps claim {slide_map.deck_id.output_name}: "
                f"{indexed[slide_map.deck_id].source}, {path}"
            )
        indexed[slide_map.deck_id] = slide_map
    return indexed


def load_live_slide_maps(
    manifest_file: Optional[Path],
    manifest_url: Optional[str],
    curl: str,
    temp_root: Path,
) -> dict[DeckId, SlideMap]:
    if bool(manifest_file) == bool(manifest_url):
        raise ValueError("Provide exactly one of --slide-manifest-file or --slide-manifest-url.")
    if manifest_file:
        return index_live_slide_maps(read_json(manifest_file), str(manifest_file))

    downloaded = temp_root / "live-slide-manifest.json"
    download_file(str(manifest_url), downloaded, curl, "live slide manifest")
    return index_live_slide_maps(read_json(downloaded), str(manifest_url))


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


def render_pdf_to_images(
    pdf: Path, out_dir: Path, pdftoppm: str, dpi: int
) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf), str(out_dir / "slide")],
        check=True,
    )

    images = sorted(out_dir.glob("slide-*.png"))
    renamed: list[Path] = []
    for image in images:
        match = re.search(r"-(\d+)\.png$", image.name)
        if not match:
            continue
        destination = out_dir / f"slide-{int(match.group(1)):03d}.png"
        image.rename(destination)
        renamed.append(destination)
    return renamed


def create_stable_aliases(
    deck_dir: Path,
    numbered_images: list[Path],
    slide_map: SlideMap,
) -> list[dict[str, object]]:
    stable_dir = deck_dir / "by-id"
    stable_dir.mkdir()
    manifest_slides: list[dict[str, object]] = []
    for image, slide in zip(numbered_images, slide_map.slides, strict=True):
        stable_name = f"{slide.object_id}.png"
        shutil.copy2(image, stable_dir / stable_name)
        manifest_slides.append(
            {
                "position": slide.position,
                "id": slide.object_id,
                "title": slide.title,
                "image": f"{slide_map.deck_id.output_name}/by-id/{stable_name}",
                "numbered_image": (
                    f"{slide_map.deck_id.output_name}/slide-{slide.position:03d}.png"
                ),
            }
        )
    return manifest_slides


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
    parser.add_argument("--slide-manifest-file", type=Path)
    parser.add_argument("--slide-manifest-url")
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
    filled_slide_maps = index_filled_slide_maps(args.filled_dir)
    all_decks = sorted(set(slide_decks) | set(filled_decks))

    if not all_decks:
        raise RuntimeError("No renderable Google Slides pointers or filled PDFs were found.")

    manifest: dict[str, object] = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decks": {},
    }
    with tempfile.TemporaryDirectory(prefix="lecture-images-") as temp_dir:
        temp_root = Path(temp_dir)
        live_slide_maps = load_live_slide_maps(
            args.slide_manifest_file,
            args.slide_manifest_url,
            curl,
            temp_root,
        )
        staged_output = temp_root / "Lecture_Images"
        staged_output.mkdir()

        for deck_id in all_decks:
            filled_pdf = filled_decks.get(deck_id)
            pointer = slide_decks.get(deck_id)
            presentation_id = read_presentation_id(pointer) if pointer else None
            live_map = live_slide_maps.get(deck_id)
            frozen_map = filled_slide_maps.get(deck_id)

            if filled_pdf and frozen_map:
                slide_map = frozen_map
                map_type = "filled_snapshot"
            elif live_map:
                slide_map = live_map
                map_type = "live_google_slides"
                if filled_pdf:
                    print(
                        f"{deck_id.output_name}: no frozen slide map yet; "
                        "using the current Google Slides order"
                    )
            else:
                raise RuntimeError(
                    f"{deck_id.output_name}: no stable slide-ID map is available."
                )

            if presentation_id and slide_map.presentation_id != presentation_id:
                raise RuntimeError(
                    f"{deck_id.output_name}: slide map presentation ID does not match {pointer.name}."
                )

            if filled_pdf:
                source_pdf = filled_pdf
                source_type = "filled_pdf"
                source_name = filled_pdf.name
            elif pointer:
                source_pdf = temp_root / f"{deck_id.output_name}.pdf"
                print(f"{deck_id.output_name}: exporting {pointer.name}")
                export_google_slides_pdf(str(presentation_id), source_pdf, curl)
                source_type = "google_slides"
                source_name = pointer.name
            else:
                raise AssertionError(f"No source for {deck_id.output_name}")

            page_count = pdf_page_count(source_pdf, pdfinfo)
            if len(slide_map.slides) != page_count:
                raise RuntimeError(
                    f"{deck_id.output_name}: PDF has {page_count} pages but its stable "
                    f"slide map has {len(slide_map.slides)} entries ({slide_map.source})."
                )

            deck_dir = staged_output / deck_id.output_name
            numbered_images = render_pdf_to_images(
                source_pdf,
                deck_dir,
                pdftoppm,
                args.dpi,
            )
            if len(numbered_images) != page_count or not numbered_images:
                raise RuntimeError(
                    f"{deck_id.output_name}: expected {page_count} images, "
                    f"rendered {len(numbered_images)}"
                )

            stable_slides = create_stable_aliases(deck_dir, numbered_images, slide_map)
            print(
                f"{deck_id.output_name}: rendered {len(numbered_images)} slides "
                f"from {source_type} ({source_name})"
            )
            manifest["decks"][deck_id.output_name] = {
                "presentation_id": slide_map.presentation_id,
                "source_type": source_type,
                "source_name": source_name,
                "slide_map_type": map_type,
                "slide_count": len(numbered_images),
                "slides": stable_slides,
            }

        total_images = len(list(staged_output.glob("*/slide-*.png")))
        stable_images = len(list(staged_output.glob("*/by-id/*.png")))
        if total_images == 0 or stable_images != total_images:
            raise RuntimeError(
                "Rendering did not produce matching numbered and stable PNG sets; "
                "refusing to deploy."
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

    print(
        f"Rendered {total_images} numbered and {stable_images} stable images "
        f"across {len(manifest['decks'])} decks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
