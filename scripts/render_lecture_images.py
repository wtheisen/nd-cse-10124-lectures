#!/usr/bin/env python3
"""Render Google Slides and filled PDFs with stable slide-ID image aliases."""

import argparse
import hashlib
from io import BytesIO
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote, urljoin


DECK_RE = re.compile(
    r"^(Lecture|Programming[ _-]?Day)[ _-]?(\d{1,3})(?:[ _-].*)?$",
    re.IGNORECASE,
)
SLIDE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
NOTABILITY_RENDER_VERSION = 12
APL_FONT_PATH = Path(__file__).resolve().parent / "assets" / "LectureAPL-Regular.ttf.b64"


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
    presentation_last_updated: str
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
        presentation_last_updated=str(payload.get("lastUpdated", "")).strip(),
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


@contextmanager
def google_slides_editor_page(chromium_executable: Optional[str] = None) -> Iterator[object]:
    """Launch one headless editor session for pixel-faithful slide captures."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "Playwright is required for Notability-ready PDFs. "
            "Install it with `python3 -m pip install playwright` and run "
            "`python3 -m playwright install chromium`."
        ) from error

    manager = sync_playwright().start()
    launch_options: dict[str, object] = {"headless": True}
    if chromium_executable:
        launch_options["executable_path"] = chromium_executable
    browser = manager.chromium.launch(**launch_options)
    context = browser.new_context(
        # This is the viewport used by the verified Mac render. At DPR 1 the
        # Slides editor produces a roughly 3178x2384 image for a 4:3 slide.
        viewport={"width": 3400, "height": 2500},
        device_scale_factor=1,
    )
    page = context.new_page()
    try:
        yield page
    finally:
        context.close()
        browser.close()
        manager.stop()


def capture_google_slides_editor_images(
    page: object,
    slide_map: SlideMap,
    out_dir: Path,
) -> list[Path]:
    """Capture the live Slides editor canvas, whose layout matches the authoring view."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    first_slide = slide_map.slides[0]
    first_slide_id = quote(first_slide.object_id, safe="_-")
    first_url = (
        "https://docs.google.com/presentation/d/"
        f"{slide_map.presentation_id}/edit?usp=sharing&rm=minimal"
        f"#slide=id.{first_slide_id}"
    )
    page.goto(first_url, wait_until="domcontentloaded", timeout=120_000)
    use_font_shims = platform.system() != "Darwin"
    if use_font_shims:
        install_export_font_shims(page)
    canvas = page.locator("#canvas")
    canvas.wait_for(state="visible", timeout=120_000)
    page.evaluate("() => document.fonts.ready")

    images: list[Path] = []
    capture_size: Optional[tuple[int, int]] = None
    for index, slide in enumerate(slide_map.slides):
        slide_id = quote(slide.object_id, safe="_-")
        if index:
            page.evaluate(
                "slideId => { window.location.hash = 'slide=id.' + slideId; }",
                slide_id,
            )
        slide_element = page.locator(f'[id="editor-{slide.object_id}"]')
        slide_element.wait_for(
            state="visible", timeout=120_000
        )
        # The editor node becomes visible before Slides inserts all of its SVG
        # text, so let the slide paint before repairing font-dependent layout.
        page.wait_for_timeout(1_000)
        repaired_gradients = (
            apply_export_font_shims(page, slide.object_id) if use_font_shims else 0
        )
        if repaired_gradients:
            print(
                f"{slide_map.deck_id.output_name} slide {slide.position}: "
                f"repaired {repaired_gradients} gradient symbol(s)"
            )
        destination = out_dir / f"slide-{slide.position:03d}.png"
        page.wait_for_timeout(50)
        slide_element.screenshot(path=str(destination), animations="disabled")
        # The editor element sometimes includes objects positioned just beyond
        # the page boundary. Crop that overflow back to the actual page
        # rectangle established by the first slide. Locator screenshots can
        # also differ by a pixel due to fractional CSS bounds, so normalize
        # those tiny undershoots. A materially smaller element still signals a
        # bad node; oversized elements are legitimate off-page authoring data.
        from PIL import Image

        with Image.open(destination) as captured:
            current_size = captured.size
            if capture_size is None:
                capture_size = current_size
            elif current_size != capture_size:
                width_ratio = current_size[0] / capture_size[0]
                height_ratio = current_size[1] / capture_size[1]
                if width_ratio < 0.999 or height_ratio < 0.999:
                    raise RuntimeError(
                        f"{slide_map.deck_id.output_name} slide {slide.position}: "
                        f"unexpected editor capture size {current_size}; "
                        f"expected {capture_size}"
                    )
                crop_width = min(current_size[0], capture_size[0])
                crop_height = min(current_size[1], capture_size[1])
                normalized = captured.crop((0, 0, crop_width, crop_height))
                if normalized.size != capture_size:
                    normalized = normalized.resize(
                        capture_size, Image.Resampling.LANCZOS
                    )
                normalized.save(destination, format="PNG", optimize=True)
        images.append(destination)
    return images


def install_export_font_shims(page: object) -> None:
    """Match the macOS fallback metrics used by the live Slides editor.

    Arial does not contain U+2375 (APL functional symbol omega). macOS falls
    back to Arial Unicode MS, while Ubuntu picks a wider glyph and wraps it in
    narrow formula boxes. This tiny OFL-licensed, renamed Noto Sans Math subset
    supplies only U+2375 with the same 600-unit advance as Arial Unicode MS.
    """
    try:
        encoded_font = APL_FONT_PATH.read_text(encoding="ascii").strip()
    except OSError as error:
        raise RuntimeError(f"Cannot load export font shim: {APL_FONT_PATH}") from error
    page.add_style_tag(
        content=(
            "@font-face {"
            "font-family: 'Lecture APL';"
            f"src: url(data:font/ttf;base64,{encoded_font}) format('truetype');"
            "font-style: normal; font-weight: 400; font-display: block;"
            "unicode-range: U+2375;"
            "}"
        )
    )
    page.evaluate("() => document.fonts.load('67px \\\"Lecture APL\\\"', '\u2375')")
    page.evaluate("() => document.fonts.ready")


def apply_export_font_shims(page: object, slide_object_id: str) -> int:
    """Apply the metric-compatible glyph and undo Slides' stale line wrap."""
    repaired = page.evaluate(
        """slideObjectId => {
            const slideId = `editor-${slideObjectId}`;
            const glyphs = Array.from(
                document.querySelectorAll(`#${CSS.escape(slideId)} text`)
            );
            const nablas = glyphs.filter(element =>
                element.textContent === '\u2207' &&
                Number.parseFloat(getComputedStyle(element).fontSize) >= 45
            );
            let repairs = 0;
            for (const omega of glyphs) {
                if (omega.textContent !== '\u2375') continue;
                omega.style.setProperty(
                    'font-family',
                    "'Lecture APL', Arial, sans-serif",
                    'important'
                );

                if (Number.parseFloat(getComputedStyle(omega).fontSize) < 60) continue;
                const omegaRect = omega.getBoundingClientRect();
                const omegaCenter = {
                    x: omegaRect.x + omegaRect.width / 2,
                    y: omegaRect.y + omegaRect.height / 2,
                };
                const nabla = nablas
                    .map(element => {
                        const rect = element.getBoundingClientRect();
                        const dx = rect.x + rect.width / 2 - omegaCenter.x;
                        const dy = rect.y + rect.height / 2 - omegaCenter.y;
                        return {element, distance: dx * dx + dy * dy};
                    })
                    .sort((left, right) => left.distance - right.distance)[0];
                if (!nabla || nabla.distance > 40000) continue;

                // Ubuntu may have already laid these out as separate lines
                // before the web font is available. Put omega on nabla's
                // baseline and recompute its horizontal position directly.
                const nablaElement = nabla.element;
                const nablaRect = nablaElement.getBoundingClientRect();
                const originalCenterY = (
                    nablaRect.y + nablaRect.height / 2 + omegaCenter.y
                ) / 2;
                const nablaX = Number.parseFloat(nablaElement.getAttribute('x') || '0');
                const omegaX = nablaX + nablaElement.getComputedTextLength();
                omega.setAttribute('x', String(omegaX));
                omega.removeAttribute('y');
                const nablaParent = nablaElement.parentElement;
                const omegaParent = omega.parentElement;
                if (omegaParent !== nablaParent) {
                    const nablaTransform = nablaParent.transform.baseVal
                        .consolidate()?.matrix;
                    const outerTransform = nablaParent.parentElement?.getScreenCTM();
                    if (nablaTransform && outerTransform) {
                        // The line separation is not necessarily encoded in the
                        // two line-group matrices. Center the repaired pair at
                        // the midpoint of the glyphs' actual pre-repair screen
                        // positions, which is the center Slides intended for
                        // the original one-line text box.
                        nablaParent.appendChild(omega);
                        const repairedNablaRect = nablaElement.getBoundingClientRect();
                        const repairedOmegaRect = omega.getBoundingClientRect();
                        const repairedTop = Math.min(
                            repairedNablaRect.top,
                            repairedOmegaRect.top
                        );
                        const repairedBottom = Math.max(
                            repairedNablaRect.bottom,
                            repairedOmegaRect.bottom
                        );
                        const screenDeltaY = originalCenterY - (
                            repairedTop + repairedBottom
                        ) / 2;
                        const parentDeltaY = screenDeltaY / outerTransform.d;
                        nablaParent.setAttribute(
                            'transform',
                            `matrix(${nablaTransform.a} ${nablaTransform.b} ` +
                            `${nablaTransform.c} ${nablaTransform.d} ` +
                            `${nablaTransform.e} ${nablaTransform.f + parentDeltaY})`
                        );
                    } else {
                        nablaParent.appendChild(omega);
                    }
                }
                repairs += 1;
            }
            return repairs;
        }""",
        slide_object_id,
    )
    page.evaluate("() => document.fonts.ready")
    return int(repaired)


def matching_previous_notability_entry(
    deck_id: DeckId,
    live_map: SlideMap,
    previous_manifest: object,
) -> Optional[dict[str, object]]:
    """Return a reusable export only when its source deck is provably unchanged."""
    if not live_map.presentation_last_updated or not isinstance(previous_manifest, dict):
        return None
    decks = previous_manifest.get("decks")
    if not isinstance(decks, dict):
        return None
    entry = decks.get(deck_id.output_name)
    if not isinstance(entry, dict):
        return None
    if entry.get("presentation_id") != live_map.presentation_id:
        return None
    if entry.get("presentation_last_updated") != live_map.presentation_last_updated:
        return None
    if entry.get("notability_render_version") != NOTABILITY_RENDER_VERSION:
        return None

    previous_ids = entry.get("notability_source_slide_ids")
    if not isinstance(previous_ids, list):
        # Backward-compatible fallback for manifests written before the live
        # Notability IDs were stored separately from filled-handout IDs.
        previous_slides = entry.get("slides")
        if not isinstance(previous_slides, list):
            return None
        previous_ids = [
            str(slide.get("id", ""))
            for slide in previous_slides
            if isinstance(slide, dict)
        ]
    if previous_ids != [slide.object_id for slide in live_map.slides]:
        return None

    required = (
        "notability_pdf",
        "notability_md5",
        "notability_size_bytes",
        "notability_slide_count",
        "notability_render_version",
    )
    if any(not entry.get(field) for field in required):
        return None
    return entry


def reuse_previous_notability_pdf(
    entry: dict[str, object],
    manifest_url: str,
    destination: Path,
    curl: str,
    pdfinfo: str,
) -> dict[str, object]:
    """Download and validate a prior immutable export before reusing it."""
    relative_path = str(entry["notability_pdf"])
    download_file(
        urljoin(manifest_url, relative_path),
        destination,
        curl,
        f"previous Notability PDF {destination.name}",
    )
    expected_size = int(entry["notability_size_bytes"])
    expected_md5 = str(entry["notability_md5"])
    expected_slides = int(entry["notability_slide_count"])
    if destination.stat().st_size != expected_size:
        raise RuntimeError(f"Previous Notability PDF has the wrong size: {destination.name}")
    if file_md5(destination) != expected_md5:
        raise RuntimeError(f"Previous Notability PDF has the wrong checksum: {destination.name}")
    with destination.open("rb") as pdf:
        if pdf.read(5) != b"%PDF-":
            raise RuntimeError(f"Previous Notability export is not a PDF: {destination.name}")
    if pdf_page_count(destination, pdfinfo) != expected_slides:
        raise RuntimeError(f"Previous Notability PDF has the wrong page count: {destination.name}")
    return {
        field: entry[field]
        for field in (
            "notability_pdf",
            "notability_md5",
            "notability_size_bytes",
            "notability_slide_count",
            "notability_render_version",
        )
    }


def create_raster_pdf(images: list[Path], destination: Path) -> None:
    """Create an image-only PDF so downstream apps cannot reflow slide text."""
    if not images:
        raise ValueError("Cannot create a raster PDF without slide images.")
    try:
        from PIL import Image
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as error:
        raise RuntimeError(
            "Pillow and reportlab are required for Notability-ready PDFs."
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(images[0]) as first_image:
        first_width, first_height = first_image.size
    page_width = 10 * 72
    source_aspect = first_width / first_height
    page_aspect = 4 / 3 if abs(source_aspect - (4 / 3)) < 0.01 else source_aspect
    page_height = page_width / page_aspect
    pdf = canvas.Canvas(str(destination), pagesize=(page_width, page_height))
    pdf.setTitle(destination.stem)
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
            rgb_image = image.convert("RGB")
            encoded_image = BytesIO()
            rgb_image.save(
                encoded_image,
                format="JPEG",
                quality=92,
                subsampling=0,
                optimize=True,
            )
            encoded_image.seek(0)
        if abs((width / height) - (first_width / first_height)) > 0.001:
            raise RuntimeError(f"Slide image has inconsistent aspect ratio: {image_path}")
        pdf.drawImage(
            ImageReader(encoded_image),
            0,
            0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=True,
            anchor="c",
        )
        pdf.showPage()
    pdf.save()


def file_md5(path: Path) -> str:
    """Return the content digest Google Drive exposes for binary files."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--previous-output-manifest-url")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdftoppm")
    parser.add_argument("--pdfinfo")
    parser.add_argument("--curl")
    parser.add_argument("--chromium-executable")
    parser.add_argument(
        "--skip-notability-pdfs",
        action="store_true",
        help="Skip browser captures and raster-backed PDF generation.",
    )
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
    missing_filled_maps = sorted(set(filled_decks) - set(filled_slide_maps))
    if missing_filled_maps:
        missing_names = ", ".join(
            f"{deck_id.output_name}.slide-map.json" for deck_id in missing_filled_maps
        )
        raise RuntimeError(
            "Every filled PDF requires the frozen slide-ID map created with it. "
            f"Missing: {missing_names}. Re-upload the filled slides before rerunning."
        )
    manifest: dict[str, object] = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": os.environ.get("GITHUB_SHA", ""),
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
        previous_manifest: object = {}
        if args.previous_output_manifest_url:
            previous_manifest_path = temp_root / "previous-output-manifest.json"
            download_file(
                args.previous_output_manifest_url,
                previous_manifest_path,
                curl,
                "previous output manifest",
            )
            previous_manifest = read_json(previous_manifest_path)
        all_decks = sorted(set(slide_decks) | set(filled_decks) | set(live_slide_maps))
        if not all_decks:
            raise RuntimeError(
                "No renderable Google Slides presentations or filled PDFs were found."
            )
        staged_output = temp_root / "Lecture_Images"
        staged_output.mkdir()

        editor_captures: dict[DeckId, list[Path]] = {}
        notability_exports: dict[DeckId, dict[str, object]] = {}
        if not args.skip_notability_pdfs:
            notability_dir = staged_output / "Notability_PDFs"
            notability_dir.mkdir(parents=True)
            with google_slides_editor_page(args.chromium_executable) as editor_page:
                for deck_id, live_map in sorted(live_slide_maps.items()):
                    pdf_name = f"{deck_id.output_name}_Notability.pdf"
                    pdf_path = notability_dir / pdf_name
                    previous_entry = matching_previous_notability_entry(
                        deck_id, live_map, previous_manifest
                    )
                    if previous_entry and args.previous_output_manifest_url:
                        print(f"{deck_id.output_name}: reusing unchanged Notability PDF")
                        notability_exports[deck_id] = reuse_previous_notability_pdf(
                            previous_entry,
                            args.previous_output_manifest_url,
                            pdf_path,
                            curl,
                            pdfinfo,
                        )
                        notability_exports[deck_id]["notability_source_slide_ids"] = [
                            slide.object_id for slide in live_map.slides
                        ]
                        if deck_id not in filled_decks:
                            # Live-only decks also use the browser-faithful export
                            # for their website images. Re-rasterizing at the
                            # original 3300px width avoids the vector PDF fallback.
                            capture_dir = temp_root / "editor-captures" / deck_id.output_name
                            editor_captures[deck_id] = render_pdf_to_images(
                                pdf_path, capture_dir, pdftoppm, 330
                            )
                        continue
                    print(f"{deck_id.output_name}: capturing editor canvas for Notability")
                    capture_dir = temp_root / "editor-captures" / deck_id.output_name
                    captures = capture_google_slides_editor_images(
                        editor_page,
                        live_map,
                        capture_dir,
                    )
                    editor_captures[deck_id] = captures
                    if len(captures) != len(live_map.slides):
                        raise RuntimeError(
                            f"{deck_id.output_name}: expected {len(live_map.slides)} "
                            f"editor captures, created {len(captures)}"
                        )
                    create_raster_pdf(captures, pdf_path)
                    notability_exports[deck_id] = {
                        "notability_pdf": f"Notability_PDFs/{pdf_name}",
                        "notability_md5": file_md5(pdf_path),
                        "notability_size_bytes": pdf_path.stat().st_size,
                        "notability_slide_count": len(captures),
                        "notability_render_version": NOTABILITY_RENDER_VERSION,
                        "notability_source_slide_ids": [
                            slide.object_id for slide in live_map.slides
                        ],
                    }

        for deck_id in all_decks:
            filled_pdf = filled_decks.get(deck_id)
            pointer = slide_decks.get(deck_id)
            live_map = live_slide_maps.get(deck_id)
            frozen_map = filled_slide_maps.get(deck_id)

            if filled_pdf:
                # Missing maps are rejected before rendering so a filled PDF can
                # never be paired with IDs from a later version of the live deck.
                assert frozen_map is not None
                slide_map = frozen_map
                map_type = "filled_snapshot"
            elif live_map:
                slide_map = live_map
                map_type = "live_google_slides"
            else:
                raise RuntimeError(
                    f"{deck_id.output_name}: no stable slide-ID map is available."
                )

            presentation_id = (
                read_presentation_id(pointer) if pointer else slide_map.presentation_id
            )

            if pointer and slide_map.presentation_id != presentation_id:
                raise RuntimeError(
                    f"{deck_id.output_name}: slide map presentation ID does not match {pointer.name}."
                )

            if filled_pdf:
                source_pdf = filled_pdf
                source_type = "filled_pdf"
                source_name = filled_pdf.name
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
            elif deck_id in editor_captures:
                source_type = "google_slides_editor"
                source_name = (
                    pointer.name
                    if pointer
                    else f"{deck_id.output_name} (Drive discovery)"
                )
                page_count = len(slide_map.slides)
                deck_dir = staged_output / deck_id.output_name
                deck_dir.mkdir(parents=True)
                numbered_images = []
                for source_image, slide in zip(
                    editor_captures[deck_id], slide_map.slides, strict=True
                ):
                    destination = deck_dir / f"slide-{slide.position:03d}.png"
                    shutil.copy2(source_image, destination)
                    numbered_images.append(destination)
            else:
                source_pdf = temp_root / f"{deck_id.output_name}.pdf"
                source_name = (
                    pointer.name
                    if pointer
                    else f"{deck_id.output_name} (Drive discovery)"
                )
                print(f"{deck_id.output_name}: exporting {source_name}")
                export_google_slides_pdf(str(presentation_id), source_pdf, curl)
                source_type = "google_slides"
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
                "presentation_last_updated": (
                    live_map.presentation_last_updated if live_map else ""
                ),
                "source_type": source_type,
                "source_name": source_name,
                "slide_map_type": map_type,
                "slide_count": len(numbered_images),
                "slides": stable_slides,
            }
            if deck_id in notability_exports:
                manifest["decks"][deck_id.output_name].update(
                    notability_exports[deck_id]
                )

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
