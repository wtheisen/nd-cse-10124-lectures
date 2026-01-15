#!/usr/bin/env python3
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def lecture_id_from_stem(stem: str) -> Optional[str]:
    match = re.search(r"Lecture[ _-]?(\d+)", stem, re.IGNORECASE)
    if not match:
        return None
    number = match.group(1)
    width = max(2, len(number))
    return number.zfill(width)


def find_pdf_override(filled_dir: Path, lecture_id: str, stem: str) -> Optional[Path]:
    exact = filled_dir / f"{stem}.pdf"
    if exact.exists():
        return exact
    matches = []
    for pdf in filled_dir.glob("*.pdf"):
        if lecture_id_from_stem(pdf.stem) == lecture_id:
            matches.append(pdf)
    if matches:
        return sorted(matches)[0]
    return None


def convert_pptx_to_pdf(pptx: Path, out_dir: Path) -> Path:
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(pptx),
        ],
        check=True,
    )
    expected = out_dir / f"{pptx.stem}.pdf"
    if expected.exists():
        return expected
    pdfs = list(out_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"LibreOffice did not produce a PDF for {pptx}")
    return sorted(pdfs)[0]


def render_pdf_to_images(pdf: Path, out_dir: Path) -> None:
    prefix = out_dir / "slide"
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "200",
            str(pdf),
            str(prefix),
        ],
        check=True,
    )
    for image in sorted(out_dir.glob("slide-*.png")):
        match = re.search(r"-(\d+)\.png$", image.name)
        if not match:
            continue
        number = int(match.group(1))
        image.rename(out_dir / f"slide-{number:03d}.png")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    blank_dir = repo_root / "Blank"
    filled_dir = repo_root / "Filled"
    output_root = repo_root / "Lecture_Images"

    if not blank_dir.exists():
        print(f"Missing {blank_dir}; nothing to do.")
        return 0

    pptx_files = sorted(blank_dir.glob("*.pptx"))
    if not pptx_files:
        print(f"No PPTX files found in {blank_dir}")
        return 0

    for pptx in pptx_files:
        lecture_id = lecture_id_from_stem(pptx.stem)
        if not lecture_id:
            print(f"Skipping {pptx.name}: cannot parse lecture number")
            continue

        out_dir = output_root / f"Lecture{lecture_id}"
        pdf_override = find_pdf_override(filled_dir, lecture_id, pptx.stem)
        source_label = "PDF override" if pdf_override else "PPTX"
        print(f"{pptx.name}: using {source_label}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            if pdf_override:
                source_pdf = pdf_override
            else:
                source_pdf = convert_pptx_to_pdf(pptx, tmp_path)

            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            render_pdf_to_images(source_pdf, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
