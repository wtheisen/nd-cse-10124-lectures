"""Render one new Google Slides deck and preserve the published catalog."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

from render_lecture_images import download_file

BASE = 'https://williamtheisen.com/nd-cse-10124-lectures/Lecture_Images/'


def asset_path(root, value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Invalid catalog asset path: {value}')
    result = root.joinpath(*path.parts)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def main():
    deck = os.environ['NEW_DECK']
    if not re.fullmatch(r'(Lecture|ProgrammingDay)\d{2}', deck):
        raise ValueError('Expected LectureNN or ProgrammingDayNN')
    repo = Path(__file__).resolve().parents[1]
    curl = shutil.which('curl')
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        previous_path = root / 'previous.json'
        download_file(BASE + 'manifest.json', previous_path, curl, 'published catalog')
        previous = json.loads(previous_path.read_text())
        if previous.get('version') != 2 or not previous.get('decks'):
            raise ValueError('Published catalog is missing or invalid')
        if deck in previous['decks']:
            raise ValueError('Deck already published; use the regular regeneration workflow')
        live_path = root / 'live.json'
        download_file(os.environ['SLIDE_MANIFEST_URL'], live_path, curl, 'live catalog')
        live = json.loads(live_path.read_text())
        selected = live['decks'][deck]
        if not selected['slides']:
            raise ValueError('New deck is empty')
        live['decks'] = {deck: selected}
        live_path.write_text(json.dumps(live))
        empty = root / 'empty-slides'
        empty.mkdir()
        output = root / 'output'
        subprocess.run([
            sys.executable, str(repo / 'scripts/render_lecture_images.py'),
            '--slides-dir', str(empty), '--slide-manifest-file', str(live_path),
            '--output-dir', str(output), '--chromium-executable',
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        ], check=True)
        new = json.loads((output / 'manifest.json').read_text())
        jobs = []
        for entry in previous['decks'].values():
            for slide in entry['slides']:
                jobs.append((slide['image'], None, None))
                jobs.append((slide['numbered_image'], None, None))
            if entry.get('notability_pdf'):
                jobs.append((entry['notability_pdf'], entry['notability_md5'], entry['notability_size_bytes']))

        def restore(job):
            name, checksum, size = job
            destination = asset_path(output, name)
            download_file(BASE + name, destination, curl, name)
            data = destination.read_bytes()
            if name.endswith('.png') and not data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'Invalid PNG: {name}')
            if checksum and (hashlib.md5(data).hexdigest() != checksum or len(data) != size):
                raise ValueError(f'PDF integrity failure: {name}')

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(restore, jobs))
        # Fail if another publisher changed the catalog while we were rendering.
        check_path = root / 'check.json'
        download_file(BASE + 'manifest.json', check_path, curl, 'catalog consistency check')
        if json.loads(check_path.read_text()) != previous:
            raise ValueError('Published catalog changed; rerun to preserve the latest decks')
        new['decks'] = {**previous['decks'], **new['decks']}
        (output / 'manifest.json').write_text(json.dumps(new, indent=2, sort_keys=True) + '\n')
        destination = repo / 'Lecture_Images'
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(output), destination)
        print(f'Added {deck} with {len(selected["slides"])} slides; preserved {len(previous["decks"])} decks.')


if __name__ == '__main__':
    main()
