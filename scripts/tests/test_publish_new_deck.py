import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish_new_deck as publisher


class PublicationTests(unittest.TestCase):
    def test_preserves_catalog_and_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / 'scripts').mkdir()
            old = {'version': 2, 'decks': {'Lecture01': {'slides': [
                {'image': 'Lecture01/by-id/a.png', 'numbered_image': 'Lecture01/slide-001.png'}]}}}
            live = {'decks': {'ProgrammingDay05': {'slides': [{'id': 'b'}]}}}
            def download(url, destination, *args):
                if url == 'live':
                    destination.write_text(json.dumps(live))
                elif url.endswith('manifest.json'):
                    destination.write_text(json.dumps(old))
                else:
                    destination.write_bytes(b'\x89PNG\r\n\x1a\noriginal')
            def render(args, **kwargs):
                output = Path(args[args.index('--output-dir') + 1])
                output.mkdir()
                output.joinpath('manifest.json').write_text(json.dumps({
                    'version': 2, 'decks': {'ProgrammingDay05': {'slide_count': 1}}}))
            with patch.dict(os.environ, NEW_DECK='ProgrammingDay05', SLIDE_MANIFEST_URL='live'), patch.object(publisher, '__file__', str(repo / 'scripts/publish_new_deck.py')), patch.object(publisher, 'download_file', side_effect=download), patch.object(publisher.subprocess, 'run', side_effect=render):
                publisher.main()
            result = json.loads((repo / 'Lecture_Images/manifest.json').read_text())
            self.assertEqual(result['decks']['Lecture01'], old['decks']['Lecture01'])
            self.assertIn('ProgrammingDay05', result['decks'])
            self.assertEqual((repo / 'Lecture_Images/Lecture01/by-id/a.png').read_bytes(), b'\x89PNG\r\n\x1a\noriginal')

    def test_replacement_requires_opt_in_and_keeps_new_images(self):
        for replace in ('false', 'true'):
            with self.subTest(replace=replace), tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                (repo / 'scripts').mkdir()
                old = {'version': 2, 'decks': {
                    'Lecture01': {'slides': []},
                    'ProgrammingDay06': {'slides': [{
                        'image': 'ProgrammingDay06/by-id/a.png',
                        'numbered_image': 'ProgrammingDay06/slide-001.png'}]}}}
                live = {'decks': {'ProgrammingDay06': {'slides': [{'id': 'a'}]}}}
                def download(url, destination, *args):
                    if url == 'live':
                        destination.write_text(json.dumps(live))
                    elif url.endswith('manifest.json'):
                        destination.write_text(json.dumps(old))
                    else:
                        self.fail('Replacement must not restore the old deck images')
                def render(args, **kwargs):
                    output = Path(args[args.index('--output-dir') + 1])
                    (output / 'ProgrammingDay06').mkdir(parents=True)
                    (output / 'ProgrammingDay06/slide-001.png').write_bytes(b'corrected')
                    (output / 'manifest.json').write_text(json.dumps({
                        'version': 2, 'decks': {'ProgrammingDay06': {'slide_count': 1}}}))
                with patch.dict(os.environ, NEW_DECK='ProgrammingDay06', REPLACE_EXISTING=replace, SLIDE_MANIFEST_URL='live'), patch.object(publisher, '__file__', str(repo / 'scripts/publish_new_deck.py')), patch.object(publisher, 'download_file', side_effect=download), patch.object(publisher.subprocess, 'run', side_effect=render):
                    if replace == 'false':
                        with self.assertRaisesRegex(ValueError, 'explicitly enable replacement'):
                            publisher.main()
                    else:
                        publisher.main()
                        result = json.loads((repo / 'Lecture_Images/manifest.json').read_text())
                        self.assertEqual(result['decks']['Lecture01'], old['decks']['Lecture01'])
                        self.assertEqual((repo / 'Lecture_Images/ProgrammingDay06/slide-001.png').read_bytes(), b'corrected')

    def test_rejects_unsafe_paths(self):
        for name in ['../escape', '/absolute']:
            with self.assertRaises(ValueError):
                publisher.asset_path(Path('/tmp'), name)

    def test_rejects_invalid_deck(self):
        with patch.dict(os.environ, NEW_DECK='../invalid'):
            with self.assertRaises(ValueError):
                publisher.main()


if __name__ == '__main__':
    unittest.main()
