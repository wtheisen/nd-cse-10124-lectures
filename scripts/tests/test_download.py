"""Exercise bounded download retries without contacting Google or waiting."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_lecture_images import download_file


class DownloadTest(unittest.TestCase):
    def download(self, responses):
        output = StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'manifest.json'
            def request(command, **kwargs):
                code, status = next(responses)
                destination.write_text('partial' if code else '{"decks": {}}')
                return subprocess.CompletedProcess(command, code, status, 'private response')
            with patch('render_lecture_images.subprocess.run', side_effect=request) as run, \
                 patch('render_lecture_images.time.sleep') as sleep, redirect_stdout(output):
                try:
                    download_file('https://example.com/?token=secret', destination, 'curl', 'manifest')
                    error = None
                except RuntimeError as exc:
                    error = str(exc)
            return error, run.call_count, sleep.call_args_list, output.getvalue(), destination.exists()

    def test_recovers_from_404_and_503(self):
        error, calls, delays, log, exists = self.download(iter([(22, '404'), (22, '503'), (0, '200')]))
        self.assertIsNone(error)
        self.assertEqual(calls, 3)
        self.assertEqual([c.args[0] for c in delays], [5, 10])
        self.assertTrue(exists)
        self.assertIn('attempt 3/3, HTTP 200, curl exit 0', log)
        self.assertNotIn('secret', log)
        self.assertNotIn('private response', log)

    def test_persistent_failure_stops_and_removes_partial_file(self):
        error, calls, delays, log, exists = self.download(iter([(28, '000')] * 3))
        self.assertIn('after 3 attempt(s)', error)
        self.assertEqual(calls, 3)
        self.assertEqual(len(delays), 2)
        self.assertFalse(exists)

    def test_auth_failure_is_not_retried(self):
        error, calls, delays, log, exists = self.download(iter([(56, '403')]))
        self.assertIn('HTTP 403', error)
        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])
        self.assertFalse(exists)

    def test_success_does_not_wait(self):
        error, calls, delays, log, exists = self.download(iter([(0, '200')]))
        self.assertIsNone(error)
        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])
        self.assertTrue(exists)


if __name__ == '__main__':
    unittest.main()
