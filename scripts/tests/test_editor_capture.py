"""Regression coverage for cropped images extending outside a slide's SVG bounds."""
import base64
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_lecture_images import DeckId, SlideInfo, SlideMap, capture_google_slides_editor_images


class EditorCaptureTest(unittest.TestCase):
    def test_clipped_image_does_not_shift_page_capture(self):
        photo = BytesIO()
        Image.new('RGB', (100, 400), 'red').save(photo, format='PNG')
        data = base64.b64encode(photo.getvalue()).decode()
        html = f'''<body style="margin:0"><svg id="canvas" width="800" height="800">
          <g transform="translate(100,250)"><g id="editor-test">
            <path fill="white" d="M0 0H600V450H0Z"/>
            <defs><clipPath id="crop"><rect x="20" y="20" width="100" height="40"/></clipPath></defs>
            <image x="20" y="-200" width="100" height="400" clip-path="url(#crop)" href="data:image/png;base64,{data}"/>
            <rect x="100" y="300" width="300" height="120" fill="blue"/>
          </g></g></svg></body>'''
        slide_map = SlideMap(DeckId('ProgrammingDay', '03'), 'test', '', (SlideInfo(1, 'test', ''),), 'test')
        with sync_playwright() as pw, tempfile.TemporaryDirectory() as tmp:
            browser = pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
            page = browser.new_page(viewport={'width': 1000, 'height': 1000})
            page.route('https://docs.google.com/**', lambda route: route.fulfill(body=html, content_type='text/html'))
            paths = capture_google_slides_editor_images(page, slide_map, Path(tmp)/'captures')
            # The old group screenshot measures 650 px high due to the clipped image.
            self.assertEqual(page.locator('#editor-test').bounding_box()['height'], 650)
            with Image.open(paths[0]) as captured:
                self.assertEqual(captured.size, (600, 450))
                self.assertEqual(captured.convert('RGB').getpixel((30, 30)), (255, 0, 0))
                self.assertEqual(captured.convert('RGB').getpixel((200, 400)), (0, 0, 255))
            browser.close()


if __name__ == '__main__':
    unittest.main()
