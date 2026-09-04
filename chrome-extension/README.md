# CSE 10124 Slide Picker

This unpacked Chrome extension opens a compact slide selector while working in
Google Colab.

- The Colab title is used to infer `LectureNN` or `ProgrammingDayNN`.
- Clicking thumbnails selects them in click order and shows numbered badges.
- Copying produces stable `by-id` image embeds with `---` between slides.
- Each deck offers a high-resolution, image-only PDF for importing into
  Notability without text reflow.
- GitHub authentication uses a repository-scoped GitHub App device flow.
- A manual regeneration starts the existing `lecture-images.yml` workflow,
  keeps its status in the popup, and sends a Chrome notification when complete.

## Local installation

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select this `chrome-extension` directory.

The GitHub App client ID is intentionally public and is stored in `config.js`.
No client secret is embedded in the extension. User and refresh tokens remain in
Chrome's extension-local storage.

## Notability workflow

1. Choose a deck and click **Download Notability PDF**.
2. Import that PDF into Notability and add handwriting normally.
3. Export the completed note as a PDF and upload it to the filled-slides folder.

The daily image workflow captures the Google Slides editor canvas and builds an
image-only PDF. Because the slide background contains pixels rather than live
text, Notability and later PDF renderers cannot rewrap tightly formatted content.
