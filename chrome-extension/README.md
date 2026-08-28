# CSE 10124 Slide Picker

This unpacked Chrome extension opens a compact slide selector while working in
Google Colab.

- The Colab title is used to infer `LectureNN` or `ProgrammingDayNN`.
- Clicking thumbnails selects them in click order and shows numbered badges.
- Copying produces stable `by-id` image embeds with `---` between slides.
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
