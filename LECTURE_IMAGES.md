# Lecture image pipeline

The GitHub Pages workflow runs on the course Mac and renders slide decks to PNG
files that can be embedded directly in Colab notebooks.

## Naming

- `Slides/Lecture_01_*.gslides` publishes to `Lecture_Images/Lecture01/`.
- `Slides/Programming_Day_01_*.gslides` publishes to
  `Lecture_Images/ProgrammingDay01/`.
- Filled PDFs use the same filename prefixes. A filled PDF takes precedence over
  the corresponding Google Slides deck.
- Every slide is also published under its stable Google Slides object ID in a
  `by-id/` directory. These URLs survive slide insertion and reordering.

Lecture and programming-day numbers are independent; they do not need to fit into
one chronology.

## Sharing requirements

The Google Slides decks and the CSE 10124 filled-slides folder must be shared as
**Anyone with the link — Viewer**. The workflow uses anonymous downloads and does
not store Google credentials in GitHub.

## Automation

`.github/workflows/lecture-images.yml` runs daily at 11:10 AM Indianapolis time.
It can also be started manually from GitHub Actions or the Chrome extension. It
does not run on repository pushes. It:

1. Downloads the current contents of the filled-slides folder.
2. Discovers source decks from the configured Google Drive Slides folder.
3. Captures any deck without a filled PDF from the Google Slides editor in
   Chrome on the course Mac, using the same locally installed fonts as the
   authoring view.
4. Builds the Notability PDF from those captured pixels and publishes the same
   captures as numbered PNGs.
5. Pairs pages with native slide IDs and creates stable `by-id/` aliases.
6. Verifies page counts and deploys the result to GitHub Pages.

If any download or render fails, the workflow stops before deployment, leaving
the previous Pages site intact.

## Notebook markup

Generate a range of image tags with:

```sh
python3 scripts/html_img_generator.py lecture 2 1 4 8
python3 scripts/html_img_generator.py programming-day 1 1 2 7
```

The numbers are the slides' current positions. The generated URLs use stable IDs,
preserve the requested order, and place `---` between consecutive slide embeds.

## Local render

With Poppler (`pdfinfo` and `pdftoppm`) installed:

```sh
python3 scripts/render_lecture_images.py \
  --filled-dir "/path/to/Filled Slides Uploads/Fall 2026/cse 10124" \
  --slide-manifest-url "https://script.google.com/macros/s/DEPLOYMENT_ID/exec"
```

The renderer builds into a temporary directory and only replaces
`Lecture_Images/` after every deck succeeds. The Mac must be awake and its
self-hosted GitHub Actions runner service must be online at the scheduled time.
The runner is repository-scoped, requires the `cse10124-slides` label, runs with
an explicit macOS sandbox profile, and checks out without persisting GitHub
credentials.
