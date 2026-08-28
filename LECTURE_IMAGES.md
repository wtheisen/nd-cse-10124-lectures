# Lecture image pipeline

The GitHub Pages workflow renders slide decks to PNG files that can be embedded
directly in Colab notebooks.

## Naming

- `Slides/Lecture_01_*.gslides` publishes to `Lecture_Images/Lecture01/`.
- `Slides/Programming_Day_01_*.gslides` publishes to
  `Lecture_Images/ProgrammingDay01/`.
- Filled PDFs use the same filename prefixes. A filled PDF takes precedence over
  the corresponding Google Slides deck.

Lecture and programming-day numbers are independent; they do not need to fit into
one chronology.

## Sharing requirements

The Google Slides decks and the CSE 10124 filled-slides folder must be shared as
**Anyone with the link — Viewer**. The workflow uses anonymous downloads and does
not store Google credentials in GitHub.

## Automation

`.github/workflows/lecture-images.yml` runs at 11:25 AM Indianapolis time on
Monday, Wednesday, and Friday. It can also be started manually from GitHub
Actions. It:

1. Downloads the current contents of the filled-slides folder.
2. Exports any deck without a filled PDF directly from Google Slides as PDF.
3. Renders every PDF page to a numbered PNG.
4. Verifies page counts and deploys the result to GitHub Pages.

If any download or render fails, the workflow stops before deployment, leaving
the previous Pages site intact.

## Notebook markup

Generate a range of image tags with:

```sh
python3 scripts/html_img_generator.py lecture 2 1 29
python3 scripts/html_img_generator.py programming-day 1 1 18
```

The ending slide number is inclusive.

## Local render

With Poppler (`pdfinfo` and `pdftoppm`) installed:

```sh
python3 scripts/render_lecture_images.py \
  --filled-dir "/path/to/Filled Slides Uploads/Fall 2026/cse 10124"
```

The renderer builds into a temporary directory and only replaces
`Lecture_Images/` after every deck succeeds.
