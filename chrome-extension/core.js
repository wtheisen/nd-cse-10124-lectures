const SlidePickerCore = (() => {
  function inferDeckKey(title) {
    const value = String(title || "").replace(/\s+-\s+Colab.*$/i, "");
    const programming = /Programming[ _-]*Day[ _-]*(\d{1,3})/i.exec(value);
    if (programming) return `ProgrammingDay${programming[1].padStart(2, "0")}`;
    const lecture = /Lecture[ _-]*(\d{1,3})/i.exec(value);
    if (lecture) return `Lecture${lecture[1].padStart(2, "0")}`;
    return null;
  }

  function deckLabel(deckKey) {
    const programming = /^ProgrammingDay(\d+)$/.exec(deckKey);
    if (programming) return `Programming Day ${Number(programming[1])}`;
    const lecture = /^Lecture(\d+)$/.exec(deckKey);
    if (lecture) return `Lecture ${Number(lecture[1])}`;
    return deckKey;
  }

  function absoluteImageUrl(baseUrl, imagePath) {
    return `${String(baseUrl).replace(/\/$/, "")}/${String(imagePath).replace(/^\//, "")}`;
  }

  function renderEmbed(url) {
    return [
      '<div class="thumbnail">',
      `    <img src="${url}" class="img-responsive"/>`,
      "</div>"
    ].join("\n");
  }

  function buildClipboardText(slides, baseUrl) {
    return slides
      .map((slide) => renderEmbed(absoluteImageUrl(baseUrl, slide.image)))
      .join("\n\n---\n\n");
  }

  function toggleSelection(selectedIds, slideId) {
    const next = selectedIds.slice();
    const existing = next.indexOf(slideId);
    if (existing >= 0) next.splice(existing, 1);
    else next.push(slideId);
    return next;
  }

  function orderedSlides(deckSlides, selectedIds) {
    const byId = new Map(deckSlides.map((slide) => [slide.id, slide]));
    return selectedIds.map((id) => byId.get(id)).filter(Boolean);
  }

  return {
    absoluteImageUrl,
    buildClipboardText,
    deckLabel,
    inferDeckKey,
    orderedSlides,
    renderEmbed,
    toggleSelection
  };
})();

if (typeof module !== "undefined") module.exports = SlidePickerCore;
