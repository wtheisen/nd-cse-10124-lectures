const assert = require("assert");
const core = require("../core.js");

assert.strictEqual(
  core.inferDeckKey("Programming_Day_01_Linear_Transformations.ipynb - Colab"),
  "ProgrammingDay01"
);
assert.strictEqual(core.inferDeckKey("Lecture 2 - High Dimensional Data - Colab"), "Lecture02");
assert.strictEqual(core.inferDeckKey("Unrelated notebook - Colab"), null);

assert.deepStrictEqual(core.toggleSelection([], "a"), ["a"]);
assert.deepStrictEqual(core.toggleSelection(["a", "b"], "a"), ["b"]);
assert.deepStrictEqual(core.toggleSelection(["a"], "b"), ["a", "b"]);

const slides = [
  { id: "a", image: "Lecture01/by-id/a.png" },
  { id: "b", image: "Lecture01/by-id/b.png" }
];
assert.deepStrictEqual(core.orderedSlides(slides, ["b", "a"]).map((slide) => slide.id), ["b", "a"]);

const copied = core.buildClipboardText(
  core.orderedSlides(slides, ["b", "a"]),
  "https://example.test/images/",
  "2026-09-04T14:50:51Z"
);
assert.ok(copied.indexOf("/b.png") < copied.indexOf("/a.png"));
assert.strictEqual((copied.match(/\n\n---\n\n/g) || []).length, 1);
assert.ok(!copied.endsWith("---"));
assert.ok(copied.includes("?v=2026-09-04T14%3A50%3A51Z"));

assert.strictEqual(
  core.versionedAssetUrl("https://example.test/images/", "a.png", "revision 2"),
  "https://example.test/images/a.png?v=revision%202"
);
assert.strictEqual(
  core.versionedAssetUrl("https://example.test/images/", "a.png", ""),
  "https://example.test/images/a.png"
);

console.log("Chrome extension core tests passed.");
