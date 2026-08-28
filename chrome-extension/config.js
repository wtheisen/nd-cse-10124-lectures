const SLIDE_PICKER_CONFIG = Object.freeze({
  manifestUrl:
    "https://williamtheisen.com/nd-cse-10124-lectures/Lecture_Images/manifest.json",
  imageBaseUrl:
    "https://williamtheisen.com/nd-cse-10124-lectures/Lecture_Images",
  github: Object.freeze({
    owner: "wtheisen",
    repository: "nd-cse-10124-lectures",
    repositoryId: 1134891396,
    workflow: "lecture-images.yml",
    ref: "master",
    apiVersion: "2026-03-10",
    clientId: "Iv23liGlI4iGuIayzjo9"
  })
});

if (typeof module !== "undefined") module.exports = SLIDE_PICKER_CONFIG;
