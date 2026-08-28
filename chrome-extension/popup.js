const state = {
  manifest: null,
  deckKey: null,
  selectedIds: [],
  activeTabTitle: ""
};

const elements = {};

function cacheElements() {
  [
    "clear-selection",
    "connect-github",
    "copy-selected",
    "deck-select",
    "disconnect-github",
    "github-status-control",
    "inference-status",
    "notice",
    "refresh-manifest",
    "run-badge",
    "run-description",
    "run-generator",
    "run-generator-control",
    "selection-count",
    "slide-grid",
    "view-run"
  ].forEach((id) => { elements[id] = document.getElementById(id); });
}

function showNotice(message, isError = false) {
  elements.notice.textContent = message;
  elements.notice.classList.toggle("error", isError);
  elements.notice.hidden = !message;
}

async function runtimeMessage(type) {
  const response = await chrome.runtime.sendMessage({ type });
  if (!response || !response.ok) throw new Error(response && response.error || "Extension request failed.");
  return response.result;
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab || {};
}

async function loadManifest() {
  showNotice("");
  elements["slide-grid"].innerHTML = '<div class="empty-state">Loading slide thumbnails…</div>';
  const url = `${SLIDE_PICKER_CONFIG.manifestUrl}?t=${Date.now()}`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load slides (${response.status}).`);
  const manifest = await response.json();
  if (manifest.version !== 2 || !manifest.decks) {
    throw new Error("The slide site has not been upgraded to stable links yet.");
  }
  state.manifest = manifest;
  populateDecks();
}

function populateDecks() {
  const deckKeys = Object.keys(state.manifest.decks).sort();
  const inferred = SlidePickerCore.inferDeckKey(state.activeTabTitle);
  elements["deck-select"].replaceChildren();
  deckKeys.forEach((deckKey) => {
    const option = document.createElement("option");
    option.value = deckKey;
    option.textContent = SlidePickerCore.deckLabel(deckKey);
    elements["deck-select"].append(option);
  });

  if (inferred && deckKeys.includes(inferred)) {
    state.deckKey = inferred;
    elements["inference-status"].textContent = "Matched from Colab title";
  } else {
    state.deckKey = deckKeys[0] || null;
    elements["inference-status"].textContent = inferred
      ? "No published match—choose a deck"
      : "Choose a deck";
  }
  elements["deck-select"].value = state.deckKey || "";
  state.selectedIds = [];
  renderSlides();
}

function currentSlides() {
  if (!state.manifest || !state.deckKey) return [];
  return state.manifest.decks[state.deckKey].slides || [];
}

function renderSlides() {
  const slides = currentSlides();
  elements["slide-grid"].replaceChildren();
  if (!slides.length) {
    elements["slide-grid"].innerHTML = '<div class="empty-state">No slides are published for this deck.</div>';
    updateSelectionUi();
    return;
  }

  slides.forEach((slide) => {
    const selectedIndex = state.selectedIds.indexOf(slide.id);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `slide-card${selectedIndex >= 0 ? " selected" : ""}`;
    card.dataset.slideId = slide.id;
    card.title = slide.title || `Slide ${slide.position}`;

    const image = document.createElement("img");
    image.src = SlidePickerCore.absoluteImageUrl(
      SLIDE_PICKER_CONFIG.imageBaseUrl,
      slide.image
    );
    image.alt = slide.title || `Slide ${slide.position}`;
    image.loading = "lazy";
    card.append(image);

    if (selectedIndex >= 0) {
      const badge = document.createElement("span");
      badge.className = "order-badge";
      badge.textContent = String(selectedIndex + 1);
      card.append(badge);
    }

    const meta = document.createElement("span");
    meta.className = "slide-meta";
    const position = document.createElement("span");
    position.className = "slide-position";
    position.textContent = `#${slide.position}`;
    const title = document.createElement("span");
    title.className = "slide-title";
    title.textContent = slide.title || "Untitled slide";
    meta.append(position, title);
    card.append(meta);

    card.addEventListener("click", () => {
      state.selectedIds = SlidePickerCore.toggleSelection(state.selectedIds, slide.id);
      renderSlides();
    });
    elements["slide-grid"].append(card);
  });
  updateSelectionUi();
}

function updateSelectionUi() {
  const count = state.selectedIds.length;
  elements["selection-count"].textContent = count
    ? `${count} slide${count === 1 ? "" : "s"} selected`
    : "No slides selected";
  elements["clear-selection"].disabled = count === 0;
  elements["copy-selected"].disabled = count === 0;
  elements["copy-selected"].textContent = count
    ? `Copy ${count} slide${count === 1 ? "" : "s"} for Colab`
    : "Select slides to copy";
}

async function copySelected() {
  const slides = SlidePickerCore.orderedSlides(currentSlides(), state.selectedIds);
  const text = SlidePickerCore.buildClipboardText(
    slides,
    SLIDE_PICKER_CONFIG.imageBaseUrl
  );
  await navigator.clipboard.writeText(text);
  elements["copy-selected"].textContent = `Copied ${slides.length} slide${slides.length === 1 ? "" : "s"}!`;
  setTimeout(updateSelectionUi, 1600);
}

function setRunUi(run, connected = true) {
  const badge = elements["run-badge"];
  const workflowControl = elements["run-generator-control"];
  badge.className = "status-badge idle";
  workflowControl.className = "icon-button workflow-status-button idle";
  workflowControl.disabled = !connected;
  elements["run-generator"].disabled = !connected;
  elements["view-run"].hidden = true;
  if (!connected) {
    workflowControl.classList.replace("idle", "disconnected");
    workflowControl.title = "Connect GitHub to regenerate images";
    workflowControl.setAttribute("aria-label", workflowControl.title);
  }
  if (!run) {
    badge.textContent = "Idle";
    elements["run-description"].textContent = "Ready to regenerate the published slide images.";
    if (connected) {
      workflowControl.title = "Regenerate slide images";
      workflowControl.setAttribute("aria-label", workflowControl.title);
    }
    return;
  }
  if (run.status !== "completed") {
    badge.className = "status-badge running";
    badge.textContent = run.status || "Running";
    elements["run-description"].textContent = "Generation is running. You can close this popup; Chrome will notify you.";
    workflowControl.classList.replace("idle", "running");
    workflowControl.disabled = true;
    elements["run-generator"].disabled = true;
    workflowControl.title = "Slide image generation is running";
  } else if (run.conclusion === "success") {
    badge.className = "status-badge success";
    badge.textContent = "Ready";
    elements["run-description"].textContent = "The latest slide images were published successfully.";
    workflowControl.classList.replace("idle", "success");
    workflowControl.title = "Images ready — click to regenerate again";
  } else {
    badge.className = "status-badge failure";
    badge.textContent = "Failed";
    elements["run-description"].textContent = "Generation failed. Open the workflow run for details.";
    workflowControl.classList.replace("idle", "failure");
    workflowControl.title = "Last run failed — click to try again";
  }
  workflowControl.setAttribute("aria-label", workflowControl.title);
  if (run.htmlUrl) {
    elements["view-run"].href = run.htmlUrl;
    elements["view-run"].hidden = false;
  }
}

async function refreshGithubUi() {
  const auth = await runtimeMessage("AUTH_STATE");
  const githubControl = elements["github-status-control"];
  githubControl.classList.remove("disconnected", "pending", "connected");
  const connectionState = auth.connected ? "connected" : auth.pending ? "pending" : "disconnected";
  githubControl.classList.add(connectionState);
  githubControl.title = auth.connected
    ? "GitHub connected"
    : auth.pending
      ? `Waiting for GitHub code ${auth.pending.userCode}`
      : "Connect GitHub";
  githubControl.setAttribute("aria-label", githubControl.title);
  elements["connect-github"].hidden = auth.connected;
  elements["disconnect-github"].hidden = !auth.connected;
  elements["run-generator"].disabled = !auth.connected;
  if (auth.pending && !auth.connected) {
    elements["connect-github"].textContent = `Waiting for ${auth.pending.userCode}`;
  } else {
    elements["connect-github"].textContent = "Connect GitHub";
  }
  if (auth.connected) {
    const run = await runtimeMessage("GET_WORKFLOW_RUN");
    setRunUi(run, true);
  } else {
    setRunUi(null, false);
    elements["run-description"].textContent = "Connect GitHub to regenerate and publish slide images.";
  }
}

async function connectGithub() {
  const pending = await runtimeMessage("BEGIN_GITHUB_AUTH");
  await navigator.clipboard.writeText(pending.userCode);
  showNotice(`GitHub code ${pending.userCode} was copied. Paste it into the page that just opened.`);
  await chrome.tabs.create({ url: pending.verificationUri });
}

async function handleGithubStatusClick() {
  const auth = await runtimeMessage("AUTH_STATE");
  if (auth.connected) {
    showNotice("GitHub is connected and ready.");
    return;
  }
  if (auth.pending) {
    showNotice(`Waiting for GitHub code ${auth.pending.userCode}.`);
    return;
  }
  await connectGithub();
  await refreshGithubUi();
}

async function dispatchGenerator() {
  elements["run-generator"].disabled = true;
  elements["run-generator"].textContent = "Starting…";
  elements["run-generator-control"].disabled = true;
  elements["run-generator-control"].title = "Starting slide image generation";
  elements["run-generator-control"].setAttribute("aria-label", elements["run-generator-control"].title);
  const run = await runtimeMessage("DISPATCH_WORKFLOW");
  setRunUi(run, true);
  elements["run-generator"].textContent = "Regenerate images";
}

function installListeners() {
  elements["deck-select"].addEventListener("change", (event) => {
    state.deckKey = event.target.value;
    state.selectedIds = [];
    elements["inference-status"].textContent = "Selected manually";
    renderSlides();
  });
  elements["clear-selection"].addEventListener("click", () => {
    state.selectedIds = [];
    renderSlides();
  });
  elements["copy-selected"].addEventListener("click", () => copySelected().catch((error) => showNotice(error.message, true)));
  elements["refresh-manifest"].addEventListener("click", () => loadManifest().catch((error) => showNotice(error.message, true)));
  elements["connect-github"].addEventListener("click", () => connectGithub().catch((error) => showNotice(error.message, true)));
  elements["github-status-control"].addEventListener("click", () => handleGithubStatusClick().catch((error) => showNotice(error.message, true)));
  elements["run-generator-control"].addEventListener("click", () => dispatchGenerator().catch((error) => {
    elements["run-generator"].textContent = "Regenerate images";
    showNotice(error.message, true);
    refreshGithubUi().catch(() => {});
  }));
  elements["disconnect-github"].addEventListener("click", async () => {
    await runtimeMessage("DISCONNECT_GITHUB");
    await refreshGithubUi();
  });
  elements["run-generator"].addEventListener("click", () => dispatchGenerator().catch((error) => {
    elements["run-generator"].textContent = "Regenerate images";
    showNotice(error.message, true);
    refreshGithubUi().catch(() => {});
  }));
}

async function initialize() {
  cacheElements();
  installListeners();
  const tab = await getActiveTab();
  state.activeTabTitle = tab.title || "";
  await Promise.all([loadManifest(), refreshGithubUi()]);
  setInterval(() => refreshGithubUi().catch(() => {}), 5000);
}

initialize().catch((error) => showNotice(error.message, true));
