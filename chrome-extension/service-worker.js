importScripts("config.js", "github-api.js");

const AUTH_ALARM = "github-auth-poll";
const RUN_ALARM = "workflow-run-poll";

async function getStored(keys) {
  return chrome.storage.local.get(keys);
}

async function setStored(values) {
  return chrome.storage.local.set(values);
}

async function removeStored(keys) {
  return chrome.storage.local.remove(keys);
}

function tokenRecord(payload) {
  const now = Date.now();
  return {
    accessToken: payload.access_token,
    tokenType: payload.token_type || "bearer",
    expiresAt: payload.expires_in ? now + payload.expires_in * 1000 : null,
    refreshToken: payload.refresh_token || null,
    refreshExpiresAt: payload.refresh_token_expires_in
      ? now + payload.refresh_token_expires_in * 1000
      : null
  };
}

async function ensureAccessToken() {
  const { githubAuth } = await getStored("githubAuth");
  if (!githubAuth || !githubAuth.accessToken) throw new Error("Connect GitHub first.");
  if (!githubAuth.expiresAt || githubAuth.expiresAt > Date.now() + 5 * 60 * 1000) {
    return githubAuth.accessToken;
  }
  if (!githubAuth.refreshToken || githubAuth.refreshExpiresAt < Date.now()) {
    await removeStored("githubAuth");
    throw new Error("Your GitHub session expired. Connect again.");
  }
  const refreshed = await SlidePickerGitHub.refreshAccessToken(
    SLIDE_PICKER_CONFIG.github.clientId,
    githubAuth.refreshToken
  );
  const next = tokenRecord(refreshed);
  await setStored({ githubAuth: next });
  return next.accessToken;
}

async function beginGithubAuth() {
  const config = SLIDE_PICKER_CONFIG.github;
  if (config.clientId.startsWith("__")) {
    throw new Error("The extension's GitHub App has not been configured yet.");
  }
  const payload = await SlidePickerGitHub.requestDeviceCode(
    config.clientId,
    config.repositoryId
  );
  const pendingGithubAuth = {
    deviceCode: payload.device_code,
    userCode: payload.user_code,
    verificationUri: payload.verification_uri,
    expiresAt: Date.now() + payload.expires_in * 1000,
    intervalSeconds: Math.max(Number(payload.interval) || 5, 5)
  };
  await setStored({ pendingGithubAuth });
  chrome.alarms.create(AUTH_ALARM, { periodInMinutes: 0.5 });
  return pendingGithubAuth;
}

async function pollGithubAuth() {
  const { pendingGithubAuth } = await getStored("pendingGithubAuth");
  if (!pendingGithubAuth) return { state: "idle" };
  if (pendingGithubAuth.expiresAt <= Date.now()) {
    await removeStored("pendingGithubAuth");
    await chrome.alarms.clear(AUTH_ALARM);
    return { state: "expired" };
  }

  const payload = await SlidePickerGitHub.pollDeviceToken(
    SLIDE_PICKER_CONFIG.github.clientId,
    pendingGithubAuth.deviceCode
  );
  if (payload.access_token) {
    await setStored({ githubAuth: tokenRecord(payload) });
    await removeStored("pendingGithubAuth");
    await chrome.alarms.clear(AUTH_ALARM);
    await chrome.notifications.create("github-connected", {
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: "CSE 10124 Slide Picker",
      message: "GitHub is connected. You can regenerate slide images from Colab."
    });
    return { state: "connected" };
  }
  if (payload.error === "authorization_pending" || payload.error === "slow_down") {
    return { state: "pending", userCode: pendingGithubAuth.userCode };
  }
  await removeStored("pendingGithubAuth");
  await chrome.alarms.clear(AUTH_ALARM);
  throw new Error(payload.error_description || payload.error || "GitHub authorization failed.");
}

async function dispatchWorkflow() {
  const token = await ensureAccessToken();
  const run = await SlidePickerGitHub.dispatchWorkflow(
    SLIDE_PICKER_CONFIG.github,
    token
  );
  if (!run || !run.workflow_run_id) {
    throw new Error("GitHub accepted the request but did not return a workflow run ID.");
  }
  const currentRun = {
    id: run.workflow_run_id,
    apiUrl: run.run_url,
    htmlUrl: run.html_url,
    status: "queued",
    conclusion: null,
    notified: false,
    startedAt: new Date().toISOString()
  };
  await setStored({ currentRun });
  chrome.alarms.create(RUN_ALARM, { periodInMinutes: 0.5 });
  return currentRun;
}

async function refreshWorkflowRun() {
  const { currentRun } = await getStored("currentRun");
  if (!currentRun) return null;
  if (currentRun.status === "completed") return currentRun;

  const token = await ensureAccessToken();
  const run = await SlidePickerGitHub.getWorkflowRun(
    SLIDE_PICKER_CONFIG.github,
    token,
    currentRun.id
  );
  const updated = {
    ...currentRun,
    status: run.status,
    conclusion: run.conclusion,
    htmlUrl: run.html_url || currentRun.htmlUrl
  };

  if (updated.status === "completed" && !updated.notified) {
    updated.notified = true;
    const succeeded = updated.conclusion === "success";
    await chrome.notifications.create(`workflow-${updated.id}`, {
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: succeeded ? "Lecture slides are ready" : "Lecture slide generation failed",
      message: succeeded
        ? "The new slide images and stable links are live."
        : "Open the workflow run to see what needs attention."
    });
    await chrome.alarms.clear(RUN_ALARM);
  }
  await setStored({ currentRun: updated });
  return updated;
}

async function authState() {
  const { githubAuth, pendingGithubAuth } = await getStored([
    "githubAuth",
    "pendingGithubAuth"
  ]);
  return {
    connected: Boolean(githubAuth && githubAuth.accessToken),
    pending: pendingGithubAuth || null
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const handlers = {
    AUTH_STATE: authState,
    BEGIN_GITHUB_AUTH: beginGithubAuth,
    POLL_GITHUB_AUTH: pollGithubAuth,
    DISCONNECT_GITHUB: async () => {
      await removeStored(["githubAuth", "pendingGithubAuth"]);
      await chrome.alarms.clear(AUTH_ALARM);
      return { connected: false };
    },
    DISPATCH_WORKFLOW: dispatchWorkflow,
    GET_WORKFLOW_RUN: refreshWorkflowRun
  };
  const handler = handlers[message && message.type];
  if (!handler) return false;
  handler()
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === AUTH_ALARM) pollGithubAuth().catch(() => {});
  if (alarm.name === RUN_ALARM) refreshWorkflowRun().catch(() => {});
});

chrome.notifications.onClicked.addListener(async (notificationId) => {
  if (!notificationId.startsWith("workflow-")) return;
  const { currentRun } = await getStored("currentRun");
  if (currentRun && currentRun.htmlUrl) chrome.tabs.create({ url: currentRun.htmlUrl });
});
