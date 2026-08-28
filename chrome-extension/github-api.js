const SlidePickerGitHub = (() => {
  async function parseResponse(response) {
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (_error) {
        data = { message: text };
      }
    }
    if (!response.ok) {
      const message = data && data.message ? data.message : `${response.status} ${response.statusText}`;
      throw new Error(message);
    }
    return data;
  }

  async function requestDeviceCode(clientId, repositoryId) {
    const body = new URLSearchParams({ client_id: clientId });
    if (repositoryId) body.set("repository_id", String(repositoryId));
    const response = await fetch("https://github.com/login/device/code", {
      method: "POST",
      headers: { Accept: "application/json" },
      body
    });
    return parseResponse(response);
  }

  async function pollDeviceToken(clientId, deviceCode) {
    const response = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { Accept: "application/json" },
      body: new URLSearchParams({
        client_id: clientId,
        device_code: deviceCode,
        grant_type: "urn:ietf:params:oauth:grant-type:device_code"
      })
    });
    return parseResponse(response);
  }

  async function refreshAccessToken(clientId, refreshToken) {
    const response = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { Accept: "application/json" },
      body: new URLSearchParams({
        client_id: clientId,
        refresh_token: refreshToken,
        grant_type: "refresh_token"
      })
    });
    return parseResponse(response);
  }

  async function apiRequest(config, token, path, options = {}) {
    const response = await fetch(`https://api.github.com${path}`, {
      ...options,
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "X-GitHub-Api-Version": config.apiVersion,
        ...(options.headers || {})
      }
    });
    return parseResponse(response);
  }

  async function dispatchWorkflow(config, token) {
    const path = `/repos/${config.owner}/${config.repository}/actions/workflows/${config.workflow}/dispatches`;
    return apiRequest(config, token, path, {
      method: "POST",
      body: JSON.stringify({ ref: config.ref })
    });
  }

  async function getWorkflowRun(config, token, runId) {
    return apiRequest(
      config,
      token,
      `/repos/${config.owner}/${config.repository}/actions/runs/${runId}`
    );
  }

  return {
    dispatchWorkflow,
    getWorkflowRun,
    pollDeviceToken,
    refreshAccessToken,
    requestDeviceCode
  };
})();

if (typeof module !== "undefined") module.exports = SlidePickerGitHub;
