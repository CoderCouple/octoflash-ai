// Octoflash Chrome Extension - Popup

const DEFAULT_API_URL = "http://localhost:8000";

const urlInput = document.getElementById("urlInput");
const queueBtn = document.getElementById("queueBtn");
const statusEl = document.getElementById("status");
const apiUrlInput = document.getElementById("apiUrl");
const saveBtn = document.getElementById("saveBtn");

// Load saved API URL
chrome.storage.sync.get({ apiUrl: DEFAULT_API_URL }, (data) => {
  apiUrlInput.value = data.apiUrl;
});

// Auto-fill current tab URL if it's YouTube
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const tab = tabs[0];
  if (tab?.url && (tab.url.includes("youtube.com") || tab.url.includes("youtu.be"))) {
    urlInput.value = tab.url;
  }
});

// Queue video
queueBtn.addEventListener("click", async () => {
  const url = urlInput.value.trim();
  if (!url) {
    showStatus("Please enter a YouTube URL", "error");
    return;
  }

  queueBtn.disabled = true;
  queueBtn.textContent = "Queuing...";
  statusEl.className = "status";

  try {
    const { apiUrl } = await chrome.storage.sync.get({ apiUrl: DEFAULT_API_URL });

    const res = await fetch(`${apiUrl}/api/videos`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, source: "extension" }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    showStatus(`Queued! ID: ${data.video_id}`, "success");
    urlInput.value = "";
  } catch (err) {
    showStatus(err.message, "error");
  } finally {
    queueBtn.disabled = false;
    queueBtn.textContent = "Queue Video";
  }
});

// Save settings
saveBtn.addEventListener("click", () => {
  const apiUrl = apiUrlInput.value.trim().replace(/\/+$/, "") || DEFAULT_API_URL;
  chrome.storage.sync.set({ apiUrl }, () => {
    apiUrlInput.value = apiUrl;
    saveBtn.textContent = "Saved!";
    setTimeout(() => (saveBtn.textContent = "Save Settings"), 1500);
  });
});

function showStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className = `status ${type}`;
}
