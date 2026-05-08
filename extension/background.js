// Octoflash Chrome Extension - Background Service Worker

const DEFAULT_API_URL = "http://localhost:8000";

// Create context menu on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "queue-octoflash",
    title: "Queue in Octoflash",
    contexts: ["page", "link"],
    documentUrlPatterns: [
      "*://www.youtube.com/*",
      "*://youtu.be/*",
      "*://m.youtube.com/*",
    ],
  });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "queue-octoflash") return;

  const url = info.linkUrl || info.pageUrl || tab?.url;
  if (!url) {
    showBadge("ERR", "#dc2626");
    return;
  }

  await queueVideo(url);
});

async function queueVideo(url) {
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

    showBadge("OK", "#16a34a");
  } catch (err) {
    console.error("Octoflash queue failed:", err);
    showBadge("ERR", "#dc2626");
  }
}

function showBadge(text, color) {
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
  setTimeout(() => chrome.action.setBadgeText({ text: "" }), 3000);
}
