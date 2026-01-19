// frontend/app.js（認証・テキストチャット専用）
// - ログイン必須：最初に /me を確認し、未ログインなら /auth.html へ遷移
// - /chat は同一オリジン・ログインセッション前提

const $ = (id) => document.getElementById(id);

const chatForm = $("chatForm");
const chatInput = $("chatInput");
const chatArea = $("chatArea");

const pdfForm = $("pdfForm");
const pdfFile = $("pdfFile");
const pdfStatus = $("pdfStatus");
const pdfDocs = $("pdfDocs");
const pdfUploadButton = $("pdfUploadButton");
const logoutButton = $("logoutButton");
const logoutStatus = $("logoutStatus");

function setLogoutStatus(message, isError = false) {
  if (!logoutStatus) return;
  logoutStatus.textContent = message || "";
  logoutStatus.style.color = isError ? "#b42318" : "var(--muted)";
}

function setLogoutBusy(isBusy) {
  const elements = document.querySelectorAll("button, input");
  elements.forEach((el) => {
    if (isBusy) {
      if (!el.hasAttribute("data-logout-prev-disabled")) {
        el.setAttribute("data-logout-prev-disabled", el.disabled ? "1" : "0");
      }
      el.disabled = true;
      return;
    }
    const prev = el.getAttribute("data-logout-prev-disabled");
    if (prev === null) return;
    el.disabled = prev === "1";
    el.removeAttribute("data-logout-prev-disabled");
  });
}

function setPdfStatus(message, isError = false) {
  if (!pdfStatus) return;
  pdfStatus.textContent = message || "";
  pdfStatus.style.color = isError ? "#b42318" : "var(--muted)";
}

function formatTimestamp(value) {
  if (!value) return "";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return String(value);
  }
}

async function loadDocs() {
  if (!pdfDocs) return;
  pdfDocs.innerHTML = "";
  try {
    const r = await fetch("/rag/docs", { credentials: "include" });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      setPdfStatus(data.error || data.detail || "一覧の取得に失敗しました。", true);
      return;
    }
    const docs = Array.isArray(data.docs) ? data.docs : [];
    if (!docs.length) {
      const empty = document.createElement("div");
      empty.className = "doc-sub";
      empty.textContent = "アップロード済みPDFはありません。";
      pdfDocs.appendChild(empty);
      return;
    }
    for (const doc of docs) {
      const card = document.createElement("div");
      card.className = "doc-card";

      const meta = document.createElement("div");
      meta.className = "doc-meta";

      const title = document.createElement("div");
      title.className = "doc-title";
      title.textContent = doc.filename || "PDF";

      const sub = document.createElement("div");
      sub.className = "doc-sub";
      const stamp = formatTimestamp(doc.created_at);
      const chunks = typeof doc.chunks === "number" ? doc.chunks : 0;
      sub.textContent = `${stamp} · ${chunks} chunks`;

      meta.appendChild(title);
      meta.appendChild(sub);

      const actions = document.createElement("div");
      const del = document.createElement("button");
      del.textContent = "削除";
      del.addEventListener("click", async () => {
        if (!confirm("このPDFを削除しますか？")) return;
        del.disabled = true;
        try {
          const resp = await fetch(`/rag/docs/${doc.doc_id}`, {
            method: "DELETE",
            credentials: "include",
          });
          const resData = await resp.json().catch(() => ({}));
          if (!resp.ok) {
            setPdfStatus(resData.error || resData.detail || "削除に失敗しました。", true);
            return;
          }
          setPdfStatus("PDFを削除しました。");
          await loadDocs();
        } catch (e) {
          setPdfStatus(e?.message || String(e), true);
        } finally {
          del.disabled = false;
        }
      });
      actions.appendChild(del);

      card.appendChild(meta);
      card.appendChild(actions);
      pdfDocs.appendChild(card);
    }
  } catch (e) {
    setPdfStatus(e?.message || String(e), true);
  }
}

(async function ensureAuthed(){
  if (chatInput) chatInput.disabled = true;
  if (pdfFile) pdfFile.disabled = true;
  if (pdfUploadButton) pdfUploadButton.disabled = true;
  if (logoutButton) logoutButton.disabled = true;
  try {
    const r = await fetch("/me", { credentials: "same-origin" });
    if (!r.ok) throw new Error("not authed");
    await r.json();
    await loadDocs();
  } catch {
    location.replace("/auth.html");
    return;
  } finally {
    if (chatInput) chatInput.disabled = false;
    if (pdfFile) pdfFile.disabled = false;
    if (pdfUploadButton) pdfUploadButton.disabled = false;
    if (logoutButton) logoutButton.disabled = false;
  }
})();

logoutButton?.addEventListener("click", async () => {
  setLogoutStatus("ログアウト中...");
  setLogoutBusy(true);
  try {
    const r = await fetch("/auth/logout", {
      method: "POST",
      credentials: "same-origin",
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const err = data.error || data.detail || data.message || "ログアウトに失敗しました。";
      setLogoutStatus(err, true);
      return;
    }
    location.replace("/auth.html");
  } catch (e) {
    setLogoutStatus(e?.message || String(e), true);
  } finally {
    setLogoutBusy(false);
  }
});

pdfForm?.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!pdfFile || !pdfUploadButton) return;
  const file = pdfFile.files && pdfFile.files[0];
  if (!file) {
    setPdfStatus("PDFを選択してください。", true);
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  pdfUploadButton.disabled = true;
  setPdfStatus("アップロード中...");

  try {
    const r = await fetch("/rag/upload/pdf", {
      method: "POST",
      body: formData,
      credentials: "include",
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const err = data.error || data.detail || data.message || "アップロードに失敗しました。";
      setPdfStatus(err, true);
      return;
    }
    setPdfStatus(`アップロード完了: ${data.filename} (${data.chunks} chunks)`);
    pdfFile.value = "";
    await loadDocs();
  } catch (e) {
    setPdfStatus(e?.message || String(e), true);
  } finally {
    pdfUploadButton.disabled = false;
  }
});

chatForm?.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!chatInput || !chatArea) return;
  const text = chatInput.value.trim();
  if (!text) return;

  appendMessage("あなた", text);
  chatInput.value = "";

  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const err = data.error || data.detail || data.reply || `${r.status} ${r.statusText}`;
      appendMessage("Bot", err);
      return;
    }
    appendMessage("Bot", data.reply || "");
  } catch (e) {
    appendMessage("Bot", e?.message || String(e));
  }
});

function appendMessage(label, text){
  if (!chatArea) return;
  const p = document.createElement("p");
  p.innerHTML = `<b>${escapeHtml(label)}:</b> ${escapeHtml(text)}`;
  chatArea.appendChild(p);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}
