import {renderMarkdown} from './markdown.js';

const element = id => document.getElementById(id);
let session = null;
let activeJob = null;
let timer = null;
let enabled = false;
let loading = false;
let renderSignature = "";
let appliedJob = null;
let previewedJob = null;

async function request(path, body) {
    const response = await fetch(path, body ? {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
    } : {});
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
}

function showTab(name) {
    for (const tab of ["inspection", "chat"]) {
        const selected = name === tab;
        element(`${tab}-panel`).hidden = !selected;
        element(`${tab}-tab`).setAttribute("aria-selected", String(selected));
        element(`${tab}-tab`).tabIndex = 0;
    }
    element("load-board").hidden = name !== "inspection";
    element("run-inspection").hidden = name !== "inspection";
    element("review-banner").hidden = name !== "inspection" || element("mark-uncertain").hidden;
    if (name === "chat") resizeComposer();
}

for (const name of ["inspection", "chat"]) {
    element(`${name}-tab`).addEventListener("click", () => showTab(name));
}

function controls() {
    element("chat-send").disabled = !enabled || !!activeJob || loading;
    element("chat-new").disabled = !!activeJob || loading;
    element("chat-stop").hidden = !activeJob;
    element("chat-message").disabled = !enabled;
}

function textNode(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
}

function render() {
    element("chat-board").textContent = session.board;
    const messages = element("chat-messages");
    messages.replaceChildren();
    for (const message of session.messages) {
        const article = textNode("article", "", `chat-message ${message.role}`);
        article.append(textNode("strong", message.role === "assistant" ? "Inspection assistant" : message.role === "user" ? "You" : "Notice"));
        article.append(message.role === "assistant" ? renderMarkdown(message.text) : textNode("p", message.text));
        messages.append(article);
    }
    messages.scrollTop = messages.scrollHeight;
    const stats = element("chat-stats");
    stats.replaceChildren();
    const card = [...session.results].reverse().find(item => item.inspection);
    stats.hidden = !card;
    if (card) {
        stats.setAttribute("title", "Saved inspection results. Current freshness has not been verified.");
        const scores = card.inspection;
        const counts = [
            [scores.pw_flags.flat().filter(value => value !== -1).length, "Segments"],
            [scores.pw_flags.flat().filter(value => value === 1).length, "Pixel flagged"],
            [scores.ae_flags.flat().filter(value => value === 1).length, "AE flagged"],
            [scores.hybrid_flags.flat().filter(value => value === 1).length, "Both flagged"]
        ];
        for (const [count, label] of counts) {
            const stat = textNode("div", "", "chat-stat");
            stat.append(textNode("strong", String(count)), textNode("span", label));
            stats.append(stat);
        }
    }
}

async function refresh() {
    clearTimeout(timer);
    if (!session) return;
    try {
        session = await request(`/api/chat/session?${new URLSearchParams({id: session.id})}`);
        activeJob = session.jobs.find(job => ["queued", "running"].includes(job.status)) || null;
        const signature = JSON.stringify([session.messages, session.results]);
        if (signature !== renderSignature) {
            renderSignature = signature;
            render();
        }
        const latest = activeJob || session.jobs.at(-1);
        await showActivity(latest);
        element("chat-status").textContent = activeJob ? "Thinking…" : latest?.error || "";
    } catch (error) {
        element("chat-status").textContent = `Connection interrupted: ${error.message}. Reopen this conversation to reconnect.`;
    }
    controls();
    if (activeJob) timer = setTimeout(refresh, 1000);
}

async function send() {
    const message = element("chat-message").value.trim();
    if (!message || loading || activeJob) return;
    loading = true;
    controls();
    element("chat-status").textContent = "Thinking…";
    try {
        activeJob = await request("/api/chat/request", {
            session_id: session?.id, board_path: element("board-path").value.trim(), message
        });
        session = {id: activeJob.session_id};
        element("board-path").value = activeJob.board_path;
        element("chat-message").value = "";
        resizeComposer();
        await refresh();
    } catch (error) {
        element("chat-status").textContent = error.message;
    } finally {
        loading = false;
        controls();
    }
}

element("chat-send").addEventListener("click", send);
element("chat-message").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        send();
    }
});
element("chat-new").addEventListener("click", () => {
    session = null;
    renderSignature = "";
    element("chat-messages").replaceChildren(textNode("p", "Ask about the board selected above.", "chat-welcome"));
    element("chat-stats").hidden = true;
    element("chat-activity").hidden = true;
    element("chat-board").textContent = element("board-path").value;
    element("chat-status").textContent = "";
    element("chat-message").focus();
});
element("chat-stop").addEventListener("click", async () => {
    try {
        await request("/api/chat/cancel", {job_id: activeJob.id});
        element("chat-status").textContent = "Stopping…";
    } catch (error) {
        element("chat-status").textContent = error.message;
    }
});
controls();
request("/api/chat/status").then(status => {
    enabled = status.enabled;
    element("chat-status").textContent = "";
    controls();
}).catch(error => {
    element("chat-status").textContent = error.message;
    controls();
});

function resizeComposer() {
    const input = element("chat-message");
    const line = parseFloat(getComputedStyle(input).lineHeight);
    input.style.height = "0px";
    const lines = Math.max(1, Math.ceil((input.scrollHeight - 1) / line));
    input.style.height = `${Math.min(lines, 6) * line}px`;
    input.style.overflowY = lines > 6 ? "auto" : "hidden";
}

element("chat-message").addEventListener("input", resizeComposer);
window.addEventListener("resize", resizeComposer);
resizeComposer();

async function showActivity(job) {
    const box = element("chat-activity");
    box.hidden = !job || !["inspect", "refresh"].includes(job.action);
    if (box.hidden) return;
    const running = ["queued", "running"].includes(job.status);
    const last = job.events.at(-1);
    const preflight = [...session.results].reverse().find(item => item.tool === "VisualInspectionPreflight");
    const blocked = !running && job.status === "completed" && preflight?.payload.ready === false;
    const detail = blocked ? "Readiness checks blocked inspection. See the assistant’s explanation."
        : running ? last?.tool === "InspectBoard" ? "Inspecting the board and collecting results…" : "Checking inputs and preparing inspection evidence…"
        : job.error || "Inspection finished. Results are ready for human review.";
    box.dataset.status = blocked ? "failed" : running ? "running" : job.status;
    const heading = textNode("div", "");
    heading.append(textNode("span", "", "activity-dot"), textNode("strong", job.command));
    box.replaceChildren(heading, textNode("p", detail));
    window.dispatchEvent(new CustomEvent("inspection:activity", {detail: {job, running, text: detail}}));
    if (previewedJob !== job.id) {
        const payload = await request(`/api/board?${new URLSearchParams({path: job.board_path, inspect: "false"})}`);
        window.dispatchEvent(new CustomEvent("inspection:evidence", {detail: {payload, showStatus: false}}));
        previewedJob = job.id;
    }
    if (!running && !blocked && job.status === "completed" && appliedJob !== job.id) {
        const card = [...session.results].reverse().find(item => item.result_id && item.tool !== "SavedInspection");
        if (card) {
            const payload = await request(`/api/chat/evidence?${new URLSearchParams({session_id: session.id, result_id: card.result_id})}`);
            window.dispatchEvent(new CustomEvent("inspection:evidence", {detail: {payload}}));
            appliedJob = job.id;
        }
    }
}
