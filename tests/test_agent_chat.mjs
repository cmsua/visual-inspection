import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

class Node {
    constructor() { this.children = []; this.events = {}; this.value = ''; this.hidden = false; this.textContent = ''; this.attributes = {}; this.style = {}; this.dataset = {}; this.scrollHeight = 24; }
    addEventListener(name, callback) { const previous = this.events[name]; this.events[name] = async event => { await previous?.(event); return callback(event); }; }
    setAttribute(key, value) { this.attributes[key] = value; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    add(node) { this.append(node); }
    focus() {}
    async click() { return this.events.click?.({target: this}); }
}
const nodes = new Map();
const get = id => { if (!nodes.has(id)) nodes.set(id, new Node()); return nodes.get(id); };
get('mark-uncertain').hidden = true;
get('chat-panel').hidden = true;
get('board-path').value = '/board.npy';
const calls = [];
const events = [];
let holdInspection = false;
const session = {id: 'demo', board: '/board.npy', messages: [], results: [], artifacts: {}, jobs: []};
const context = vm.createContext({
    document: {
        getElementById: get,
        createElement: tag => Object.assign(new Node(), {tagName: tag}),
        createTextNode: text => Object.assign(new Node(), {tagName: '#text', textContent: text})
    },
    window: {dispatchEvent: event => events.push(event), addEventListener: () => {}},
    getComputedStyle: () => ({lineHeight: "24px"}),
    CustomEvent: class {constructor(name, options) {this.type = name; this.detail = options.detail;}},
    Option: class extends Node {constructor(label, value) {super(); this.textContent = label; this.value = value;}},
    URL, URLSearchParams, console, setTimeout: () => 0, clearTimeout: () => {},
    fetch: async (path, options) => {
        const body = options?.body ? JSON.parse(options.body) : null;
        calls.push({path, body});
        let result;
        if (path === '/api/chat/status') result = {enabled: true, sessions: [{id: session.id, board: session.board}]};
        else if (path === '/api/chat/sessions' || path.startsWith('/api/chat/session?')) result = session;
        else if (path === '/api/chat/request') {
            session.board = body.board_path;
            session.messages.push({role: 'assistant', text: '<img src=x onerror=evil()> Human review required.'});
            session.jobs = [{id: String(calls.length), session_id: session.id, board_path: session.board, action: 'chat', status: 'completed', events: [], error: null}];
            if (body.message === 'Inspect board 445') {
                session.jobs[0].action = 'inspect';
                session.jobs[0].status = holdInspection ? 'running' : 'completed';
                session.jobs[0].command = 'Inspect /board.npy';
                session.results = [{tool: 'InspectBoard', result_id: 'result', inspection: {
                    pw_flags: [[1, 0]], ae_flags: [[1, 1]], hybrid_flags: [[1, 0]]
                }}];
            }
            result = session.jobs[0];
        } else if (path.startsWith('/api/board?')) result = {inspection: {metadata: {inspection_status: 'not_run'}, pw_flags: [[null]]}};
        else if (path.startsWith('/api/chat/evidence?')) result = {review_context: {}, inspection: {}};
        else throw new Error(`Unexpected request: ${path}`);
        return {ok: true, json: async () => structuredClone(result)};
    }
});
vm.runInContext(fs.readFileSync('web/interface/markdown.js', 'utf8').replace('export function', 'function'), context);
vm.runInContext(fs.readFileSync('web/interface/chat.js', 'utf8').replace("import {renderMarkdown} from './markdown.js';", ''), context);
const settle = () => new Promise(resolve => setImmediate(resolve));
await settle();
assert.equal(calls.length, 1, 'startup only requests availability');
await get('chat-tab').click();
assert.equal(get('chat-panel').hidden, false);
assert.equal(get('inspection-panel').hidden, true);
assert.equal(calls.length, 1, 'opening chat must not invoke Codex');
get('chat-message').value = 'Summarize the latest results';
await get('chat-send').click();
const summary = calls.find(call => call.path === '/api/chat/request');
assert.equal(summary.body.message, 'Summarize the latest results');
assert.equal(summary.body.authorize_inspection, undefined);
assert.equal(summary.body.session_id, undefined);
const message = get('chat-messages').children[0].children[1];
const descendants = node => [node, ...node.children.flatMap(descendants)];
const renderedText = node => descendants(node).map(item => item.textContent).join('');
assert.match(renderedText(message), /<img src=x/);
assert.equal(descendants(message).some(node => node.tagName === 'img'), false, 'raw HTML remains text');
get('board-path').value = './data/another.npy';
get('chat-message').value = 'Summarize this board';
await get('chat-send').click();
assert.equal(calls.filter(call => call.path === '/api/chat/request').at(-1).body.board_path, './data/another.npy');
await get('inspection-tab').click();
assert.equal(get('inspection-panel').hidden, false);
assert.equal(get('review-banner').hidden, true);
const html = fs.readFileSync('web/interface/index.html', 'utf8');
for (const removed of ['chat-inspect', 'chat-preflight', 'chat-artifacts', 'chat-evidence', 'chat-result-path']) {
    assert.equal(html.includes(`id="${removed}"`), false, `removed control ${removed}`);
}
console.log('Chat UI: plain chat, board changes, no implicit inspection, tabs, safe text rendering passed.');

assert.equal(get('chat-message').style.height, '24px');
get('chat-message').scrollHeight = 72;
await get('chat-message').events.input({});
assert.equal(get('chat-message').style.height, '72px');
get('chat-message').scrollHeight = 200;
await get('chat-message').events.input({});
assert.equal(get('chat-message').style.height, '144px');
assert.equal(get('chat-message').style.overflowY, 'auto');
get('chat-message').scrollHeight = 24;
await get('chat-message').events.input({});
assert.equal(get('chat-message').style.height, '24px');
console.log('Composer grows by lines, caps at six lines, scrolls, and shrinks.');

get('chat-message').value = 'Inspect board 445';
await get('chat-send').click();
assert.equal(get('chat-activity').hidden, false);
assert.ok(events.some(event => event.type === 'inspection:activity'));
assert.ok(events.some(event => event.type === 'inspection:evidence'));
assert.equal(calls.filter(call => call.path.startsWith('/api/chat/evidence?')).length, 1);
console.log('Inspection activity and saved-result handoff reach the inspection view.');

const previews = calls.filter(call => call.path.startsWith('/api/board?'));
assert.equal(previews.length, 1);
assert.equal(new URLSearchParams(previews[0].path.split('?')[1]).get('inspect'), 'false');
const handoffs = events.filter(event => event.type === 'inspection:evidence');
assert.equal(handoffs[0].detail.payload.inspection.pw_flags[0][0], null);
assert.equal(handoffs[0].detail.showStatus, false);
assert.ok(calls.indexOf(previews[0]) < calls.findIndex(call => call.path.startsWith('/api/chat/evidence?')));
console.log('Matched board preview loads before classified results, without inference.');

holdInspection = true;
get('chat-message').value = 'Inspect board 445';
await get('chat-send').click();
assert.equal(events.filter(event => event.type === 'inspection:activity').at(-1).detail.running, true);
assert.equal(events.filter(event => event.type === 'inspection:evidence').at(-1).detail.payload.inspection.pw_flags[0][0], null);
assert.equal(calls.filter(call => call.path.startsWith('/api/chat/evidence?')).length, 1);
session.jobs[0].status = 'completed';
await vm.runInContext('refresh()', context);
assert.equal(calls.filter(call => call.path.startsWith('/api/board?')).length, 2, 'one preview per job');
assert.equal(calls.filter(call => call.path.startsWith('/api/chat/evidence?')).length, 2);
console.log('Running job displays unclassified preview, then replaces it on completion.');

const markdownSample = [
    'Board **001** has **14 flags**.', '',
    '- **AE:** 14', '- *Pixel:* 0', '',
    '1. Review `source (1, 3)`', '',
    '```bash', 'python -m scripts.calibrate', '<script>alert(1)</script>', '```', '',
    '| Method | Flags |', '| --- | --- |', '| AE | 14 |', '',
    '[Docs](https://example.com) [Unsafe](javascript:alert)', '',
    '<img src=x onerror=alert(1)>'
].join('\n');
context.markdownSample = markdownSample;
const rendered = vm.runInContext('renderMarkdown(markdownSample)', context);
const tags = descendants(rendered);
assert.equal(tags.filter(node => node.tagName === 'strong').length, 3);
for (const tag of ['ul', 'ol', 'li', 'em', 'code', 'pre', 'table', 'th', 'td']) assert.ok(tags.some(node => node.tagName === tag), tag);
assert.equal(tags.filter(node => node.tagName === 'a').length, 1);
assert.equal(tags.find(node => node.tagName === 'a').rel, 'noopener noreferrer');
assert.equal(tags.some(node => ['script', 'img'].includes(node.tagName)), false);
assert.match(renderedText(rendered), /<script>alert/);
console.log('Markdown formatting and inert HTML/unsafe links passed.');
