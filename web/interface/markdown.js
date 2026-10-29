// Render a small Markdown subset using DOM nodes only; raw HTML stays text.
function inline(parent, text, depth = 0) {
    if (depth > 8) {
        parent.append(document.createTextNode(text));
        return;
    }
    const pattern = /\*\*([^\n]+?)\*\*|__([^\n]+?)__|`([^`\n]+)`|\*([^*\n]+)\*|\[([^\]\n]+)\]\(([^\s)]+)\)/g;
    let offset = 0;
    for (const match of text.matchAll(pattern)) {
        parent.append(document.createTextNode(text.slice(offset, match.index)));
        let node;
        if (match[1] || match[2]) {
            node = document.createElement('strong');
            inline(node, match[1] || match[2], depth + 1);
        } else if (match[3]) {
            node = document.createElement('code');
            node.textContent = match[3];
        } else if (match[4]) {
            node = document.createElement('em');
            inline(node, match[4], depth + 1);
        } else {
            let address;
            try { address = new URL(match[6]); } catch { /* Keep invalid links as text. */ }
            if (address && ['https:', 'http:'].includes(address.protocol)) {
                node = document.createElement('a');
                node.href = address.href;
                node.target = '_blank';
                node.rel = 'noopener noreferrer';
                inline(node, match[5], depth + 1);
            } else {
                node = document.createTextNode(match[0]);
            }
        }
        parent.append(node);
        offset = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(offset)));
}

export function renderMarkdown(text) {
    const root = document.createElement('div');
    root.className = 'chat-markdown';
    const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
    const listItem = line => /^\s*(?:([-+*])|\d+[.)])\s+(.+)$/.exec(line);
    const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
    let index = 0;
    while (index < lines.length) {
        const line = lines[index];
        if (!line.trim()) { index++; continue; }
        const fence = /^\s*(`{3,}|~{3,})/.exec(line);
        if (fence) {
            const content = [];
            index++;
            while (index < lines.length && !lines[index].trim().startsWith(fence[1])) content.push(lines[index++]);
            if (index < lines.length) index++;
            const pre = document.createElement('pre');
            const code = document.createElement('code');
            code.textContent = content.join('\n');
            pre.append(code);
            root.append(pre);
            continue;
        }
        const heading = /^(#{1,6})\s+(.+)$/.exec(line);
        if (heading) {
            const node = document.createElement(`h${Math.min(heading[1].length + 2, 6)}`);
            inline(node, heading[2]);
            root.append(node);
            index++;
            continue;
        }
        if (line.includes('|') && index + 1 < lines.length && cells(lines[index + 1]).every(cell => /^:?-{3,}:?$/.test(cell))) {
            const table = document.createElement('table');
            const head = document.createElement('tr');
            for (const value of cells(line)) {
                const cell = document.createElement('th');
                inline(cell, value);
                head.append(cell);
            }
            table.append(head);
            index += 2;
            while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
                const row = document.createElement('tr');
                for (const value of cells(lines[index++])) {
                    const cell = document.createElement('td');
                    inline(cell, value);
                    row.append(cell);
                }
                table.append(row);
            }
            root.append(table);
            continue;
        }
        const item = listItem(line);
        if (item) {
            const ordered = !item[1];
            const list = document.createElement(ordered ? 'ol' : 'ul');
            if (ordered) list.start = Number(line.trim().match(/^\d+/)[0]);
            while (index < lines.length) {
                const next = listItem(lines[index]);
                if (!next || !next[1] !== ordered) break;
                const node = document.createElement('li');
                inline(node, next[2]);
                list.append(node);
                index++;
            }
            root.append(list);
            continue;
        }
        const paragraph = document.createElement(line.startsWith('> ') ? 'blockquote' : 'p');
        inline(paragraph, line.startsWith('> ') ? line.slice(2) : line);
        index++;
        while (index < lines.length && lines[index].trim() && !/^(?:\s*[-+*]\s|\s*\d+[.)]\s|#{1,6}\s|\s*`{3}|\s*~{3}|> )/.test(lines[index])) {
            paragraph.append(document.createTextNode('\n'));
            inline(paragraph, lines[index++]);
        }
        root.append(paragraph);
    }
    return root;
}
