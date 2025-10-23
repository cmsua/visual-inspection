const DEFAULT_BOARD = "./data/bad_example/aligned_images5.npy";

const COLORS = {
    pixel: "#cadef5",
    auto: "#ffd7a3",
    both: "#ffcccc",
    skipped: "#9ba3b3",
    neutral: "#b6bed3",
};

const state = {
    boardPath: DEFAULT_BOARD,
    grid: { rows: 0, cols: 0 },
    segmentShape: { height: 0, width: 0, channels: 0 },
    inspection: {
        pw_flags: [],
        ae_flags: [],
        hybrid_flags: [],
        metadata: {},
    },
    damagedSegments: new Set(),
    visitedSegments: new Set(),
    currentRow: 0,
    currentCol: 0,
    isRunningInspection: false,
    isDarkMode: false,
};

const THEME_STORAGE_KEY = "vi-theme-mode";

// --------------------------------------------------------------------------- //
// DOM helpers

const dom = {
    boardPathInput: document.getElementById("board-path"),
    loadBoardBtn: document.getElementById("load-board"),
    runButton: document.getElementById("run-inspection"),
    runStatus: document.getElementById("run-status"),
    runProgress: document.getElementById("run-progress"),
    segmentImage: document.getElementById("segment-image"),
    imageLoader: document.getElementById("image-loader"),
    segmentLabel: document.getElementById("segment-label"),
    navUp: document.getElementById("nav-up"),
    navDown: document.getElementById("nav-down"),
    navLeft: document.getElementById("nav-left"),
    navRight: document.getElementById("nav-right"),
    markDamagedBtn: document.getElementById("mark-damaged"),
    markOkBtn: document.getElementById("mark-ok"),
    flagSkipped: document.getElementById("flag-skipped"),
    flagPixel: document.getElementById("flag-pixel"),
    flagAE: document.getElementById("flag-autoencoder"),
    minimapGrid: document.getElementById("minimap-grid"),
    themeToggle: document.getElementById("theme-toggle"),
};

let segmentFetchToken = 0;
let activeRunToken = 0;

// --------------------------------------------------------------------------- //
// Networking

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!response.ok) {
        const text = await response.text();
        let message = text || response.statusText;
        try {
            const data = JSON.parse(text);
            if (data && typeof data === "object") {
                message =
                    data.error ||
                    data.stderr ||
                    data.stdout ||
                    data.status ||
                    JSON.stringify(data);
            }
        } catch (_err) {
            // ignore parse failure
        }
        throw new Error(message);
    }
    return response.json();
}

function applyTheme(isDark) {
    state.isDarkMode = isDark;
    document.body.classList.toggle("dark-mode", isDark);
    if (dom.themeToggle) {
        dom.themeToggle.setAttribute("aria-pressed", String(isDark));
    }
}

function initializeTheme() {
    try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        if (stored === "dark" || stored === "light") {
            applyTheme(stored === "dark");
            return;
        }
    } catch (error) {
        console.warn("Unable to access theme preference:", error);
    }

    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(prefersDark);

    if (window.matchMedia) {
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        const listener = (event) => {
            try {
                if (localStorage.getItem(THEME_STORAGE_KEY)) {
                    return;
                }
            } catch (error) {
                console.warn("Unable to access theme preference:", error);
            }
            applyTheme(event.matches);
        };

        if (typeof mediaQuery.addEventListener === "function") {
            mediaQuery.addEventListener("change", listener);
        } else if (typeof mediaQuery.addListener === "function") {
            mediaQuery.addListener(listener);
        }
    }
}

// --------------------------------------------------------------------------- //
// State utilities

function segmentKey(row, col) {
    return `${row},${col}`;
}

function markVisited(row, col) {
    state.visitedSegments.add(segmentKey(row, col));
}

function clearVisited(row, col) {
    state.visitedSegments.delete(segmentKey(row, col));
}

function syncVisitedWithDamaged() {
    for (const key of Array.from(state.visitedSegments)) {
        if (state.damagedSegments.has(key)) {
            state.visitedSegments.delete(key);
        }
    }
}

function isSkipped(row, col) {
    if (!state.inspection.pw_flags.length) return false;
    return state.inspection.pw_flags[row][col] < 0;
}

function isPixelFlagged(row, col) {
    if (!state.inspection.pw_flags.length) return false;
    return state.inspection.pw_flags[row][col] === 1;
}

function isAEFlagged(row, col) {
    if (!state.inspection.ae_flags.length) return false;
    return state.inspection.ae_flags[row][col] === 1;
}

function minimapClass(row, col) {
    if (isSkipped(row, col)) {
        return "status-skipped";
    }

    const pixel = isPixelFlagged(row, col);
    const ae = isAEFlagged(row, col);
    if (pixel && ae) {
        return "status-double";
    }
    if (pixel) {
        return "status-pixel";
    }
    if (ae) {
        return "status-auto";
    }
    return "status-ok";
}

function manualDamaged(row, col) {
    return state.damagedSegments.has(segmentKey(row, col));
}

// --------------------------------------------------------------------------- //
// UI updates

function updateSegmentLabel() {
    const { currentRow, currentCol, grid, segmentShape } = state;
    dom.segmentLabel.textContent = `Segment (${currentRow + 1}, ${currentCol + 1}) / (${grid.rows}, ${grid.cols}) — ${segmentShape.height}×${segmentShape.width}`;
}

function setAccentColor(element, color) {
    if (element) {
        element.style.setProperty("--accent-color", color);
    }
}

function updateFlagCheckboxes() {
    const { currentRow, currentCol } = state;
    const skipped = isSkipped(currentRow, currentCol);
    const pixel = isPixelFlagged(currentRow, currentCol);
    const ae = isAEFlagged(currentRow, currentCol);

    dom.flagSkipped.checked = skipped;
    dom.flagPixel.checked = pixel;
    dom.flagAE.checked = ae;

    const pixelColor = pixel ? (ae ? COLORS.both : COLORS.pixel) : COLORS.neutral;
    const aeColor = ae ? (pixel ? COLORS.both : COLORS.auto) : COLORS.neutral;
    const skippedColor = skipped ? COLORS.skipped : COLORS.neutral;

    setAccentColor(dom.flagPixel, pixelColor);
    setAccentColor(dom.flagAE, aeColor);
    setAccentColor(dom.flagSkipped, skippedColor);
}

function updateManualControls() {
    const disabled = isSkipped(state.currentRow, state.currentCol);
    dom.markDamagedBtn.disabled = disabled;
    dom.markOkBtn.disabled = disabled;
}

function updateNavigationButtons() {
    const { rows, cols } = state.grid;
    const { currentRow } = state;
    const disableAll = rows <= 0 || cols <= 0;
    const total = totalSegments();
    const index = currentLinearIndex();

    if (dom.navUp) {
        dom.navUp.disabled = disableAll || currentRow <= 0;
    }
    if (dom.navDown) {
        dom.navDown.disabled = disableAll || currentRow >= rows - 1;
    }
    if (dom.navLeft) {
        dom.navLeft.disabled = disableAll || index <= 0;
    }
    if (dom.navRight) {
        dom.navRight.disabled = disableAll || index >= total - 1;
    }
}

function updateMinimapSelection() {
    const cells = dom.minimapGrid.querySelectorAll(".segment-cell");
    cells.forEach((cell) => cell.classList.remove("current"));

    const selector = `.segment-cell[data-row="${state.currentRow}"][data-col="${state.currentCol}"]`;
    const currentCell = dom.minimapGrid.querySelector(selector);
    if (currentCell) {
        currentCell.classList.add("current");
    }
}

function updateMinimapGrid() {
    const { rows, cols } = state.grid;
    dom.minimapGrid.innerHTML = "";
    dom.minimapGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    dom.minimapGrid.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    if (state.segmentShape.height > 0 && state.segmentShape.width > 0) {
        const aspect = state.segmentShape.width / state.segmentShape.height;
        dom.minimapGrid.style.setProperty("--segment-aspect", aspect.toFixed(3));
    }

    for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
            const cell = document.createElement("button");
            cell.className = `segment-cell ${minimapClass(r, c)}`;
            if (manualDamaged(r, c)) {
                cell.classList.add("manual-damaged");
            } else if (state.visitedSegments.has(segmentKey(r, c))) {
                cell.classList.add("visited-ok");
            }
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.type = "button";
            cell.setAttribute("role", "gridcell");
            cell.setAttribute("aria-label", `Segment row ${r + 1}, column ${c + 1}`);
            cell.textContent = "";
            cell.addEventListener("click", () => {
                setSegment(r, c);
                cell.blur();
            });
            dom.minimapGrid.appendChild(cell);
        }
    }

    updateMinimapSelection();
}

async function refreshSegmentImage() {
    const token = ++segmentFetchToken;
    dom.imageLoader.classList.remove("hidden");

    try {
        const params = new URLSearchParams({
            path: state.boardPath,
            row: state.currentRow,
            col: state.currentCol,
        });
        const data = await fetchJSON(`/api/segment?${params.toString()}`);
        if (token !== segmentFetchToken) {
            return;
        }
        dom.segmentImage.src = `data:image/png;base64,${data.image}`;
        dom.segmentImage.alt = `Segment (${state.currentRow + 1}, ${state.currentCol + 1})`;
    } catch (error) {
        console.error("Failed to load segment image:", error);
        if (token === segmentFetchToken) {
            dom.segmentImage.src = "";
            dom.segmentImage.alt = "Failed to load segment";
        }
    } finally {
        if (token === segmentFetchToken) {
            dom.imageLoader.classList.add("hidden");
        }
    }
}

function updateManualDamagedSet(damagedList) {
    state.damagedSegments = new Set(damagedList.map(({ row, col }) => segmentKey(row, col)));
    syncVisitedWithDamaged();
}

function renderSegmentState() {
    updateSegmentLabel();
    updateFlagCheckboxes();
    updateManualControls();
    updateNavigationButtons();
    updateMinimapSelection();
    refreshSegmentImage();
}

function updateUIAfterBoardChange() {
    updateMinimapGrid();
    renderSegmentState();
}

function setProgressActive(active) {
    if (!dom.runProgress) {
        return;
    }

    dom.runProgress.classList.toggle("hidden", !active);
    dom.runProgress.setAttribute("aria-hidden", active ? "false" : "true");
}

function finalizeInspectionRun(runToken) {
    if (activeRunToken !== runToken) {
        return;
    }

    activeRunToken = 0;
    state.isRunningInspection = false;
    setProgressActive(false);
}

// --------------------------------------------------------------------------- //
// Actions

function totalSegments() {
    return state.grid.rows * state.grid.cols;
}

function currentLinearIndex() {
    return state.currentRow * state.grid.cols + state.currentCol;
}

function setSegment(row, col) {
    const { rows, cols } = state.grid;
    if (rows <= 0 || cols <= 0) {
        return;
    }
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        return;
    }
    state.currentRow = row;
    state.currentCol = col;
    renderSegmentState();
}

function moveToLinear(index) {
    const total = totalSegments();
    if (total <= 0) {
        return false;
    }
    if (index < 0 || index >= total) {
        return false;
    }
    const cols = state.grid.cols;
    const row = Math.floor(index / cols);
    const col = index % cols;
    setSegment(row, col);
    return true;
}

function moveNextSegment() {
    const nextIndex = currentLinearIndex() + 1;
    return moveToLinear(nextIndex);
}

function movePreviousSegment() {
    const prevIndex = currentLinearIndex() - 1;
    return moveToLinear(prevIndex);
}

function moveUp() {
    if (state.currentRow <= 0) {
        return false;
    }
    setSegment(state.currentRow - 1, state.currentCol);
    return true;
}

function moveDown() {
    if (state.currentRow >= state.grid.rows - 1) {
        return false;
    }
    setSegment(state.currentRow + 1, state.currentCol);
    return true;
}

function moveLeft() {
    if (currentLinearIndex() === 0) {
        return false;
    }
    return movePreviousSegment();
}

function moveRight() {
    if (currentLinearIndex() >= totalSegments() - 1) {
        return false;
    }
    return moveNextSegment();
}

function advanceAfterLabel() {
    if (currentLinearIndex() >= totalSegments() - 1) {
        return;
    }
    moveNextSegment();
}

function handleKeyboardNavigation(event) {
    const { key } = event;
    if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) {
        return;
    }

    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
        return;
    }

    const target = event.target;
    if (target) {
        const tag = target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable) {
            return;
        }
    }

    const gridLoaded = state.grid.rows > 0 && state.grid.cols > 0;

    let handled = false;
    let attempted = false;
    switch (key) {
        case "ArrowUp":
            attempted = true;
            handled = moveUp();
            break;
        case "ArrowDown":
            attempted = true;
            handled = moveDown();
            break;
        case "ArrowLeft":
            attempted = true;
            handled = moveLeft();
            break;
        case "ArrowRight":
            attempted = true;
            handled = moveRight();
            break;
        default:
            break;
    }

    if ((attempted && gridLoaded) || handled) {
        event.preventDefault();
    }
}

async function handleLoadBoard(options = {}) {
    const { showStatus = true } = options;
    const requestedPath = dom.boardPathInput.value.trim() || DEFAULT_BOARD;
    try {
        if (showStatus) {
            dom.runStatus.textContent = "Loading board...";
        }
        const params = new URLSearchParams({ path: requestedPath });
        const payload = await fetchJSON(`/api/board?${params.toString()}`);
        const previousBoardPath = state.boardPath;
        state.boardPath = payload.board_path;
        state.grid.rows = payload.grid_shape.rows;
        state.grid.cols = payload.grid_shape.cols;
        state.segmentShape = payload.segment_shape;
        state.inspection = payload.inspection;
        updateManualDamagedSet(payload.damaged_segments);
        if (previousBoardPath !== state.boardPath) {
            state.visitedSegments.clear();
        }
        dom.boardPathInput.value = payload.board_path;

        state.currentRow = 0;
        state.currentCol = 0;
        updateUIAfterBoardChange();
        if (showStatus && !state.isRunningInspection) {
            dom.runStatus.textContent = `Loaded ${payload.board_path}`;
        }
    } catch (error) {
        console.error("Failed to load board:", error);
        dom.runStatus.textContent = `Error loading board: ${error.message ?? error}`;
    }
}

async function handleRunInspection() {
    const runToken = Date.now();
    try {
        dom.runStatus.textContent = "Running inspection...";
        activeRunToken = runToken;
        state.isRunningInspection = true;
        setProgressActive(true);
        const payload = await fetchJSON("/api/run", {
            method: "POST",
            body: JSON.stringify({ path: state.boardPath }),
        });
        if (activeRunToken !== runToken) {
            return;
        }
        const completionMessage = payload.stdout.trim()
            ? `Inspection complete.\n${payload.stdout.trim()}`
            : "Inspection complete.";
        await handleLoadBoard({ showStatus: false });
        dom.runStatus.textContent = completionMessage;
        finalizeInspectionRun(runToken);
    } catch (error) {
        if (activeRunToken === runToken) {
            console.error("Inspection failed:", error);
            dom.runStatus.textContent = `Inspection failed: ${error.message ?? error}`;
            finalizeInspectionRun(runToken);
        }
    }
}

async function handleManualLabel(isDamaged) {
    try {
        dom.runStatus.textContent = isDamaged ? "Marking segment as damaged..." : "Marking segment as OK...";
        const response = await fetchJSON("/api/label", {
            method: "POST",
            body: JSON.stringify({
                path: state.boardPath,
                row: state.currentRow,
                col: state.currentCol,
                damaged: isDamaged,
            }),
        });
        const prevIndex = segmentKey(state.currentRow, state.currentCol);
        updateManualDamagedSet(response.damaged_segments);
        clearVisited(state.currentRow, state.currentCol);
        if (!state.damagedSegments.has(prevIndex)) {
            markVisited(state.currentRow, state.currentCol);
        }
        updateMinimapGrid();
        renderSegmentState();
        advanceAfterLabel();
        dom.runStatus.textContent = isDamaged ? "Segment marked as damaged." : "Segment marked as OK.";
    } catch (error) {
        console.error("Failed to persist label:", error);
        dom.runStatus.textContent = `Failed to update label: ${error.message ?? error}`;
    }
}

// --------------------------------------------------------------------------- //
// Event bindings

function bindEvents() {
    dom.loadBoardBtn.addEventListener("click", () => {
        handleLoadBoard();
    });

    dom.runButton.addEventListener("click", () => {
        handleRunInspection();
    });

    if (dom.themeToggle) {
        dom.themeToggle.addEventListener("click", () => {
            const next = !state.isDarkMode;
            applyTheme(next);
            try {
                localStorage.setItem(THEME_STORAGE_KEY, next ? "dark" : "light");
            } catch (error) {
                console.warn("Unable to store theme preference:", error);
            }
        });
    }

    if (dom.navUp) {
        dom.navUp.addEventListener("click", () => {
            moveUp();
        });
    }

    if (dom.navDown) {
        dom.navDown.addEventListener("click", () => {
            moveDown();
        });
    }

    if (dom.navLeft) {
        dom.navLeft.addEventListener("click", () => {
            moveLeft();
        });
    }

    if (dom.navRight) {
        dom.navRight.addEventListener("click", () => {
            moveRight();
        });
    }

    dom.markDamagedBtn.addEventListener("click", () => {
        handleManualLabel(true);
    });

    dom.markOkBtn.addEventListener("click", () => {
        handleManualLabel(false);
    });

    dom.boardPathInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            handleLoadBoard();
        }
    });

    document.addEventListener("keydown", handleKeyboardNavigation);
}

// --------------------------------------------------------------------------- //
// Init

initializeTheme();
bindEvents();
updateNavigationButtons();
handleLoadBoard();
