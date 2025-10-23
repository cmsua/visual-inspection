const DEFAULT_BOARD = "./data/bad_example/aligned_images5.npy";

const COLORS = {
    pixel: "#ffe8a3",
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
    currentRow: 0,
    currentCol: 0,
};

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
    prevBtn: document.getElementById("prev-segment"),
    nextBtn: document.getElementById("next-segment"),
    markDamagedBtn: document.getElementById("mark-damaged"),
    markOkBtn: document.getElementById("mark-ok"),
    flagSkipped: document.getElementById("flag-skipped"),
    flagPixel: document.getElementById("flag-pixel"),
    flagAE: document.getElementById("flag-autoencoder"),
    minimapGrid: document.getElementById("minimap-grid"),
};

let segmentFetchToken = 0;

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

// --------------------------------------------------------------------------- //
// State utilities

function segmentKey(row, col) {
    return `${row},${col}`;
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
    if (manualDamaged(row, col)) {
        return "manual-damaged";
    }

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
    const linearIndex = state.currentRow * state.grid.cols + state.currentCol;
    const lastIndex = Math.max(0, totalSegments() - 1);
    dom.prevBtn.disabled = linearIndex <= 0;
    dom.nextBtn.disabled = linearIndex >= lastIndex;
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
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.type = "button";
            cell.setAttribute("role", "gridcell");
            cell.setAttribute("aria-label", `Segment row ${r + 1}, column ${c + 1}`);
            cell.textContent = "";
            cell.addEventListener("click", () => {
                setSegment(r, c);
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
    if (dom.runProgress) {
        dom.runProgress.classList.toggle("hidden", !active);
        dom.runProgress.setAttribute("aria-hidden", active ? "false" : "true");
    }
}

// --------------------------------------------------------------------------- //
// Actions

function totalSegments() {
    return state.grid.rows * state.grid.cols;
}

function currentLinearIndex() {
    return state.currentRow * state.grid.cols + state.currentCol;
}

function setSegmentByIndex(index) {
    const total = totalSegments();
    if (total === 0 || state.grid.cols <= 0) {
        return;
    }

    const clamped = Math.min(Math.max(index, 0), total - 1);
    state.currentRow = Math.floor(clamped / state.grid.cols);
    state.currentCol = clamped % state.grid.cols;
    renderSegmentState();
}

function setSegment(row, col) {
    if (
        state.grid.cols <= 0 ||
        row < 0 ||
        col < 0 ||
        row >= state.grid.rows ||
        col >= state.grid.cols
    ) {
        return;
    }
    const index = row * state.grid.cols + col;
    setSegmentByIndex(index);
}

function goToNextSegment() {
    const total = totalSegments();
    if (total === 0) {
        return;
    }

    const current = currentLinearIndex();
    if (current >= total - 1) {
        return;
    }
    setSegmentByIndex(current + 1);
}

function goToPreviousSegment() {
    const total = totalSegments();
    if (total === 0) {
        return;
    }

    const current = currentLinearIndex();
    if (current <= 0) {
        return;
    }
    setSegmentByIndex(current - 1);
}

async function handleLoadBoard() {
    const requestedPath = dom.boardPathInput.value.trim() || DEFAULT_BOARD;
    try {
        dom.runStatus.textContent = "Loading board...";
        const params = new URLSearchParams({ path: requestedPath });
        const payload = await fetchJSON(`/api/board?${params.toString()}`);
        state.boardPath = payload.board_path;
        state.grid.rows = payload.grid_shape.rows;
        state.grid.cols = payload.grid_shape.cols;
        state.segmentShape = payload.segment_shape;
        state.inspection = payload.inspection;
        updateManualDamagedSet(payload.damaged_segments);
        dom.boardPathInput.value = payload.board_path;

        state.currentRow = 0;
        state.currentCol = 0;
        updateUIAfterBoardChange();
        dom.runStatus.textContent = `Loaded ${payload.board_path}`;
    } catch (error) {
        console.error("Failed to load board:", error);
        dom.runStatus.textContent = `Error loading board: ${error.message ?? error}`;
    }
}

async function handleRunInspection() {
    try {
        dom.runStatus.textContent = "Running inspection...";
        setProgressActive(true);
        const payload = await fetchJSON("/api/run", {
            method: "POST",
            body: JSON.stringify({ path: state.boardPath }),
        });
        const completionMessage = payload.stdout.trim()
            ? `Inspection complete.\n${payload.stdout.trim()}`
            : "Inspection complete.";
        await handleLoadBoard();
        dom.runStatus.textContent = completionMessage;
    } catch (error) {
        console.error("Inspection failed:", error);
        dom.runStatus.textContent = `Inspection failed: ${error.message ?? error}`;
    } finally {
        setProgressActive(false);
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
        updateManualDamagedSet(response.damaged_segments);
        updateMinimapGrid();
        renderSegmentState();
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

    dom.prevBtn.addEventListener("click", () => {
        goToPreviousSegment();
    });

    dom.nextBtn.addEventListener("click", () => {
        goToNextSegment();
    });

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
}

// --------------------------------------------------------------------------- //
// Init

bindEvents();
handleLoadBoard();
