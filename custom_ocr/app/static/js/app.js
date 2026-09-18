let currentDocId = null;
let currentMetadata = null;
let currentOcrResult = null;
let currentExtraction = null;
let currentZoom = 1.0;

document.addEventListener('DOMContentLoaded', () => {
    initUI();
    checkHealth();
});

function initUI() {
    // Dropzone events
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');

    browseBtn.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Config accordion
    const configToggle = document.getElementById('config-toggle');
    const configBody = document.getElementById('config-body');
    configToggle.addEventListener('click', () => {
        configBody.style.display = configBody.style.display === 'none' ? 'flex' : 'none';
    });

    // Run Pipeline Button
    document.getElementById('process-pipeline-btn').addEventListener('click', runFullPipeline);

    // Load Demo Invoice Button
    document.getElementById('load-sample-btn').addEventListener('click', loadSampleInvoice);

    // Tabs
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');
        });
    });

    // Zoom controls
    document.getElementById('zoom-in').addEventListener('click', () => adjustZoom(0.15));
    document.getElementById('zoom-out').addEventListener('click', () => adjustZoom(-0.15));
    document.getElementById('zoom-reset').addEventListener('click', () => resetZoom());

    // JSON export buttons
    document.getElementById('copy-json-btn').addEventListener('click', copyJSONToClipboard);
    document.getElementById('export-json-btn').addEventListener('click', () => exportData('json'));
    document.getElementById('export-csv-btn').addEventListener('click', () => exportData('csv'));

    // Initialize Custom Model Key Manager & RAG Chat
    initKeyManager();
    initRAGChat();
}

async function checkHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('engine-status').textContent = 'Engine: Ready (v' + data.version + ')';
        }
    } catch (e) {
        document.getElementById('engine-status').textContent = 'Engine: Offline';
    }
}

async function handleFileUpload(file) {
    showLoader("Uploading & Validating Document...", "Executing Step 1: Type, size, and page validation");
    setStepper(1);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await response.json();
        currentDocId = data.doc_id;
        currentMetadata = data;

        // Render first page preview
        if (data.pages && data.pages.length > 0) {
            const firstPage = data.pages[0];
            const img = document.getElementById('doc-canvas-image');
            img.src = firstPage.image_url;
            img.onload = () => {
                document.getElementById('canvas-empty').style.display = 'none';
                clearOverlay();
            };
        }

        document.getElementById('process-pipeline-btn').disabled = false;
        hideLoader();

        // Run preview of preprocessing automatically
        runPreprocessingPreview();
    } catch (err) {
        hideLoader();
        alert("Upload Error: " + err.message);
    }
}

async function runPreprocessingPreview() {
    if (!currentDocId) return;
    setStepper(2);

    try {
        const opts = getPreprocessingOptions();
        const res = await fetch('/api/ocr/preprocess-only', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                doc_id: currentDocId,
                preprocessing: opts
            })
        });

        if (res.ok) {
            const data = await res.json();
            renderPreprocessingStages(data.steps, data.skew_angle);
        }
    } catch (e) {
        console.error("Preview error", e);
    }
}

function renderPreprocessingStages(steps, skewAngle) {
    const inspector = document.getElementById('prep-inspector');
    const tabsList = document.getElementById('stage-tabs-list');
    const previewImg = document.getElementById('stage-preview-img');
    const descText = document.getElementById('stage-desc-text');
    const skewDisplay = document.getElementById('skew-display');

    if (!steps || steps.length === 0) return;

    inspector.style.display = 'flex';
    tabsList.innerHTML = '';
    skewDisplay.textContent = `Skew: ${skewAngle}°`;

    steps.forEach((step, idx) => {
        const btn = document.createElement('button');
        btn.className = `stage-tab-btn ${idx === steps.length - 1 ? 'active' : ''}`;
        btn.textContent = step.title.split('.')[0];
        btn.title = step.title;

        btn.addEventListener('click', () => {
            tabsList.querySelectorAll('.stage-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            previewImg.src = step.image_url;
            descText.textContent = step.description;
        });

        tabsList.appendChild(btn);
    });

    // Show last clean stage by default
    const last = steps[steps.length - 1];
    previewImg.src = last.image_url;
    descText.textContent = last.description;
}

async function runFullPipeline() {
    if (!currentDocId) return;

    showLoader("Running OCR Pipeline...", "Steps 2-6: Preprocessing, Custom Detection, Recognition, Tables & Field Extraction");
    setStepper(3);

    const opts = getPreprocessingOptions();
    const engine = document.getElementById('engine-select').value;

    try {
        const response = await fetch('/api/ocr/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                doc_id: currentDocId,
                engine: engine,
                preprocessing: opts,
                extract_fields: true,
                extract_tables: true
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Pipeline failed");
        }

        const data = await response.json();
        currentOcrResult = data.ocr_result;
        currentExtraction = data.extraction_result;

        setStepper(4);
        setTimeout(() => setStepper(5), 250);
        setTimeout(() => setStepper(6), 500);
        setTimeout(() => {
            setStepper(7);
            onDocumentIndexedInRAG(currentDocId, currentMetadata);
        }, 750);

        // Update Stats
        document.getElementById('stats-confidence').textContent = `Confidence: ${Math.round((data.ocr_result.overall_confidence || 0) * 100)}%`;
        document.getElementById('stats-words').textContent = `Words: ${data.ocr_result.total_words || 0}`;
        document.getElementById('stats-time').textContent = `Time: ${Math.round(data.processing_time_total_ms || 0)} ms`;

        // Render Canvas Overlay with detected bounding boxes
        if (data.ocr_result.pages && data.ocr_result.pages.length > 0) {
            renderBoundingBoxes(data.ocr_result.pages[0]);
        }

        // Render Extracted Fields
        renderExtractedFields(data.extraction_result);

        // Render Tables
        renderTables(data.ocr_result.pages);

        // Render JSON
        document.getElementById('json-code-block').textContent = JSON.stringify(data, null, 2);

        hideLoader();
    } catch (err) {
        hideLoader();
        alert("OCR Pipeline Error: " + err.message);
    }
}

function renderBoundingBoxes(pageResult) {
    const overlay = document.getElementById('overlay-layer');
    overlay.innerHTML = '';

    const img = document.getElementById('doc-canvas-image');
    const naturalW = img.naturalWidth || 1;
    const naturalH = img.naturalHeight || 1;

    const words = pageResult.words || [];

    words.forEach(w => {
        const box = document.createElement('div');
        box.className = 'ocr-bbox';

        // Using percentage position for responsive overlay
        const leftPct = (w.bbox.x / naturalW) * 100;
        const topPct = (w.bbox.y / naturalH) * 100;
        const widthPct = (w.bbox.width / naturalW) * 100;
        const heightPct = (w.bbox.height / naturalH) * 100;

        box.style.left = `${leftPct}%`;
        box.style.top = `${topPct}%`;
        box.style.width = `${widthPct}%`;
        box.style.height = `${heightPct}%`;

        // Tooltip hover
        box.addEventListener('mouseenter', (e) => showTooltip(e, w));
        box.addEventListener('mouseleave', hideTooltip);

        overlay.appendChild(box);
    });
}

function showTooltip(e, word) {
    const tooltip = document.getElementById('bbox-tooltip');
    const textEl = document.getElementById('tooltip-text');
    const confEl = document.getElementById('tooltip-conf');
    const coordsEl = document.getElementById('tooltip-coords');

    textEl.textContent = `"${word.text}"`;
    confEl.textContent = `Conf: ${Math.round(word.confidence * 100)}%`;
    coordsEl.textContent = `[${word.bbox.x}, ${word.bbox.y}, ${word.bbox.width}x${word.bbox.height}]`;

    tooltip.style.left = `${e.clientX + 14}px`;
    tooltip.style.top = `${e.clientY + 14}px`;
    tooltip.style.display = 'block';
}

function hideTooltip() {
    document.getElementById('bbox-tooltip').style.display = 'none';
}

function renderExtractedFields(extraction) {
    const fieldsGrid = document.getElementById('primary-fields-grid');
    const kvGrid = document.getElementById('kv-pairs-grid');
    const fieldsBadge = document.getElementById('fields-badge');
    const docTypeBadge = document.getElementById('doc-type-badge');
    const docSummaryText = document.getElementById('doc-summary-text');

    fieldsGrid.innerHTML = '';
    kvGrid.innerHTML = '';

    if (!extraction) {
        fieldsGrid.innerHTML = '<div class="empty-text">No fields extracted.</div>';
        return;
    }

    docTypeBadge.textContent = (extraction.document_type || 'GENERIC').toUpperCase();
    docSummaryText.textContent = extraction.summary || "Entities parsed successfully.";

    const fields = extraction.fields || {};
    const fieldKeys = Object.keys(fields);
    fieldsBadge.textContent = fieldKeys.length;

    if (fieldKeys.length === 0) {
        fieldsGrid.innerHTML = '<div class="empty-text">No primary fields matched.</div>';
    } else {
        fieldKeys.forEach(k => {
            const f = fields[k];
            const card = document.createElement('div');
            card.className = 'field-card';
            card.innerHTML = `
                <div class="field-card-header">
                    <span class="field-label">${f.field_name}</span>
                    <span class="field-conf">${Math.round(f.confidence * 100)}%</span>
                </div>
                <div class="field-value">${f.value}</div>
            `;
            fieldsGrid.appendChild(card);
        });
    }

    const kvPairs = extraction.key_values || [];
    if (kvPairs.length === 0) {
        kvGrid.innerHTML = '<div class="empty-text">No spatial key-value pairs detected.</div>';
    } else {
        kvPairs.forEach(kv => {
            const item = document.createElement('div');
            item.className = 'kv-item';
            item.innerHTML = `
                <span class="kv-key">${kv.key}</span>
                <span class="kv-val">${kv.value}</span>
            `;
            kvGrid.appendChild(item);
        });
    }
}

function renderTables(pages) {
    const tablesContainer = document.getElementById('tables-container');
    const tablesBadge = document.getElementById('tables-badge');
    tablesContainer.innerHTML = '';

    const allTables = [];
    (pages || []).forEach(p => {
        (p.tables || []).forEach(t => allTables.push(t));
    });

    tablesBadge.textContent = allTables.length;

    if (allTables.length === 0) {
        tablesContainer.innerHTML = '<div class="empty-state"><p>No tabular structures detected on this document.</p></div>';
        return;
    }

    allTables.forEach(t => {
        const block = document.createElement('div');
        block.className = 'table-block';
        
        let headersHtml = '';
        if (t.headers && t.headers.length > 0) {
            headersHtml = `<tr>${t.headers.map(h => `<th>${h || '&nbsp;'}</th>`).join('')}</tr>`;
        }

        let rowsHtml = '';
        if (t.rows && t.rows.length > 0) {
            rowsHtml = t.rows.map(r => `<tr>${r.map(c => `<td>${c || '&nbsp;'}</td>`).join('')}</tr>`).join('');
        }

        block.innerHTML = `
            <div class="table-title">${t.table_id} (${t.num_rows} rows × ${t.num_cols} cols)</div>
            <table class="custom-data-table">
                <thead>${headersHtml}</thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
        tablesContainer.appendChild(block);
    });
}

function clearOverlay() {
    document.getElementById('overlay-layer').innerHTML = '';
}

function adjustZoom(delta) {
    currentZoom = Math.max(0.4, Math.min(3.0, currentZoom + delta));
    document.getElementById('canvas-wrapper').style.transform = `scale(${currentZoom})`;
}

function resetZoom() {
    currentZoom = 1.0;
    document.getElementById('canvas-wrapper').style.transform = 'scale(1)';
}

function setStepper(stepNum) {
    for (let i = 1; i <= 6; i++) {
        const el = document.getElementById(`step-nav-${i}`);
        if (el) {
            if (i <= stepNum) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        }
    }
}

function getPreprocessingOptions() {
    return {
        enable_deskew: document.getElementById('opt-deskew').checked,
        enable_contrast_enhancement: document.getElementById('opt-clahe').checked,
        enable_denoise: document.getElementById('opt-denoise').checked,
        enable_thresholding: document.getElementById('opt-thresh').checked,
        threshold_method: 'adaptive',
        target_dpi: 300
    };
}

function showLoader(title, subtext) {
    document.getElementById('loader-status').textContent = title;
    document.getElementById('loader-subtext').textContent = subtext;
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoader() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function copyJSONToClipboard() {
    const code = document.getElementById('json-code-block').textContent;
    navigator.clipboard.writeText(code);
    alert("Structured JSON copied to clipboard!");
}

function exportData(format) {
    if (!currentDocId) {
        alert("Please upload and process a document first.");
        return;
    }
    window.open(`/api/documents/${currentDocId}/export/${format}`, '_blank');
}

async function loadSampleInvoice() {
    showLoader("Generating Demo Invoice Document...", "Synthesizing test document with invoice entities & tables");
    
    // Draw synthetic canvas invoice image and submit to upload
    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 1500;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Title / Vendor
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 44px sans-serif";
    ctx.fillText("ACME CORPORATION", 80, 110);

    ctx.font = "20px sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.fillText("Enterprise Cloud & Software Solutions", 80, 145);
    ctx.fillText("Email: billing@acme-corp.com  |  Phone: +1 (555) 234-5678", 80, 175);

    // Invoice Header Box
    ctx.fillStyle = "#4f46e5";
    ctx.fillRect(850, 70, 270, 60);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 28px sans-serif";
    ctx.fillText("TAX INVOICE", 880, 112);

    // Metadata Key-Values
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 22px sans-serif";
    ctx.fillText("Invoice Number: INV-2026-8891", 850, 180);
    ctx.fillText("Invoice Date: 2026-09-10", 850, 220);
    ctx.fillText("Due Date: 2026-10-10", 850, 260);

    // Bill To
    ctx.fillStyle = "#334155";
    ctx.font = "bold 24px sans-serif";
    ctx.fillText("BILL TO:", 80, 250);
    ctx.font = "20px sans-serif";
    ctx.fillText("Global Tech Enterprises Inc.", 80, 285);
    ctx.fillText("742 Evergreen Terrace, Suite 400", 80, 315);
    ctx.fillText("contact@globaltech.io", 80, 345);

    // Table Header Grid
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 2;
    ctx.strokeRect(80, 420, 1040, 450);

    ctx.fillStyle = "#f1f5f9";
    ctx.fillRect(80, 420, 1040, 50);

    ctx.beginPath();
    ctx.moveTo(80, 470);
    ctx.lineTo(1120, 470);
    ctx.stroke();

    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 20px sans-serif";
    ctx.fillText("ITEM DESCRIPTION", 100, 452);
    ctx.fillText("QTY", 650, 452);
    ctx.fillText("UNIT PRICE", 780, 452);
    ctx.fillText("TOTAL", 980, 452);

    // Table Rows
    const items = [
        ["Cloud Server Infrastructure Cluster (Monthly)", "1", "$ 1,200.00", "$ 1,200.00"],
        ["Antigravity AI Automation Agent License", "5", "$ 250.00", "$ 1,250.00"],
        ["Dedicated Database Storage 2TB SSD", "2", "$ 180.00", "$ 360.00"],
        ["Premium 24/7 SLA Technical Support", "1", "$ 500.00", "$ 500.00"]
    ];

    let y = 525;
    ctx.font = "19px sans-serif";
    ctx.fillStyle = "#334155";

    items.forEach(row => {
        ctx.fillText(row[0], 100, y);
        ctx.fillText(row[1], 660, y);
        ctx.fillText(row[2], 790, y);
        ctx.fillText(row[3], 980, y);

        ctx.beginPath();
        ctx.moveTo(80, y + 25);
        ctx.lineTo(1120, y + 25);
        ctx.stroke();
        y += 75;
    });

    // Summary Totals
    const totalX = 750;
    ctx.font = "bold 22px sans-serif";
    ctx.fillText("Subtotal: $ 3,310.00", totalX, 940);
    ctx.fillText("Tax Amount: $ 331.00", totalX, 980);

    ctx.fillStyle = "#4f46e5";
    ctx.fillRect(totalX - 20, 1010, 390, 60);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 26px sans-serif";
    ctx.fillText("Total Amount: $ 3,641.00", totalX, 1050);

    // Footer
    ctx.fillStyle = "#94a3b8";
    ctx.font = "16px sans-serif";
    ctx.fillText("Thank you for your business! Payment is due within 30 days of invoice date.", 80, 1380);

    // Convert canvas to blob and upload
    canvas.toBlob(async (blob) => {
        const file = new File([blob], "demo_invoice_acme.png", { type: "image/png" });
        await handleFileUpload(file);
    }, "image/png");
}

// ==============================================================================
// Custom Model & RAG Chat Integration
// ==============================================================================
let activeApiKey = "sk-custom-ocr-master-v1.1";

function initRAGChat() {
    const inputForm = document.getElementById('rag-input-form');
    const inputField = document.getElementById('rag-query-input');

    if (inputForm) {
        inputForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const q = inputField.value.trim();
            if (!q) return;
            inputField.value = '';
            executeRAGQuery(q);
        });
    }

    // Suggestions chips
    document.querySelectorAll('.sugg-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-q');
            if (prompt) {
                executeRAGQuery(prompt);
            }
        });
    });
}

function onDocumentIndexedInRAG(docId, metadata) {
    const indicator = document.getElementById('rag-active-doc');
    if (indicator) {
        indicator.textContent = `Document: ${metadata?.original_filename || docId} (Indexed & Ready)`;
        indicator.style.color = '#34d399';
    }

    const messages = document.getElementById('rag-messages');
    if (messages) {
        const notice = document.createElement('div');
        notice.className = 'rag-msg assistant';
        notice.innerHTML = `
            <div class="msg-avatar">AI</div>
            <div class="msg-bubble">
                <p>✅ <strong>Document indexed into local vector store!</strong></p>
                <p>All lines, fields, and tables from <code>${metadata?.original_filename || docId}</code> are now queryable via your <strong>100% free Custom Model</strong>.</p>
            </div>
        `;
        messages.appendChild(notice);
        messages.scrollTop = messages.scrollHeight;
    }
}

async function executeRAGQuery(question) {
    if (!currentDocId) {
        alert("Please upload and run OCR on a document first, or ingest sample text.");
        return;
    }

    const messages = document.getElementById('rag-messages');
    
    // Add user message bubble
    const userDiv = document.createElement('div');
    userDiv.className = 'rag-msg user';
    userDiv.innerHTML = `
        <div class="msg-avatar">You</div>
        <div class="msg-bubble">${escapeHTML(question)}</div>
    `;
    messages.appendChild(userDiv);

    // Add loading assistant bubble
    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'rag-msg assistant';
    assistantDiv.innerHTML = `
        <div class="msg-avatar">AI</div>
        <div class="msg-bubble">
            <span class="pulse-dot" style="display:inline-block; margin-right:6px;"></span>
            <em>Custom model is reasoning over document citations...</em>
        </div>
    `;
    messages.appendChild(assistantDiv);
    messages.scrollTop = messages.scrollHeight;

    try {
        const response = await fetch('/api/rag/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${activeApiKey}`
            },
            body: JSON.stringify({
                doc_id: currentDocId,
                query: question,
                top_k: 4
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail?.error || errData.detail || "Query failed");
        }

        const data = await response.json();
        renderAssistantAnswer(assistantDiv, data);
    } catch (err) {
        assistantDiv.querySelector('.msg-bubble').innerHTML = `
            <p style="color: #ef4444;">❌ Query Error: ${escapeHTML(err.message)}</p>
            <p style="font-size:0.75rem; color:#94a3b8;">Make sure your Custom Model API Key is valid.</p>
        `;
    }
    messages.scrollTop = messages.scrollHeight;
}

function renderAssistantAnswer(bubbleElem, ragData) {
    let citationsHtml = '';
    if (ragData.citations && ragData.citations.length > 0) {
        const badges = ragData.citations.map(c => `
            <span class="citation-chip" title="${escapeHTML(c.text)}">
                📄 Page ${c.page_number} (${Math.round(c.score * 100)}% match)
            </span>
        `).join('');

        citationsHtml = `
            <div class="msg-citations">
                <span style="font-size: 0.72rem; color: #94a3b8; font-weight:600;">Grounded Citations:</span>
                <div class="citation-badge-list">${badges}</div>
            </div>
        `;
    }

    const confScore = Math.round((ragData.confidence_score || 0) * 100);
    const confBadge = `<span style="font-size:0.72rem; color:#34d399; margin-left:8px;">[Confidence: ${confScore}%]</span>`;

    bubbleElem.querySelector('.msg-bubble').innerHTML = `
        <div>${formatMarkdownBasic(ragData.answer)} ${confBadge}</div>
        ${citationsHtml}
    `;
}

function formatMarkdownBasic(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

// ==============================================================================
// Custom Model API Key Manager
// ==============================================================================
function initKeyManager() {
    const openBtn = document.getElementById('open-keys-modal-btn');
    const closeBtn = document.getElementById('close-keys-modal');
    const modal = document.getElementById('keys-modal');
    const copyKeyBtn = document.getElementById('copy-active-key-btn');
    const createKeyForm = document.getElementById('create-key-form');

    if (openBtn) {
        openBtn.addEventListener('click', () => {
            modal.style.display = 'flex';
            loadKeys();
            loadActivePrimary();
            updateSnippetCode('flutter');
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.style.display = 'none';
        });
    }

    if (copyKeyBtn) {
        copyKeyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(activeApiKey);
            copyKeyBtn.textContent = 'Copied!';
            setTimeout(() => copyKeyBtn.textContent = 'Copy Key', 1500);
        });
    }

    if (createKeyForm) {
        createKeyForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const nameInput = document.getElementById('new-key-name');
            const name = nameInput.value.trim();
            if (!name) return;

            try {
                const res = await fetch('/api/keys', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                if (res.ok) {
                    nameInput.value = '';
                    await loadKeys();
                }
            } catch (err) {
                alert("Failed to create key: " + err.message);
            }
        });
    }

    // Snippet tabs
    document.querySelectorAll('.snippet-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.snippet-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateSnippetCode(btn.getAttribute('data-lang'));
        });
    });

    // Initial silent load of primary key
    loadActivePrimary();
}

async function loadActivePrimary() {
    try {
        const res = await fetch('/api/keys/active-primary');
        if (res.ok) {
            const data = await res.json();
            activeApiKey = data.api_key;
            const display = document.getElementById('active-key-display');
            if (display) display.textContent = activeApiKey;
        }
    } catch (e) {
        console.warn("Could not fetch active primary key", e);
    }
}

async function loadKeys() {
    const tableBody = document.getElementById('keys-table-body');
    if (!tableBody) return;

    try {
        const res = await fetch('/api/keys');
        if (res.ok) {
            const data = await res.json();
            const keys = data.keys || [];
            tableBody.innerHTML = keys.map(k => `
                <tr>
                    <td><strong>${escapeHTML(k.name)}</strong></td>
                    <td><code>${escapeHTML(k.masked_key)}</code></td>
                    <td>${k.request_count}</td>
                    <td><span class="${k.is_active ? 'free-pill' : 'badge'}">${k.is_active ? 'Active' : 'Revoked'}</span></td>
                    <td>
                        ${k.is_active && k.key_id !== 'master-default' ? `
                            <button class="revoke-key-btn" onclick="revokeKey('${k.key_id}')">Revoke</button>
                        ` : '<span style="color:#64748b; font-size:0.72rem;">Default</span>'}
                    </td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.warn("Could not load keys", e);
    }
}

window.revokeKey = async function(keyId) {
    if (!confirm("Revoke this Custom Model API Key?")) return;
    try {
        const res = await fetch(`/api/keys/${keyId}`, { method: 'DELETE' });
        if (res.ok) {
            await loadKeys();
        }
    } catch (err) {
        alert("Failed to revoke: " + err.message);
    }
};

function updateSnippetCode(lang) {
    const box = document.getElementById('snippet-code-box');
    if (!box) return;

    const host = window.location.origin;
    if (lang === 'flutter') {
        box.textContent = `import 'dart:io';
import 'package:your_app/services/custom_ocr_client.dart';

final client = CustomOCRClient(
  baseUrl: '${host}',
  apiKey: '${activeApiKey}',
);

// 1. Process local image or PDF file in 1 line
final result = await client.processFile(File('/path/to/invoice.png'));

print('Recognized Text: \${result.rawText}');
print('Invoice #: \${result.fields['invoice_number']?.value}');
print('Total Amount: \${result.fields['total_amount']?.value}');`;
    } else if (lang === 'python') {
        box.textContent = `from custom_ocr_client import CustomOCRClient

client = CustomOCRClient(
    base_url="${host}",
    api_key="${activeApiKey}"
)

# 1. Process any image or PDF
result = client.process_file("invoice.png")

print("Recognized Text:", result["raw_text"])
print("Extracted Fields:", result["fields"])
print("Extracted Tables:", result["tables"])`;
    } else if (lang === 'curl') {
        box.textContent = `curl -X POST "${host}/api/ocr/predict" \\
     -H "Authorization: Bearer ${activeApiKey}" \\
     -F "file=@invoice.png" \\
     -F "engine=auto" \\
     -F "extract_fields=true" \\
     -F "extract_tables=true"`;
    } else if (lang === 'base64') {
        box.textContent = `// Direct Base64 OCR Inference (Mobile camera / Web canvas)
const res = await fetch("${host}/api/ocr/predict-base64", {
  method: "POST",
  headers: {
    "Authorization": "Bearer ${activeApiKey}",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    image_base64: "data:image/png;base64,...",
    extract_fields: true,
    extract_tables: true
  })
});
const data = await res.json();
console.log("Extracted Fields:", data.fields);`;
    }
}
