// Lunar Crater Feature Detection & Topological Graph Matching — Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const folderInput = document.getElementById("folderInput");
    const browseBtn = document.getElementById("browseBtn");
    const browseFolderBtn = document.getElementById("browseFolderBtn");
    const clearFilesBtn = document.getElementById("clearFilesBtn");
    const fileCount = document.getElementById("fileCount");
    const fileTagsContainer = document.getElementById("fileTagsContainer");
    const anchorRow = document.getElementById("anchorRow");
    const anchorSelect = document.getElementById("anchorSelect");
    const processBtn = document.getElementById("processBtn");
    const runDemoBtn = document.getElementById("runDemoBtn");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const resultsSection = document.getElementById("resultsSection");

    // Primary System Mode Switcher Elements
    const btnModeInference = document.getElementById("btnModeInference");
    const btnModeTraining = document.getElementById("btnModeTraining");
    const inferenceModeSection = document.getElementById("inferenceModeSection");
    const trainingModeSection = document.getElementById("trainingModeSection");
    const switchToInferenceBtn = document.getElementById("switchToInferenceBtn");

    // Training Workspace Elements
    const loadAngleDemoBtn = document.getElementById("loadAngleDemoBtn");
    const trainingDropzone = document.getElementById("trainingDropzone");
    const trainingFileInput = document.getElementById("trainingFileInput");
    const browseTrainingBtn = document.getElementById("browseTrainingBtn");
    const anglePreviewsGrid = document.getElementById("anglePreviewsGrid");
    const trainEpochs = document.getElementById("trainEpochs");
    const trainLr = document.getElementById("trainLr");
    const trainLossType = document.getElementById("trainLossType");
    const trainOptimizer = document.getElementById("trainOptimizer");
    const mathLossFormula = document.getElementById("mathLossFormula");
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stepTrainBtn = document.getElementById("stepTrainBtn");
    const resetWeightsBtn = document.getElementById("resetWeightsBtn");
    const trainProgressContainer = document.getElementById("trainProgressContainer");
    const trainProgressBar = document.getElementById("trainProgressBar");
    const trainProgressLabel = document.getElementById("trainProgressLabel");
    const trainProgressPct = document.getElementById("trainProgressPct");
    const trainingTelemetrySection = document.getElementById("trainingTelemetrySection");
    const kpiLoss = document.getElementById("kpiLoss");
    const kpiLossDelta = document.getElementById("kpiLossDelta");
    const kpiGradNorm = document.getElementById("kpiGradNorm");
    const kpiWeightDelta = document.getElementById("kpiWeightDelta");
    const kpiSimilarity = document.getElementById("kpiSimilarity");
    const kpiSimGain = document.getElementById("kpiSimGain");
    const chartEpochCountPill = document.getElementById("chartEpochCountPill");
    const lossChartCanvas = document.getElementById("lossChartCanvas");
    const visualComparisonRow = document.getElementById("visualComparisonRow");
    const trainingConsoleLog = document.getElementById("trainingConsoleLog");
    const trainModelBadge = document.getElementById("trainModelBadge");
    const trainParamsBadge = document.getElementById("trainParamsBadge");
    const trainStepsBadge = document.getElementById("trainStepsBadge");

    // Mode View Tabs
    const tabGraphMatchBtn = document.getElementById("tabGraphMatchBtn");
    const tabDetectedFeaturesBtn = document.getElementById("tabDetectedFeaturesBtn");
    const viewGraphMatch = document.getElementById("viewGraphMatch");
    const viewDetectedFeatures = document.getElementById("viewDetectedFeatures");

    // Pair Selector Elements
    const selectImageA = document.getElementById("selectImageA");
    const selectImageB = document.getElementById("selectImageB");
    const swapImagesBtn = document.getElementById("swapImagesBtn");
    const runPairMatchBtn = document.getElementById("runPairMatchBtn");
    const graphMatchImg = document.getElementById("graphMatchImg");
    const graphMatchTitle = document.getElementById("graphMatchTitle");
    const matchCountVal = document.getElementById("matchCountVal");
    const craterMatchCountVal = document.getElementById("craterMatchCountVal");
    const graphEdgeCountVal = document.getElementById("graphEdgeCountVal");
    const matchedFeaturesTableBody = document.getElementById("matchedFeaturesTableBody");
    const matchedCountBadge = document.getElementById("matchedCountBadge");

    // Single Image Feature View Elements
    const singleImageSelect = document.getElementById("singleImageSelect");
    const singleImgDim = document.getElementById("singleImgDim");
    const singleImgCraters = document.getElementById("singleImgCraters");
    const singleImgFeatures = document.getElementById("singleImgFeatures");
    const singleImgTitle = document.getElementById("singleImgTitle");
    const singleCraterBadge = document.getElementById("singleCraterBadge");
    const singleAnnotatedImg = document.getElementById("singleAnnotatedImg");
    const singleCraterTableBody = document.getElementById("singleCraterTableBody");

    // State
    let uploadedFiles = [];
    let designatedAnchorIndex = 0;
    let jobData = null;
    let currentGraphMatch = null;
    let activeGraphViewMode = "side_by_side"; // "side_by_side", "photo_a", "photo_b"
    let isMatchingPair = false;

    // Check API Health
    fetch("/api/health")
        .then(res => res.json())
        .then(data => {
            const devBadge = document.getElementById("deviceBadge");
            if (devBadge && data.device) {
                devBadge.textContent = `DEVICE: ${data.device.toUpperCase()}`;
            }
        })
        .catch(err => {
            console.warn("API health check warning", err);
        });

    // File Browsing & Cumulative Upload
    browseBtn.addEventListener("click", () => fileInput.click());
    if (browseFolderBtn && folderInput) {
        browseFolderBtn.addEventListener("click", () => folderInput.click());
        folderInput.addEventListener("change", (e) => {
            addFilesCumulatively(Array.from(e.target.files));
            folderInput.value = "";
        });
    }

    fileInput.addEventListener("change", (e) => {
        addFilesCumulatively(Array.from(e.target.files));
        fileInput.value = "";
    });

    // Drag & Drop
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            addFilesCumulatively(Array.from(e.dataTransfer.files));
        }
    });

    // Clear All Files
    clearFilesBtn.addEventListener("click", () => {
        uploadedFiles = [];
        designatedAnchorIndex = 0;
        jobData = null;
        currentGraphMatch = null;
        resultsSection.style.display = "none";
        updateFileListUI();
    });

    function addFilesCumulatively(newFiles) {
        if (!newFiles || newFiles.length === 0) return;
        
        // If previous job was demo, reset view when uploading new custom files
        if (jobData && jobData.isDemo) {
            jobData = null;
            currentGraphMatch = null;
            resultsSection.style.display = "none";
        }

        newFiles.forEach(nf => {
            const exists = uploadedFiles.some(f => f.name === nf.name && f.size === nf.size);
            if (!exists) {
                uploadedFiles.push(nf);
            }
        });
        updateFileListUI();
    }

    function removeFile(index) {
        uploadedFiles.splice(index, 1);
        if (designatedAnchorIndex >= uploadedFiles.length) {
            designatedAnchorIndex = Math.max(0, uploadedFiles.length - 1);
        }
        updateFileListUI();
    }

    // Set designated reference anchor image
    function setAnchor(index) {
        designatedAnchorIndex = index;
        if (anchorSelect) anchorSelect.value = index;
        updateFileListUI();
    }

    if (anchorSelect) {
        anchorSelect.addEventListener("change", (e) => {
            designatedAnchorIndex = parseInt(e.target.value, 10) || 0;
            updateFileListUI();
        });
    }

    function updateFileListUI() {
        fileCount.textContent = uploadedFiles.length;
        fileTagsContainer.innerHTML = "";
        if (anchorSelect) anchorSelect.innerHTML = "";

        if (uploadedFiles.length === 0) {
            fileTagsContainer.innerHTML = '<div class="empty-state">No images loaded yet. Upload files or click "Run Demo Dataset".</div>';
            clearFilesBtn.style.display = "none";
            processBtn.disabled = true;
            if (anchorRow) anchorRow.style.display = "none";
            return;
        }

        clearFilesBtn.style.display = "inline-block";
        processBtn.disabled = uploadedFiles.length < 2;
        if (anchorRow) anchorRow.style.display = "flex";

        if (designatedAnchorIndex >= uploadedFiles.length) {
            designatedAnchorIndex = 0;
        }

        uploadedFiles.forEach((f, idx) => {
            const isAnchor = (idx === designatedAnchorIndex);
            const tag = document.createElement("div");
            tag.className = `file-tag-item ${isAnchor ? "is-anchor" : ""}`;
            const ext = f.name.split('.').pop().toLowerCase();
            let icon = "🖼️";
            if (ext === "xml") icon = "📜";
            else if (["raw", "dat", "img", "bin"].includes(ext)) icon = "🛰️";

            tag.innerHTML = `
                <div class="file-tag-info" title="${f.name}">
                    <span>${icon}</span>
                    <strong style="max-width: 170px; overflow: hidden; text-overflow: ellipsis;">${f.name}</strong>
                    <span style="color:var(--text-dim); font-size: 0.72rem;">(${(f.size/1024).toFixed(1)} KB)</span>
                </div>
                <div class="file-tag-actions">
                    <span class="file-tag-badge ${isAnchor ? 'badge-anchor-active' : 'badge-anchor-inactive'}" 
                          title="${isAnchor ? 'Designated Reference Image (Anchor)' : 'Click to designate as Reference Image'}">
                        ${isAnchor ? '★ REFERENCE' : 'SET AS REFERENCE'}
                    </span>
                    <button type="button" class="file-tag-remove" title="Remove image">×</button>
                </div>
            `;

            // Badge click sets as Reference
            const badge = tag.querySelector(".file-tag-badge");
            badge.addEventListener("click", () => setAnchor(idx));

            const removeBtn = tag.querySelector(".file-tag-remove");
            removeBtn.addEventListener("click", () => removeFile(idx));
            fileTagsContainer.appendChild(tag);

            // Populate anchor dropdown
            if (anchorSelect) {
                const opt = document.createElement("option");
                opt.value = idx;
                opt.textContent = `${idx + 1}: ${f.name} ${isAnchor ? "★ (Reference)" : ""}`;
                if (isAnchor) opt.selected = true;
                anchorSelect.appendChild(opt);
            }
        });
    }

    // Tab Switching
    tabGraphMatchBtn.addEventListener("click", () => {
        tabGraphMatchBtn.classList.add("active");
        tabDetectedFeaturesBtn.classList.remove("active");
        viewGraphMatch.style.display = "block";
        viewDetectedFeatures.style.display = "none";
    });

    tabDetectedFeaturesBtn.addEventListener("click", () => {
        tabDetectedFeaturesBtn.classList.add("active");
        tabGraphMatchBtn.classList.remove("active");
        viewGraphMatch.style.display = "none";
        viewDetectedFeatures.style.display = "block";
    });

    // Run Demo Pipeline — ONLY when user explicitly asks for it
    runDemoBtn.addEventListener("click", async () => {
        showLoading("Loading Demo Lunar Products...", "Running native parallel NLM, extracting features, and generating two-image graph match...", true);
        try {
            const res = await fetch("/api/pipeline/demo", { method: "POST" });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Demo failed");
            }
            const data = await res.json();
            data.isDemo = true;
            jobData = data;
            renderAllResults(data);
        } catch (e) {
            alert(`Error: ${e.message}`);
        } finally {
            hideLoading();
        }
    });

    // Run Processing for Uploaded Files
    processBtn.addEventListener("click", async () => {
        if (uploadedFiles.length < 2) {
            alert("Please upload at least 2 images to compute features and matches.");
            return;
        }

        const formData = new FormData();
        uploadedFiles.forEach(f => formData.append("files", f));
        formData.append("anchor_index", designatedAnchorIndex);
        if (uploadedFiles[designatedAnchorIndex]) {
            formData.append("anchor_filename", uploadedFiles[designatedAnchorIndex].name);
        }

        showLoading("Computing Features Across Images...", "Executing OpenCV parallel NLM, extracting crater morphology, and generating dual graph match...", true);
        try {
            const res = await fetch("/api/pipeline/process", {
                method: "POST",
                body: formData
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Processing failed");
            }
            const data = await res.json();
            data.isDemo = false;
            jobData = data;
            renderAllResults(data);
        } catch (e) {
            alert(`Pipeline Error: ${e.message}`);
        } finally {
            hideLoading();
        }
    });

    // Dynamic Matching when Changing Reference or Comparison Images
    async function triggerPairMatch() {
        if (isMatchingPair || !jobData || !jobData.single_images || jobData.single_images.length < 2) return;

        let idxA = parseInt(selectImageA.value, 10);
        let idxB = parseInt(selectImageB.value, 10);

        if (isNaN(idxA)) idxA = 0;
        if (isNaN(idxB)) idxB = 1;

        // If user accidentally selected the same image, automatically choose the alternate
        if (idxA === idxB) {
            const total = jobData.single_images.length;
            idxB = (idxA + 1) % total;
            selectImageB.value = idxB;
        }

        isMatchingPair = true;
        runPairMatchBtn.disabled = true;
        runPairMatchBtn.innerHTML = `<span>⏳</span> Comparing...`;
        graphMatchTitle.textContent = `Matching Image ${idxA + 1} ⇄ Image ${idxB + 1}...`;

        try {
            const res = await fetch("/api/pipeline/match_pair", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ index_a: idxA, index_b: idxB })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Pair match failed");
            }

            const matchData = await res.json();
            currentGraphMatch = matchData;
            renderGraphMatchView(matchData);
        } catch (e) {
            console.error("Pair Match Error:", e);
            alert(`Feature Comparison Error: ${e.message}`);
        } finally {
            isMatchingPair = false;
            runPairMatchBtn.disabled = false;
            runPairMatchBtn.innerHTML = `<span>⚡</span> Match Features & Generate Graph`;
        }
    }

    // Button trigger
    runPairMatchBtn.addEventListener("click", triggerPairMatch);

    // Automatic trigger on dropdown change
    selectImageA.addEventListener("change", triggerPairMatch);
    selectImageB.addEventListener("change", triggerPairMatch);

    // Swap Reference and Comparison images
    if (swapImagesBtn) {
        swapImagesBtn.addEventListener("click", () => {
            const valA = selectImageA.value;
            const valB = selectImageB.value;
            selectImageA.value = valB;
            selectImageB.value = valA;
            triggerPairMatch();
        });
    }

    // Loading overlay helpers
    function showLoading(title, sub, hideResults = true) {
        document.getElementById("loadingTitle").textContent = title;
        document.getElementById("loadingSubtitle").textContent = sub;
        loadingOverlay.style.display = "flex";
        if (hideResults) {
            resultsSection.style.display = "none";
        }
    }

    function hideLoading() {
        loadingOverlay.style.display = "none";
    }

    // Main Renderer
    function renderAllResults(data) {
        resultsSection.style.display = "block";
        const images = data.single_images || [];
        const count = images.length || data.processed_count || 0;
        document.getElementById("totalImagesPill").textContent = `${count} Images Processed`;
        document.getElementById("timePill").textContent = `Elapsed: ${data.total_execution_time_sec}s`;

        // Populate dropdowns for Image 1 and Image 2
        populateDropdowns(images);

        // Render Two-Image Graph Match View (Default to initial match)
        if (data.graph_match) {
            currentGraphMatch = data.graph_match;
            renderGraphMatchView(data.graph_match);
        }

        // Render Single Image Feature Inspection View
        if (images.length > 0) {
            renderSingleImageView(images[0]);
        }

        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    function populateDropdowns(images) {
        selectImageA.innerHTML = "";
        selectImageB.innerHTML = "";
        singleImageSelect.innerHTML = "";

        images.forEach((img, idx) => {
            const optA = document.createElement("option");
            optA.value = idx;
            optA.textContent = `${idx + 1}: ${img.name} (${img.total_craters} craters)`;
            if (idx === 0) optA.selected = true;
            selectImageA.appendChild(optA);

            const optB = document.createElement("option");
            optB.value = idx;
            optB.textContent = `${idx + 1}: ${img.name} (${img.total_craters} craters)`;
            if (idx === 1 || (idx === 0 && images.length === 1)) optB.selected = true;
            selectImageB.appendChild(optB);

            const optSingle = document.createElement("option");
            optSingle.value = idx;
            optSingle.textContent = `${idx + 1}: ${img.name} (${img.dimensions.width}×${img.dimensions.height})`;
            singleImageSelect.appendChild(optSingle);
        });

        singleImageSelect.onchange = (e) => {
            const selectedIdx = parseInt(e.target.value, 10);
            if (jobData && jobData.single_images && jobData.single_images[selectedIdx]) {
                renderSingleImageView(jobData.single_images[selectedIdx]);
            }
        };
    }

    // Render Dual Photo Graph Match (Requirement 4 & 5)
    function renderGraphMatchView(match) {
        graphMatchTitle.textContent = `${match.image_a_name}  ⇄  ${match.image_b_name}`;
        matchCountVal.textContent = match.total_matched_features;
        craterMatchCountVal.textContent = match.matched_craters_count;
        graphEdgeCountVal.textContent = match.edges ? match.edges.length : 0;
        const isMatched = match.is_matched !== false && match.total_matched_features > 0;
        if (!isMatched) {
            matchedCountBadge.textContent = "0 Matches (Different Surfaces / No Overlap)";
            matchedCountBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
            matchedCountBadge.style.color = "var(--accent-amber)";
        } else {
            matchedCountBadge.textContent = `${match.total_matched_features} Corresponding Points (1 to ${match.total_matched_features})`;
            matchedCountBadge.style.borderColor = "";
            matchedCountBadge.style.color = "";
        }

        // Update image based on active view toggle
        updateGraphImageSrc();

        // Setup Graph View Toggle Buttons
        document.querySelectorAll("[data-graph-view]").forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll("[data-graph-view]").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                activeGraphViewMode = btn.dataset.graphView;
                updateGraphImageSrc();
            };
        });

        // Populate Matched Features & Crater Similarity Probability Table (Requirement 3, 4, 5)
        matchedFeaturesTableBody.innerHTML = "";
        const nodes = match.nodes || [];

        if (nodes.length === 0) {
            matchedFeaturesTableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align:center; padding: 2.5rem; color: var(--text-muted);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🪐</div>
                        <strong style="color: var(--text-main); font-size: 1rem;">Different Lunar Surfaces / No Geometric Overlap Detected</strong><br>
                        <span style="color: var(--text-dim); font-size: 0.85rem; display: inline-block; margin-top: 0.4rem; max-width: 560px; line-height: 1.5;">
                            Geometric verification confirmed zero common surface features between Image 1 and Image 2. Independent surface landmarks are preserved without false correspondence mapping.
                        </span>
                    </td>
                </tr>
            `;
            return;
        }

        nodes.forEach(n => {
            const row = document.createElement("tr");
            row.id = `match-row-${n.label}`;

            // Similarity Probability Styling
            const probPct = n.similarity_percentage;
            let probColor = "var(--accent-emerald)";
            if (probPct < 60) probColor = "var(--accent-amber)";
            else if (probPct < 80) probColor = "var(--accent-cyan)";

            // Type badge
            const typeHtml = n.is_crater
                ? `<span class="badge" style="background:rgba(16,185,129,0.2); color:var(--accent-emerald); border:1px solid rgba(16,185,129,0.4);">Crater Basin</span>`
                : `<span class="badge" style="background:rgba(0,240,255,0.15); color:var(--accent-cyan); border:1px solid rgba(0,240,255,0.3);">Surface Point</span>`;

            // Image 1 info
            const aInfo = n.is_crater
                ? `<strong>${n.crater_a_id}</strong><br><span style="font-size:0.75rem; color:var(--text-muted);">(X:${n.pt_a[0]}, Y:${n.pt_a[1]}) · D=${n.diameter_a.toFixed(0)}px</span>`
                : `<span style="font-size:0.8rem; color:var(--text-main);">(X: ${n.pt_a[0]}, Y: ${n.pt_a[1]})</span>`;

            // Image 2 info
            const bInfo = n.is_crater
                ? `<strong>${n.crater_b_id}</strong><br><span style="font-size:0.75rem; color:var(--text-muted);">(X:${n.pt_b[0]}, Y:${n.pt_b[1]}) · D=${n.diameter_b.toFixed(0)}px</span>`
                : `<span style="font-size:0.8rem; color:var(--text-main);">(X: ${n.pt_b[0]}, Y: ${n.pt_b[1]})</span>`;

            row.innerHTML = `
                <td>
                    <div class="circle-label-badge ${n.is_crater ? 'badge-crater-num' : 'badge-keypoint-num'}">
                        ${n.label}
                    </div>
                </td>
                <td>${typeHtml}</td>
                <td>${aInfo}</td>
                <td>${bInfo}</td>
                <td>
                    <div class="prob-cell">
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: ${probPct}%; background: ${probColor};"></div>
                        </div>
                        <span class="prob-text" style="color: ${probColor}; font-weight: 700;">${probPct.toFixed(1)}%</span>
                    </div>
                </td>
                <td style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.4;">
                    ${n.similarity_description}
                </td>
            `;

            // Hover effect on row
            row.addEventListener("mouseenter", () => row.classList.add("row-highlight"));
            row.addEventListener("mouseleave", () => row.classList.remove("row-highlight"));

            matchedFeaturesTableBody.appendChild(row);
        });
    }

    function updateGraphImageSrc() {
        if (!currentGraphMatch) return;
        if (activeGraphViewMode === "side_by_side") {
            graphMatchImg.src = currentGraphMatch.side_by_side_graph_b64;
        } else if (activeGraphViewMode === "photo_a") {
            graphMatchImg.src = currentGraphMatch.photo_a_graph_b64;
        } else if (activeGraphViewMode === "photo_b") {
            graphMatchImg.src = currentGraphMatch.photo_b_graph_b64;
        }
    }

    // Render Single Image Detected Features View
    function renderSingleImageView(imgData) {
        singleImgTitle.textContent = `${imgData.name} — Feature Profile`;
        singleImgDim.textContent = `${imgData.dimensions.width} × ${imgData.dimensions.height}`;
        singleImgCraters.textContent = imgData.total_craters;
        singleImgFeatures.textContent = imgData.total_features;
        singleCraterBadge.textContent = `${imgData.total_craters} Impact Craters Detected`;

        // Annotated Preview
        singleAnnotatedImg.src = imgData.annotated_features_b64;

        // Populate Table
        singleCraterTableBody.innerHTML = "";
        const craters = imgData.craters || [];

        if (craters.length === 0) {
            singleCraterTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-dim);">No verified crater rims in this frame.</td></tr>`;
            return;
        }

        craters.forEach(c => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td style="color:var(--accent-emerald); font-weight:600;">${c.id}</td>
                <td>(${c.x.toFixed(1)}, ${c.y.toFixed(1)})</td>
                <td>${c.diameter.toFixed(1)} px</td>
                <td>${c.rim_contrast.toFixed(1)}</td>
                <td><span style="color:var(--accent-cyan); font-weight:600;">${(c.confidence * 100).toFixed(0)}%</span></td>
            `;
            singleCraterTableBody.appendChild(row);
        });
    }

    // =========================================================================
    // SYSTEM MODE SWITCHER (INFERENCE VS NEURAL NETWORK TRAINING)
    // =========================================================================

    let currentSystemMode = "inference";
    let customTrainingFiles = [];
    let trainingDatasetLoaded = false;
    let accumulatedTrainingHistory = [];

    function setSystemMode(mode) {
        currentSystemMode = mode;
        if (mode === "training") {
            btnModeTraining.classList.add("active");
            btnModeInference.classList.remove("active");
            inferenceModeSection.style.display = "none";
            trainingModeSection.style.display = "flex";

            // Refresh training status
            fetchTrainingStatus();

            // Auto-load demo multi-angle crater dataset if not already populated
            if (!trainingDatasetLoaded && customTrainingFiles.length === 0) {
                loadMultiAngleDemoDataset();
            }
        } else {
            btnModeInference.classList.add("active");
            btnModeTraining.classList.remove("active");
            trainingModeSection.style.display = "none";
            inferenceModeSection.style.display = "block";
        }
    }

    btnModeInference.addEventListener("click", () => setSystemMode("inference"));
    btnModeTraining.addEventListener("click", () => setSystemMode("training"));
    if (switchToInferenceBtn) {
        switchToInferenceBtn.addEventListener("click", () => {
            setSystemMode("inference");
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    function fetchTrainingStatus() {
        fetch("/api/training/status")
            .then(res => res.json())
            .then(data => {
                if (trainModelBadge) trainModelBadge.textContent = `MODEL: ${data.model_name.split(' ')[0].toUpperCase()}`;
                if (trainParamsBadge) trainParamsBadge.textContent = `${data.total_parameters.toLocaleString()} PARAMS`;
                if (trainStepsBadge) trainStepsBadge.textContent = `STEPS: ${data.total_steps_executed}`;
            })
            .catch(err => console.warn("Training status error:", err));
    }

    // Dynamic Math Formula Update
    function updateMathFormula() {
        const type = trainLossType.value;
        if (type === "multi_angle_cosine") {
            mathLossFormula.innerHTML = `&Lscr;(&theta;) = <sup>1</sup>&frasl;<sub>&binom{K}{2}</sub> &sum;<sub>i &lt; j</sub> (1 - cos(z<sub>&theta;i</sub>, z<sub>&theta;j</sub>))`;
        } else {
            mathLossFormula.innerHTML = `&Lscr;(&theta;) = ||z<sub>&theta;1</sub> - z<sub>&theta;2</sub>||<sup>2</sup> + 0.5 &middot; max(0, 1.0 - ||z<sub>&theta;1</sub> - z<sub>neg</sub>||)<sup>2</sup>`;
        }
    }
    trainLossType.addEventListener("change", updateMathFormula);

    // Logging to Training Console Log
    function logTrainingConsole(msg, type = "normal") {
        if (!trainingConsoleLog) return;
        const line = document.createElement("div");
        line.className = `console-line ${type}`;
        const timeStr = new Date().toTimeString().split(" ")[0];
        line.textContent = `[${timeStr}] ${msg}`;
        trainingConsoleLog.appendChild(line);
        trainingConsoleLog.scrollTop = trainingConsoleLog.scrollHeight;
    }

    // Load Demo Multi-Angle Crater Dataset
    function loadMultiAngleDemoDataset() {
        loadAngleDemoBtn.disabled = true;
        loadAngleDemoBtn.innerHTML = `<span class="btn-icon">⏳</span> Loading...`;
        logTrainingConsole("Requesting synthesized multi-angle crater dataset (varying incidence & azimuth)...", "highlight");

        fetch("/api/training/sample_data")
            .then(res => res.json())
            .then(data => {
                trainingDatasetLoaded = true;
                customTrainingFiles = [];
                renderAnglePreviews(data.angles, data.negative);
                logTrainingConsole(`Loaded 4 distinct solar angle views of crater basin + 1 negative terrain patch.`, "success");
            })
            .catch(err => {
                logTrainingConsole(`Error loading multi-angle crater dataset: ${err.message}`, "error");
            })
            .finally(() => {
                loadAngleDemoBtn.disabled = false;
                loadAngleDemoBtn.innerHTML = `<span class="btn-icon">⚡</span> Demo Set`;
            });
    }
    loadAngleDemoBtn.addEventListener("click", loadMultiAngleDemoDataset);

    // Load Real Crater Rotations directly from D:\moon
    const loadMoonDatasetBtn = document.getElementById("loadMoonDatasetBtn");
    if (loadMoonDatasetBtn) {
        loadMoonDatasetBtn.addEventListener("click", () => {
            loadMoonDatasetBtn.disabled = true;
            loadMoonDatasetBtn.innerHTML = `<span>⏳</span> Sampling D:\\moon...`;
            logTrainingConsole("Sampling authentic multi-rotation crater features directly from D:\\moon\\moon...", "highlight");

            fetch("/api/training/moon_data")
                .then(res => res.json())
                .then(data => {
                    trainingDatasetLoaded = true;
                    customTrainingFiles = [];
                    renderAnglePreviews(data.angles, data.negative);
                    logTrainingConsole(`Successfully sampled authentic crater rotations from ${data.dataset_source}!`, "success");
                })
                .catch(err => {
                    logTrainingConsole(`Error sampling D:\\moon: ${err.message}`, "error");
                })
                .finally(() => {
                    loadMoonDatasetBtn.disabled = false;
                    loadMoonDatasetBtn.innerHTML = `<span>🌙</span> Load from D:\\moon`;
                });
        });
    }

    // Render Angle Preview Cards
    function renderAnglePreviews(angles, negative) {
        anglePreviewsGrid.innerHTML = "";
        angles.forEach((item, idx) => {
            const card = document.createElement("div");
            card.className = "angle-preview-card";
            card.innerHTML = `
                <img src="${item.preview_b64}" class="angle-thumb" alt="${item.label}">
                <div class="angle-title">${item.label.split('(')[0]}</div>
                <div class="angle-meta">Inc: ${item.incidence_deg}° | Az: ${item.azimuth_deg}°</div>
            `;
            anglePreviewsGrid.appendChild(card);
        });

        if (negative) {
            const negCard = document.createElement("div");
            negCard.className = "angle-preview-card negative-sample";
            negCard.innerHTML = `
                <img src="${negative.preview_b64}" class="angle-thumb" alt="Negative">
                <div class="angle-title" style="color:var(--accent-red);">Negative Sample</div>
                <div class="angle-meta">Non-matching terrain</div>
            `;
            anglePreviewsGrid.appendChild(negCard);
        }
    }

    // Upload Multi-Angle Images
    browseTrainingBtn.addEventListener("click", () => trainingFileInput.click());
    trainingFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleUploadedAngleFiles(Array.from(e.target.files));
        }
    });

    trainingDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        trainingDropzone.classList.add("dragover");
    });
    trainingDropzone.addEventListener("dragleave", () => trainingDropzone.classList.remove("dragover"));
    trainingDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        trainingDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleUploadedAngleFiles(Array.from(e.dataTransfer.files));
        }
    });

    function handleUploadedAngleFiles(files) {
        customTrainingFiles = files;
        trainingDatasetLoaded = false;
        anglePreviewsGrid.innerHTML = "";
        logTrainingConsole(`Processing ${files.length} custom crater images for angle invariance training...`, "highlight");

        files.forEach((f, idx) => {
            const reader = new FileReader();
            reader.onload = (ev) => {
                const card = document.createElement("div");
                card.className = "angle-preview-card";
                card.innerHTML = `
                    <img src="${ev.target.result}" class="angle-thumb" alt="${f.name}">
                    <div class="angle-title">View #${idx + 1}</div>
                    <div class="angle-meta">${f.name.substring(0, 16)}</div>
                `;
                anglePreviewsGrid.appendChild(card);
            };
            reader.readAsDataURL(f);
        });
        logTrainingConsole(`Loaded ${files.length} user crater angle frames into training buffer.`, "success");
    }

    // Execute Neural Network Training
    function executeTraining(epochsOverride = null) {
        const epochs = epochsOverride || parseInt(trainEpochs.value) || 5;
        const lr = parseFloat(trainLr.value) || 0.001;
        const lossType = trainLossType.value;
        const optimizer = trainOptimizer.value;

        startTrainBtn.disabled = true;
        stepTrainBtn.disabled = true;
        trainProgressContainer.style.display = "block";
        trainProgressBar.style.width = "0%";
        trainProgressPct.textContent = "0%";
        trainProgressLabel.textContent = `Optimizing parameters (${epochs} epochs, lr=${lr})...`;

        logTrainingConsole(`Starting training run: ${epochs} epochs, loss=${lossType}, optimizer=${optimizer}, lr=${lr}...`, "highlight");

        // Animate fake progress steps while backend completes backpropagation
        let prog = 10;
        const progressInterval = setInterval(() => {
            if (prog < 90) {
                prog += 15;
                trainProgressBar.style.width = `${prog}%`;
                trainProgressPct.textContent = `${prog}%`;
            }
        }, 200);

        const formData = new FormData();
        formData.append("epochs", epochs);
        formData.append("learning_rate", lr);
        formData.append("loss_type", lossType);
        formData.append("optimizer", optimizer);

        if (customTrainingFiles.length >= 2) {
            customTrainingFiles.forEach(f => formData.append("files", f));
        } else {
            formData.append("dataset_source", "moon");
        }

        fetch("/api/training/train", {
            method: "POST",
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            return res.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            trainProgressBar.style.width = "100%";
            trainProgressPct.textContent = "100%";
            trainProgressLabel.textContent = "Optimization complete! Parameters updated.";

            setTimeout(() => {
                trainProgressContainer.style.display = "none";
            }, 1200);

            // Accumulate history for chart
            accumulatedTrainingHistory.push(...data.history);

            // Render telemetry
            renderTrainingTelemetry(data);
            fetchTrainingStatus();
        })
        .catch(err => {
            clearInterval(progressInterval);
            trainProgressContainer.style.display = "none";
            logTrainingConsole(`Training execution error: ${err.message}`, "error");
            alert(`Training error: ${err.message}`);
        })
        .finally(() => {
            startTrainBtn.disabled = false;
            stepTrainBtn.disabled = false;
        });
    }

    startTrainBtn.addEventListener("click", () => executeTraining());
    stepTrainBtn.addEventListener("click", () => executeTraining(1));

    // Reset Model Weights
    resetWeightsBtn.addEventListener("click", () => {
        if (!confirm("Are you sure you want to reset the neural network to its initial baseline weights?")) return;
        fetch("/api/training/reset", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                accumulatedTrainingHistory = [];
                trainingTelemetrySection.style.display = "none";
                fetchTrainingStatus();
                logTrainingConsole("Neural network parameters successfully reset to initial baseline weights.", "highlight");
            })
            .catch(err => console.error("Reset error:", err));
    });

    // Render Telemetry & Charts
    function renderTrainingTelemetry(data) {
        trainingTelemetrySection.style.display = "block";

        // KPI Cards
        kpiLoss.textContent = data.final_loss.toFixed(4);
        kpiLossDelta.textContent = `Δ -${data.loss_reduction_pct}% reduction`;

        const lastStep = data.history[data.history.length - 1];
        kpiGradNorm.textContent = lastStep.grad_norm.toFixed(4);
        kpiWeightDelta.textContent = lastStep.param_delta.toFixed(5);
        kpiSimilarity.textContent = `${data.final_similarity_pct.toFixed(1)}%`;
        kpiSimGain.textContent = `+${data.similarity_gain_pct.toFixed(1)}% Gain (was ${data.initial_similarity_pct.toFixed(1)}%)`;

        chartEpochCountPill.textContent = `${accumulatedTrainingHistory.length} Steps Logged`;

        // Canvas Chart
        drawLossChart(accumulatedTrainingHistory);

        // Visual Heatmaps
        renderVisualComparisons(data.visualizations);

        // Console Log Output
        logTrainingConsole(`Epochs completed: ${data.epochs_completed} | Execution: ${data.execution_time_sec}s`, "success");
        logTrainingConsole(`Calculated Loss: ${data.initial_loss.toFixed(4)} → ${data.final_loss.toFixed(4)} (-${data.loss_reduction_pct}%)`, "highlight");
        logTrainingConsole(`Backpropagation ||∇_θ L||: ${lastStep.grad_norm.toFixed(4)} | Parameter Update ||ΔW||: ${lastStep.param_delta.toFixed(6)}`, "normal");
        logTrainingConsole(`Multi-angle crater cosine similarity elevated: ${data.initial_similarity_pct}% → ${data.final_similarity_pct}% (+${data.similarity_gain_pct}%)`, "success");
        logTrainingConsole(`CraterInvarianceNet parameters updated and active in lunar pipeline.`, "highlight");
    }

    // High-DPI Smooth Loss Curve Chart (Vanilla HTML5 Canvas)
    function drawLossChart(history) {
        if (!lossChartCanvas || !history || history.length === 0) return;
        const ctx = lossChartCanvas.getContext("2d");
        const rect = lossChartCanvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        lossChartCanvas.width = rect.width * dpr;
        lossChartCanvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const padL = 50;
        const padR = 25;
        const padT = 20;
        const padB = 30;

        ctx.clearRect(0, 0, w, h);

        const losses = history.map(d => d.loss);
        const minL = Math.max(0, Math.min(...losses) * 0.85);
        const maxL = Math.max(...losses, minL + 0.1);

        // Background grid
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        const numGridLines = 4;
        for (let i = 0; i <= numGridLines; i++) {
            const y = padT + (h - padT - padB) * (i / numGridLines);
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(w - padR, y);
            ctx.stroke();

            // Label
            const val = maxL - (i / numGridLines) * (maxL - minL);
            ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
            ctx.font = "10px JetBrains Mono, monospace";
            ctx.textAlign = "right";
            ctx.fillText(val.toFixed(3), padL - 8, y + 3);
        }

        const plotW = w - padL - padR;
        const plotH = h - padT - padB;

        // Points coordinates
        const points = history.map((d, idx) => {
            const x = history.length === 1 ? padL + plotW / 2 : padL + (idx / (history.length - 1)) * plotW;
            const y = padT + (1.0 - (d.loss - minL) / (maxL - minL)) * plotH;
            return { x, y, loss: d.loss, step: idx + 1 };
        });

        // Area Fill Gradient
        const grad = ctx.createLinearGradient(0, padT, 0, h - padB);
        grad.addColorStop(0, "rgba(0, 240, 255, 0.35)");
        grad.addColorStop(1, "rgba(16, 185, 129, 0.02)");

        ctx.beginPath();
        ctx.moveTo(points[0].x, h - padB);
        points.forEach(pt => ctx.lineTo(pt.x, pt.y));
        ctx.lineTo(points[points.length - 1].x, h - padB);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Stroke line
        ctx.beginPath();
        points.forEach((pt, idx) => {
            if (idx === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        });
        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "rgba(0, 240, 255, 0.6)";
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Data point circles & tooltips
        points.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.strokeStyle = "#10b981";
            ctx.lineWidth = 2;
            ctx.stroke();
        });

        // Bottom Axis Labels
        ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Step 1", points[0].x, h - 8);
        if (points.length > 1) {
            ctx.fillText(`Step ${points.length}`, points[points.length - 1].x, h - 8);
        }
    }

    // Render Multi-Angle Feature Activation Heatmaps
    function renderVisualComparisons(visualizations) {
        if (!visualComparisonRow || !visualizations) return;
        visualComparisonRow.innerHTML = "";

        visualizations.forEach(v => {
            const card = document.createElement("div");
            card.className = "vis-card";
            card.innerHTML = `
                <div style="font-size:0.75rem; font-weight:600; color:var(--accent-cyan); margin-bottom:0.4rem;">
                    Angle #${v.angle_index} View
                </div>
                <div class="vis-images-dual">
                    <div>
                        <img src="${v.preview_b64}" class="vis-img" alt="Original Angle">
                        <div style="font-size:0.65rem; color:var(--text-dim); margin-top:2px;">Illumination</div>
                    </div>
                    <div>
                        <img src="${v.heatmap_b64}" class="vis-img" alt="Activation Heatmap">
                        <div style="font-size:0.65rem; color:var(--accent-emerald); margin-top:2px;">Neural Energy</div>
                    </div>
                </div>
                <div style="font-size:0.68rem; color:var(--text-muted); margin-top:0.3rem;">
                    Invariant Crater Embedding Active
                </div>
            `;
            visualComparisonRow.appendChild(card);
        });
    }
});
