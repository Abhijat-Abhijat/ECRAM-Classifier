/* app.js: UI + in-browser ONNX inference. No network calls except loading this site's own files. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const MAX_MB = 2048;  // the reader streams TIFFs, so only the browser's file limit matters
  let session = null, meta = null, selected = null, busy = false;
  const history = [];

  const ext = (n) => { const i = n.lastIndexOf("."); return i < 0 ? "" : n.slice(i).toLowerCase(); };
  const isTiffName = (n) => [".tif", ".tiff"].includes(ext(n));
  const fmtMB = (b) => (b / 1048576).toFixed(b < 10485760 ? 2 : 1) + " MB";
  const setStatus = (s, t) => { $("dot").className = "dot " + s; $("statusText").textContent = t; };
  const showError = (m) => { $("errorBox").textContent = m; $("errorBox").classList.add("show"); };
  const hideError = () => $("errorBox").classList.remove("show");

  // ---------------------------------------------------------------- model
  async function loadModel() {
    try {
      meta = await (await fetch("model/model.json", { cache: "no-cache" })).json();
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      session = await ort.InferenceSession.create(`model/${meta.onnx_file}`, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all",
      });
      // warm-up so the first real prediction is fast
      await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", new Float32Array(224 * 224 * 3), [1, 224, 224, 3]) });
      setStatus("ok", "Model ready (on device)");
      if (selected) $("analyzeBtn").disabled = false;
    } catch (e) {
      console.error(e);
      setStatus("bad", "Model failed to load");
      showError("Couldn't load the model. Check your connection and reload the page. (" + e.message + ")");
    }
  }

  async function predict(file) {
    const t0 = performance.now();
    const input = await CellImaging.toModelInput(file, (f) => setProgress(true, "Reading image…", Math.round(f * 100)));
    setProgress(true, "Analyzing…", null);
    await new Promise(requestAnimationFrame);
    const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input.data, [1, 224, 224, 3]) };
    const out = await session.run(feeds);
    const p = out[session.outputNames[0]].data[0];       // P(cancerous)
    const thr = meta.threshold;
    const isC = p >= thr;
    return {
      label: isC ? "cancerous" : "non_cancerous", is_cancerous: isC,
      prob_cancerous: Math.round(p * 1000) / 10,
      confidence: Math.round((isC ? p : 1 - p) * 1000) / 10,
      threshold: Math.round(thr * 1000) / 10,
      processing_ms: Math.round(performance.now() - t0),
      kind: input.kind, raw: p,
    };
  }

  // ---------------------------------------------------------------- UI
  function handleFile(file) {
    if (!file) return;
    hideError();
    if (file.size > MAX_MB * 1048576) return showError(`This file is ${fmtMB(file.size)}; the limit is ${MAX_MB} MB.`);
    selected = file;
    const img = $("preview");
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    if (isTiffName(file.name)) {
      img.hidden = true; $("slideMsg").hidden = false;
      $("slideMain").textContent = "TIF loaded";
      $("limitsHint").textContent = "Browsers can’t preview TIF, but it will still be analyzed.";
    } else {
      img.src = URL.createObjectURL(file); img.hidden = false; $("slideMsg").hidden = true;
    }
    $("fileInfo").textContent = `${file.name} · ${fmtMB(file.size)}`;
    $("analyzeBtn").disabled = !session;
    $("clearBtn").hidden = false;
  }

  function reset() {
    selected = null; $("fileInput").value = ""; $("cameraInput").value = "";
    const img = $("preview"); if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    img.hidden = true; img.removeAttribute("src");
    $("slideMsg").hidden = false;
    $("slideMain").innerHTML = 'Tap to choose an image<span class="drag-only"> or drag it here</span>';
    $("limitsHint").textContent = "JPG · PNG · TIF (including whole‑slide TIFFs)";
    $("fileInfo").textContent = ""; $("analyzeBtn").disabled = true; $("clearBtn").hidden = true;
    hideError();
  }

  function setProgress(show, text, pct) {
    $("progress").classList.toggle("show", show);
    if (!show) return;
    $("progressText").textContent = text;
    $("progressPct").textContent = pct == null ? "" : pct + "%";
    $("progressFill").classList.toggle("indeterminate", pct == null);
    $("progressFill").style.width = pct == null ? "" : pct + "%";
  }

  async function analyze() {
    if (!selected || busy || !session) return;
    busy = true; hideError();
    $("analyzeBtn").disabled = true; $("analyzeBtn").textContent = "Analyzing…";
    setProgress(true, "Reading image…", 0);
    try {
      const d = await predict(selected);
      render(d); addHistory(selected.name, d);
      if (matchMedia("(max-width: 879px)").matches) $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      console.error(e);
      showError(e instanceof CellImaging.UnsupportedImageError ? e.message
        : "Couldn't read that image. Please try a different file (" + (e.message || e) + ").");
    } finally {
      busy = false; setProgress(false);
      $("analyzeBtn").disabled = !selected; $("analyzeBtn").textContent = "Analyze slide";
    }
  }

  function render(d) {
    const cls = d.is_cancerous ? "cancerous" : "non_cancerous";
    $("placeholder").hidden = true;
    $("resultLabel").textContent = d.is_cancerous ? "Cancerous" : "Non‑cancerous";
    $("resultLabel").className = "result-label " + cls;
    $("resultBadge").textContent = d.is_cancerous ? "Flagged for review" : "Not flagged";
    $("resultBadge").className = "badge " + cls;
    $("marker").style.left = d.prob_cancerous + "%";
    $("thrLine").style.left = d.threshold + "%";
    $("thrLabel").textContent = `threshold ${d.threshold}%`;
    $("sProb").textContent = d.prob_cancerous + "%";
    $("sConf").textContent = d.confidence + "%";
    $("sTime").textContent = d.processing_ms >= 1000 ? (d.processing_ms / 1000).toFixed(1) + " s" : d.processing_ms + " ms";
    const unsure = d.confidence < 70 ? " The score is close to the threshold, so the model is uncertain about this slide." : "";
    $("explainer").textContent = (d.is_cancerous
      ? "Patterns in this image resemble the cancerous slides the model was trained on. Treat this as a prompt for expert review, not a diagnosis."
      : "Patterns in this image resemble the non‑cancerous slides the model was trained on. This is not a guarantee. When in doubt, get it checked.") + unsure;
    $("kind").textContent = d.kind;
    $("result").classList.add("show");
  }

  function addHistory(name, d) {
    history.unshift({ name, d });
    const ul = $("historyList"); ul.innerHTML = "";
    history.slice(0, 8).forEach(({ name, d }) => {
      const li = document.createElement("li");
      const a = document.createElement("span"); a.textContent = name; a.title = name;
      const b = document.createElement("span");
      b.textContent = `${d.is_cancerous ? "Cancerous" : "Non‑cancerous"} · ${d.prob_cancerous}%`;
      b.className = d.is_cancerous ? "cancerous" : "non_cancerous";
      li.append(a, b); ul.append(li);
    });
  }

  const dz = $("dropZone");
  dz.addEventListener("click", () => $("fileInput").click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("dragover"); handleFile(e.dataTransfer.files[0]); });
  $("fileInput").addEventListener("change", (e) => handleFile(e.target.files[0]));
  $("cameraInput").addEventListener("change", (e) => handleFile(e.target.files[0]));
  $("cameraBtn").addEventListener("click", () => $("cameraInput").click());
  $("clearBtn").addEventListener("click", reset);
  $("analyzeBtn").addEventListener("click", analyze);
  document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) handleFile(f); });

  window.CellApp = { predict, ready: loadModel() };   // exposed for the automated browser test
})();
