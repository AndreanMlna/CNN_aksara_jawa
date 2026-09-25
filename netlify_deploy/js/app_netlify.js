/**
 * AksaraAI Studio - Netlify Edition Engine
 * In-Browser Deep Learning Inference with ONNX Runtime WebAssembly (WASM/WebGL)
 * Standar: Staff/Principal Machine Learning Engineer & Computer Vision Scientist
 */

document.addEventListener('DOMContentLoaded', async () => {
  // =========================================================
  // State Global Aplikasi
  // =========================================================
  const state = {
    activeTab: 'tab-canvas',
    brushSize: 6,
    isDrawing: false,
    undoStack: [],
    maxUndo: 20,
    uploadedFile: null,
    uploadedImageElem: null,
    lastPoints: [],
    // Model & Runtime
    isModelLoaded: false,
    ortSession: null,
    classIndices: null,
    idxToClass: {},
    classToIdx: {},
    totalClasses: 120,
    modelPath: 'models/aksara_efficientnet_v2_web.onnx',
    classMapPath: 'models/class_indices.json',
    apiEndpoint: null // Default null (In-Browser ONNX WASM murni)
  };

  // =========================================================
  // DOM Elements Selector
  // =========================================================
  const elements = {
    // Header & Status
    engineBadge: document.getElementById('engine-status-badge'),
    engineText: document.getElementById('engine-status-text'),

    // Tabs
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Canvas
    canvas: document.getElementById('paint-canvas'),
    canvasPlaceholder: document.getElementById('canvas-placeholder'),
    brushSlider: document.getElementById('brush-size'),
    brushIndicator: document.getElementById('brush-size-val'),
    btnUndo: document.getElementById('btn-undo'),
    btnClear: document.getElementById('btn-clear'),
    btnPredictCanvas: document.getElementById('btn-predict-canvas'),
    sampleChips: document.querySelectorAll('.chip'),

    // Upload
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    btnBrowse: document.getElementById('btn-browse'),
    previewContainer: document.getElementById('preview-container'),
    imagePreview: document.getElementById('image-preview'),
    previewFilename: document.getElementById('preview-filename'),
    btnRemovePreview: document.getElementById('btn-remove-preview'),
    btnPredictUpload: document.getElementById('btn-predict-upload'),

    // Quiz & Practice Center
    btnSubnavWrite: document.getElementById('btn-subnav-write'),
    btnSubnavMcq: document.getElementById('btn-subnav-mcq'),
    quizPanelWrite: document.getElementById('quiz-panel-write'),
    quizPanelMcq: document.getElementById('quiz-panel-mcq'),
    quizTargetCategory: document.getElementById('quiz-target-category'),
    quizTargetChar: document.getElementById('quiz-target-char'),
    quizTargetLatin: document.getElementById('quiz-target-latin'),
    quizTargetDesc: document.getElementById('quiz-target-desc'),
    quizScore: document.getElementById('quiz-score'),
    quizStreak: document.getElementById('quiz-streak'),
    btnQuizShuffle: document.getElementById('btn-quiz-shuffle'),
    btnQuizVerify: document.getElementById('btn-quiz-verify'),
    quizFeedbackCard: document.getElementById('quiz-feedback-card'),
    quizFeedbackBadge: document.getElementById('quiz-feedback-badge'),
    quizFeedbackConf: document.getElementById('quiz-feedback-conf'),
    quizFeedbackText: document.getElementById('quiz-feedback-text'),
    btnQuizNext: document.getElementById('btn-quiz-next'),
    quizPaintCanvas: document.getElementById('quiz-paint-canvas'),
    quizGhostOverlay: document.getElementById('quiz-ghost-overlay'),
    ghostBadgeIndicator: document.getElementById('ghost-badge-indicator'),
    btnQuizGhost: document.getElementById('btn-quiz-ghost'),
    quizGhostBtnText: document.getElementById('quiz-ghost-btn-text'),
    btnQuizUndo: document.getElementById('btn-quiz-undo'),
    btnQuizClear: document.getElementById('btn-quiz-clear'),
    quizBrushSize: document.getElementById('quiz-brush-size'),
    quizBrushVal: document.getElementById('quiz-brush-val'),

    // MCQ Flashcard Quiz
    mcqScore: document.getElementById('mcq-score'),
    mcqStreak: document.getElementById('mcq-streak'),
    mcqTargetGlyph: document.getElementById('mcq-target-glyph'),
    mcqCategoryHint: document.getElementById('mcq-category-hint'),
    mcqOptionsGrid: document.getElementById('mcq-options-grid'),
    btnMcqShuffle: document.getElementById('btn-mcq-shuffle'),
    mcqFeedbackHint: document.getElementById('mcq-feedback-hint'),

    // Catalog
    catalogGrid: document.getElementById('catalog-grid'),
    catalogCategoryPills: document.querySelectorAll('.cat-pill'),
    catalogSearch: document.getElementById('catalog-search'),

    // Diagnostic Results Panel
    resultPlaceholder: document.getElementById('result-placeholder'),
    resultLoading: document.getElementById('result-loading'),
    resultContent: document.getElementById('result-content'),
    latencyChip: document.getElementById('latency-chip'),
    latencyVal: document.getElementById('latency-val'),

    resUnicode: document.getElementById('res-unicode'),
    resLatin: document.getElementById('res-latin'),
    resCategory: document.getElementById('res-category'),
    resConfidenceBadge: document.getElementById('res-confidence-badge'),
    resDesc: document.getElementById('res-desc'),
    resConfidencePct: document.getElementById('res-confidence-pct'),
    resProgressBar: document.getElementById('res-progress-bar'),
    resTop5List: document.getElementById('res-top5-list'),
    resTop5Title: document.getElementById('res-top5-title'),
    confMeterLabel: document.getElementById('conf-meter-label'),
    predictionHeroCard: document.getElementById('prediction-hero-card'),
    ambiguityCard: document.getElementById('ambiguity-card'),
    ambiguityBadge: document.getElementById('ambiguity-badge'),
    ambiguityDesc: document.getElementById('ambiguity-desc'),
    breakdownCard: document.querySelector('.breakdown-card'),
    resBaseConsonant: document.getElementById('res-base-consonant'),
    resSandhanganType: document.getElementById('res-sandhangan-type'),
    resVowel: document.getElementById('res-vowel'),
    resPosition: document.getElementById('res-position'),

    toastContainer: document.getElementById('toast-container')
  };

  const ctx = elements.canvas.getContext('2d', { willReadFrequently: true });

  // =========================================================
  // 1. Inisialisasi ONNX Runtime Web Engine
  // =========================================================
  async function initOnnxEngine() {
    try {
      if (elements.engineText) elements.engineText.textContent = 'Memuat Class Map...';
      
      // Muat Class Mapping
      const mapResp = await fetch(state.classMapPath);
      state.classIndices = await mapResp.json();
      state.idxToClass = {};
      state.classToIdx = {};

      for (const [cls, idx] of Object.entries(state.classIndices)) {
        state.idxToClass[idx] = cls;
        state.classToIdx[cls] = idx;
      }
      state.totalClasses = Object.keys(state.classIndices).length;

      // Konfigurasi ONNX Web
      if (elements.engineText) elements.engineText.textContent = 'Mengunduh Model (19.3MB)...';

      // Pastikan library ort tersedia secara global
      if (typeof ort === 'undefined') {
        throw new Error('Library onnxruntime-web (ort) belum dimuat.');
      }

      // Konfigurasi opsi eksekusi WebAssembly
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ort.env.wasm.simd = true;

      // Buat Inference Session
      state.ortSession = await ort.InferenceSession.create(state.modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });

      state.isModelLoaded = true;
      if (elements.engineBadge) {
        elements.engineBadge.classList.add('live');
        elements.engineBadge.title = 'In-Browser ONNX WebAssembly Siap & Aktif';
      }
      if (elements.engineText) {
        elements.engineText.textContent = 'ONNX WASM (Siap)';
      }
      showToast('🚀 Model ONNX WebAssembly siap! Inferensi berjalan langsung di browsermu.', 'success');
    } catch (err) {
      console.warn('Gagal memuat ONNX Runtime lokal:', err);
      if (elements.engineText) {
        elements.engineText.textContent = 'Mode API / Offline';
      }
      showToast(`Info Engine: ${err.message}. Membuka mode fallback.`, 'info');
    }
  }

  // =========================================================
  // 2. Inisialisasi Kanvas Gambar
  // =========================================================
  function initCanvas() {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#05070B';
    ctx.lineWidth = state.brushSize;

    saveCanvasState();

    elements.canvas.addEventListener('mousedown', startDrawing);
    elements.canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    elements.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      startDrawing(e.touches[0]);
    }, { passive: false });

    elements.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      draw(e.touches[0]);
    }, { passive: false });

    elements.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      stopDrawing();
    });

    elements.brushSlider.addEventListener('input', (e) => {
      state.brushSize = parseInt(e.target.value, 10);
      elements.brushIndicator.textContent = `${state.brushSize}px`;
      ctx.lineWidth = state.brushSize;
    });

    elements.btnUndo.addEventListener('click', undoCanvas);
    elements.btnClear.addEventListener('click', clearCanvas);
  }

  function getCanvasCoords(e) {
    const rect = elements.canvas.getBoundingClientRect();
    const scaleX = elements.canvas.width / rect.width;
    const scaleY = elements.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  function startDrawing(e) {
    state.isDrawing = true;
    elements.canvasPlaceholder.classList.add('hidden');
    const { x, y } = getCanvasCoords(e);
    state.lastPoints = [{ x, y }];

    ctx.beginPath();
    ctx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#05070B';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function draw(e) {
    if (!state.isDrawing) return;
    const { x, y } = getCanvasCoords(e);
    state.lastPoints.push({ x, y });

    if (state.lastPoints.length > 2) {
      const p1 = state.lastPoints[state.lastPoints.length - 2];
      const p2 = state.lastPoints[state.lastPoints.length - 1];
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2;

      ctx.quadraticCurveTo(p1.x, p1.y, midX, midY);
      ctx.stroke();
    }
  }

  function stopDrawing() {
    if (!state.isDrawing) return;
    state.isDrawing = false;
    ctx.closePath();
    saveCanvasState();
  }

  function saveCanvasState() {
    if (state.undoStack.length >= state.maxUndo) {
      state.undoStack.shift();
    }
    state.undoStack.push(ctx.getImageData(0, 0, elements.canvas.width, elements.canvas.height));
  }

  function undoCanvas() {
    if (state.undoStack.length > 1) {
      state.undoStack.pop();
      const prevState = state.undoStack[state.undoStack.length - 1];
      ctx.putImageData(prevState, 0, 0);
    } else if (state.undoStack.length === 1) {
      clearCanvas();
    }
  }

  function clearCanvas() {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    elements.canvasPlaceholder.classList.remove('hidden');
    state.undoStack = [];
    saveCanvasState();
  }

  // =========================================================
  // 3. Computer Vision Preprocessing & Tensor Transformation
  // =========================================================
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];

  function getInkBoundingBox(c) {
    const w = c.width;
    const h = c.height;
    const imgData = c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h);
    const data = imgData.data;

    let minX = w, maxX = -1, minY = h, maxY = -1;
    let inkPoints = [];

    // Perimeter guard: skip 4 border pixels
    for (let y = 4; y < h - 4; y++) {
      for (let x = 4; x < w - 4; x++) {
        const idx = (y * w + x) * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const brightness = 0.299 * r + 0.587 * g + 0.114 * b;

        if (brightness < 190) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          inkPoints.push({ x, y });
        }
      }
    }

    if (inkPoints.length < 25) return null;
    return { minX, maxX, minY, maxY, sw: maxX - minX + 1, sh: maxY - minY + 1, inkPoints };
  }

  function detectSukuDescenderJS(bbox) {
    if (!bbox) return { hasSuku: false, ratio: 0.0 };
    const { minX, minY, maxY, sw, sh, inkPoints } = bbox;
    const xCutoff = minX + 0.55 * sw;

    let bodyMaxY = -1, tailMaxY = -1, bodyMinY = 9999;

    for (const pt of inkPoints) {
      if (pt.x <= xCutoff) {
        if (pt.y > bodyMaxY) bodyMaxY = pt.y;
        if (pt.y < bodyMinY) bodyMinY = pt.y;
      } else {
        if (pt.y > tailMaxY) tailMaxY = pt.y;
      }
    }

    if (bodyMaxY === -1 || tailMaxY === -1) return { hasSuku: false, ratio: 0.0 };

    const bodyH = Math.max(1, bodyMaxY - bodyMinY + 1);
    const descenderPx = tailMaxY - bodyMaxY;
    const ratio = descenderPx / bodyH;

    const hasSuku = (descenderPx >= 20) && (ratio >= 0.25);
    return { hasSuku, ratio: Math.max(0, ratio) };
  }

  function detectTalingAndTarungJS(bbox) {
    if (!bbox) return { hasTaling: false, hasTarung: false };
    const { minX, minY, maxY, sw, sh, inkPoints } = bbox;
    const aspect = sw / Math.max(sh, 1);
    if (aspect < 0.60) return { hasTaling: false, hasTarung: false };

    const leftBoundary = minX + 0.48 * sw;
    let leftCount = 0, rightCount = 0;
    let leftMinY = 9999, leftMaxY = -1;

    for (const pt of inkPoints) {
      if (pt.x <= leftBoundary) {
        leftCount++;
        if (pt.y < leftMinY) leftMinY = pt.y;
        if (pt.y > leftMaxY) leftMaxY = pt.y;
      } else {
        rightCount++;
      }
    }

    if (leftCount < 15 || rightCount < 15) return { hasTaling: false, hasTarung: false };

    const leftH = leftMaxY - leftMinY + 1;
    const hasTaling = (leftH >= 0.60 * sh);

    let hasTarung = false;
    if (aspect >= 1.25) {
      const valleyStart = minX + 0.55 * sw;
      const valleyEnd = minX + 0.88 * sw;
      let farRightCount = 0;
      let valleyColumnCounts = {};

      for (const pt of inkPoints) {
        if (pt.x >= valleyStart && pt.x <= valleyEnd) {
          const col = Math.round(pt.x);
          valleyColumnCounts[col] = (valleyColumnCounts[col] || 0) + 1;
        }
        if (pt.x >= minX + 0.85 * sw) {
          farRightCount++;
        }
      }

      const minColVal = Object.keys(valleyColumnCounts).length > 0
        ? Math.min(...Object.values(valleyColumnCounts))
        : 0;

      if (minColVal <= 3 && farRightCount >= 25) {
        hasTarung = true;
      }
    }

    return { hasTaling, hasTarung };
  }

  function renderStrokeToCard(srcCanvas, bbox) {
    const { minX, minY, sw, sh } = bbox;
    const strokeCanvas = document.createElement('canvas');
    strokeCanvas.width = sw;
    strokeCanvas.height = sh;
    strokeCanvas.getContext('2d').drawImage(srcCanvas, minX, minY, sw, sh, 0, 0, sw, sh);

    // View 0: Square 500x500
    const v0Canvas = document.createElement('canvas');
    v0Canvas.width = 500;
    v0Canvas.height = 500;
    const v0Ctx = v0Canvas.getContext('2d');
    v0Ctx.fillStyle = '#FFFFFF';
    v0Ctx.fillRect(0, 0, 500, 500);
    const scale0 = Math.min(360 / Math.max(sw, 1), 360 / Math.max(sh, 1));
    const nw0 = Math.max(1, Math.round(sw * scale0));
    const nh0 = Math.max(1, Math.round(sh * scale0));
    v0Ctx.drawImage(strokeCanvas, 0, 0, sw, sh, Math.floor((500 - nw0) / 2), Math.floor((500 - nh0) / 2), nw0, nh0);

    // Card 600x500
    const cardCanvas = document.createElement('canvas');
    cardCanvas.width = 600;
    cardCanvas.height = 500;
    const cardCtx = cardCanvas.getContext('2d');
    cardCtx.fillStyle = '#FFFFFF';
    cardCtx.fillRect(0, 0, 600, 500);
    const scaleC = Math.min(360 / Math.max(sw, 1), 340 / Math.max(sh, 1));
    const nwc = Math.max(1, Math.round(sw * scaleC));
    const nhc = Math.max(1, Math.round(sh * scaleC));
    const px = Math.max(15, Math.min(600 - nwc - 15, Math.floor(245 - nwc / 2)));
    const py = Math.max(15, Math.min(500 - nhc - 10, 430 - nhc));
    cardCtx.drawImage(strokeCanvas, 0, 0, sw, sh, px, py, nwc, nhc);

    // View 1: 1528x540 Ultra-wide
    const v1Canvas = document.createElement('canvas');
    v1Canvas.width = 1528;
    v1Canvas.height = 540;
    const v1Ctx = v1Canvas.getContext('2d');
    v1Ctx.fillStyle = '#000000';
    v1Ctx.fillRect(0, 0, 1528, 540);
    v1Ctx.drawImage(cardCanvas, 464, 20);

    // View 2: 882x540 Medium-wide
    const v2Canvas = document.createElement('canvas');
    v2Canvas.width = 882;
    v2Canvas.height = 540;
    const v2Ctx = v2Canvas.getContext('2d');
    v2Ctx.fillStyle = '#000000';
    v2Ctx.fillRect(0, 0, 882, 540);
    v2Ctx.drawImage(cardCanvas, 140, 20);

    // View 3: 683x540 Native Taling
    const v3Canvas = document.createElement('canvas');
    v3Canvas.width = 683;
    v3Canvas.height = 540;
    const v3Ctx = v3Canvas.getContext('2d');
    v3Ctx.fillStyle = '#000000';
    v3Ctx.fillRect(0, 0, 683, 540);
    v3Ctx.drawImage(cardCanvas, 41, 20);

    return { v0: v0Canvas, v1: v1Canvas, v2: v2Canvas, v3: v3Canvas };
  }

  function canvasToFloat32Tensor(c) {
    const resized = document.createElement('canvas');
    resized.width = 224;
    resized.height = 224;
    const rCtx = resized.getContext('2d');
    rCtx.drawImage(c, 0, 0, c.width, c.height, 0, 0, 224, 224);

    const imgData = rCtx.getImageData(0, 0, 224, 224).data;
    const floatArr = new Float32Array(3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
      const r = imgData[i * 4] / 255.0;
      const g = imgData[i * 4 + 1] / 255.0;
      const b = imgData[i * 4 + 2] / 255.0;

      // Planar RGB ordering [3, 224, 224]
      floatArr[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      floatArr[224 * 224 + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      floatArr[2 * 224 * 224 + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    return new ort.Tensor('float32', floatArr, [1, 3, 224, 224]);
  }

  function softmax(logits) {
    const maxVal = Math.max(...logits);
    const exps = logits.map(z => Math.exp(z - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
  }

  // =========================================================
  // 4. In-Browser Multi-View Consensus Inference
  // =========================================================
  async function runClientInference(sourceCanvas, isUploaded = false) {
    if (!state.ortSession) {
      throw new Error('Model ONNX belum siap. Silakan tunggu beberapa detik.');
    }

    const startTime = performance.now();
    let finalProbs = new Array(state.totalClasses).fill(0.0);

    if (isUploaded) {
      // Direct single-pass inference untuk citra yang diunggah
      const tensor = canvasToFloat32Tensor(sourceCanvas);
      const results = await state.ortSession.run({ input: tensor });
      const rawLogits = Array.from(results.output.data);
      finalProbs = softmax(rawLogits);
    } else {
      // Kanvas Tulis: Domain-Aware Consensus Multi-View
      const bbox = getInkBoundingBox(sourceCanvas);
      if (!bbox) {
        throw new Error('Kanvas kosong. Silakan tulis karakter aksara terlebih dahulu.');
      }

      const { hasSuku } = detectSukuDescenderJS(bbox);
      const { hasTaling, hasTarung } = detectTalingAndTarungJS(bbox);
      const views = renderStrokeToCard(sourceCanvas, bbox);

      // Jalankan inferensi pada view
      const t0 = canvasToFloat32Tensor(views.v0);
      const t1 = canvasToFloat32Tensor(views.v1);
      const t2 = canvasToFloat32Tensor(views.v2);
      const t3 = canvasToFloat32Tensor(views.v3);

      const [res0, res1, res2, res3] = await Promise.all([
        state.ortSession.run({ input: t0 }),
        state.ortSession.run({ input: t1 }),
        state.ortSession.run({ input: t2 }),
        state.ortSession.run({ input: t3 })
      ]);

      const p0 = softmax(Array.from(res0.output.data));
      const p1 = softmax(Array.from(res1.output.data));
      const p2 = softmax(Array.from(res2.output.data));
      const p3 = softmax(Array.from(res3.output.data));

      if (hasSuku) {
        const wideSukuKeys = ['su', 'du', 'pu', 'bu', 'dhu', 'ju'];
        let wideSukuScore = 0;
        for (const k of wideSukuKeys) {
          const idx = state.classToIdx[`suku_${k}`];
          if (idx !== undefined) wideSukuScore += p1[idx];
        }

        for (let i = 0; i < state.totalClasses; i++) {
          const cls = state.idxToClass[i] || '';
          if (cls.startsWith('suku_')) {
            finalProbs[i] = (wideSukuScore >= 0.25)
              ? (0.80 * p1[i] + 0.20 * p2[i])
              : (0.40 * p1[i] + 0.60 * p2[i]);
          } else if (cls.startsWith('wulu_') || cls.startsWith('taling-tarung_') || cls.startsWith('taling_')) {
            finalProbs[i] = 0.5 * p2[i];
          } else {
            finalProbs[i] = 0.0; // Supresi aksara-dasar & pepet
          }
        }
      } else if (hasTaling && !hasTarung) {
        // Taling 2-glif murni (contoh: Gé ꦺꦒ)
        for (let i = 0; i < state.totalClasses; i++) {
          const cls = state.idxToClass[i] || '';
          if (cls.startsWith('taling_')) {
            finalProbs[i] = 0.55 * p3[i] + 0.45 * p2[i];
          } else if (cls.startsWith('taling-tarung_')) {
            finalProbs[i] = 0.05 * p1[i]; // Supresi false positive Taling-Tarung
          } else if (cls.startsWith('aksara-dasar_') || cls.startsWith('pepet_')) {
            finalProbs[i] = 0.0;
          } else {
            finalProbs[i] = 0.35 * p2[i];
          }
        }
      } else if (hasTaling && hasTarung) {
        // Taling-Tarung 3-glif (contoh: Go ꦺꦒꦴ)
        for (let i = 0; i < state.totalClasses; i++) {
          const cls = state.idxToClass[i] || '';
          if (cls.startsWith('taling-tarung_')) {
            finalProbs[i] = 0.70 * p1[i] + 0.30 * p2[i];
          } else if (cls.startsWith('taling_')) {
            finalProbs[i] = 0.10 * p3[i];
          } else {
            finalProbs[i] = 0.20 * p2[i];
          }
        }
      } else {
        // Aksara-dasar / Pepet / Wulu
        for (let i = 0; i < state.totalClasses; i++) {
          const cls = state.idxToClass[i] || '';
          if (cls.startsWith('aksara-dasar_') || cls.startsWith('pepet_')) {
            finalProbs[i] = 0.60 * p0[i] + 0.40 * p2[i];
          } else if (cls.startsWith('wulu_')) {
            finalProbs[i] = 0.30 * p0[i] + 0.70 * p2[i];
          } else if (cls.startsWith('suku_')) {
            finalProbs[i] = 0.15 * p2[i];
          } else {
            finalProbs[i] = 0.20 * p2[i];
          }
        }
      }

      // Normalisasi probabilitas agar sum = 1
      const pSum = finalProbs.reduce((a, b) => a + b, 0);
      if (pSum > 0) {
        finalProbs = finalProbs.map(v => v / pSum);
      }
    }

    const latencyMs = Math.round(performance.now() - startTime);

    // Format Top-5
    const candidates = finalProbs.map((prob, idx) => ({
      class_name: state.idxToClass[idx],
      confidence: parseFloat((prob * 100).toFixed(2))
    }));

    candidates.sort((a, b) => b.confidence - a.confidence);
    const top5 = candidates.slice(0, 5).map(c => {
      const meta = window.AKSARA_DATABASE ? window.AKSARA_DATABASE[c.class_name] : null;
      return {
        ...c,
        latin: meta ? meta.latin : c.class_name,
        unicode_char: meta ? meta.unicode_char : 'ꦄ',
        category: meta ? meta.category : 'Aksara',
        category_name: meta ? meta.category_name : 'Aksara Jawa',
        desc: meta ? meta.desc : '',
        base_consonant: meta ? meta.base_consonant : '',
        vowel: meta ? meta.vowel : '',
        position: meta ? meta.position : ''
      };
    });

    const predicted = top5[0];
    const isConfident = predicted.confidence >= 50.0;

    return {
      status: 'success',
      predicted,
      confidence: predicted.confidence,
      is_confident: isConfident,
      threshold: 50.0,
      latency_ms: latencyMs,
      top5,
      clarification_message: isConfident ? null :
        `Goresan dinilai ambigu atau kurang jelas sehingga belum mencapai ambang batas keyakinan 50.0%. AI menemukan 5 kemungkinan terdekat berikut. Silakan klik aksara yang kamu maksud:`
    };
  }

  // =========================================================
  // 5. Prediction Triggers & UI Rendering
  // =========================================================
  elements.btnPredictCanvas.addEventListener('click', async () => {
    setDiagnosticLoading(true);
    try {
      const result = await runClientInference(elements.canvas, false);
      setDiagnosticLoading(false);
      renderDiagnosticResult(result);

      if (result.is_confident) {
        showToast(`Klasifikasi berhasil: Aksara ${result.predicted.latin} (${result.predicted.confidence}%)`, 'success');
      } else {
        showToast(`⚠️ Tulisan kurang jelas (${result.predicted.confidence}%). Periksa opsi klarifikasi di bawah.`, 'info');
      }
    } catch (err) {
      setDiagnosticLoading(false);
      showToast(`Gagal: ${err.message}`, 'error');
    }
  });

  elements.btnPredictUpload.addEventListener('click', async () => {
    if (!state.uploadedImageElem) {
      showToast('Silakan pilih berkas gambar terlebih dahulu!', 'error');
      return;
    }

    setDiagnosticLoading(true);
    try {
      const uploadCanvas = document.createElement('canvas');
      uploadCanvas.width = state.uploadedImageElem.naturalWidth || 224;
      uploadCanvas.height = state.uploadedImageElem.naturalHeight || 224;
      uploadCanvas.getContext('2d').drawImage(state.uploadedImageElem, 0, 0);

      const result = await runClientInference(uploadCanvas, true);
      setDiagnosticLoading(false);
      renderDiagnosticResult(result);

      showToast(`Hasil Analisis: Aksara ${result.predicted.latin} (${result.predicted.confidence}%)`, 'success');
    } catch (err) {
      setDiagnosticLoading(false);
      showToast(`Gagal memproses gambar: ${err.message}`, 'error');
    }
  });

  function setDiagnosticLoading(isLoading) {
    if (isLoading) {
      elements.resultPlaceholder.classList.add('hidden');
      elements.resultContent.classList.add('hidden');
      elements.resultLoading.classList.remove('hidden');
      elements.latencyVal.textContent = 'Memproses...';
    } else {
      elements.resultLoading.classList.add('hidden');
    }
  }

  function renderDiagnosticResult(data) {
    const pred = data.predicted;
    const CONFIDENCE_THRESHOLD = data.threshold || 50.0;
    const isConfident = data.is_confident !== undefined ? data.is_confident : (pred.confidence >= CONFIDENCE_THRESHOLD);

    elements.resultPlaceholder.classList.add('hidden');
    elements.resultLoading.classList.add('hidden');
    elements.resultContent.classList.remove('hidden');

    if (data.latency_ms !== undefined) {
      elements.latencyVal.textContent = `${data.latency_ms} ms (WASM)`;
    }

    if (isConfident) {
      if (elements.predictionHeroCard) elements.predictionHeroCard.classList.remove('hidden');
      if (elements.ambiguityCard) elements.ambiguityCard.classList.add('hidden');
      if (elements.breakdownCard) elements.breakdownCard.classList.remove('hidden');

      if (elements.confMeterLabel) elements.confMeterLabel.textContent = 'Tingkat Keyakinan Tertinggi';
      if (elements.resTop5Title) elements.resTop5Title.textContent = '5 Kemungkinan Teratas (Top-5 Predictions)';

      elements.resUnicode.textContent = pred.unicode_char || 'ꦄ';
      elements.resLatin.textContent = pred.latin ? `Aksara ${pred.latin}` : pred.class_name;
      elements.resCategory.textContent = pred.category_name || pred.category;
      elements.resConfidenceBadge.textContent = `${pred.confidence}% Pasti`;
      elements.resDesc.textContent = pred.desc || 'Aksara Jawa';

      elements.resBaseConsonant.textContent = `${pred.base_consonant || 'Baku'}`;
      elements.resSandhanganType.textContent = pred.category_name || '-';
      elements.resVowel.textContent = pred.vowel ? `${pred.vowel} (/${pred.vowel}/)` : 'a (/a/)';
      elements.resPosition.textContent = pred.position || 'Bentuk Baku';
    } else {
      if (elements.predictionHeroCard) elements.predictionHeroCard.classList.add('hidden');
      if (elements.ambiguityCard) elements.ambiguityCard.classList.remove('hidden');
      if (elements.breakdownCard) elements.breakdownCard.classList.add('hidden');

      if (elements.ambiguityBadge) elements.ambiguityBadge.textContent = `Tulisan Kurang Jelas (${pred.confidence}% < ${CONFIDENCE_THRESHOLD}%)`;
      if (elements.ambiguityDesc) {
        elements.ambiguityDesc.textContent = data.clarification_message ||
          `Goresan dinilai ambigu atau kurang jelas sehingga belum mencapai ambang keyakinan ${CONFIDENCE_THRESHOLD}%. Silakan klik aksara yang Anda maksud:`;
      }
      if (elements.confMeterLabel) elements.confMeterLabel.textContent = `Keyakinan Teratas (${pred.confidence}%)`;
      if (elements.resTop5Title) elements.resTop5Title.textContent = '🔍 Apakah yang Kamu Maksud Salah Satu Aksara Ini? (Klik untuk Memilih)';
    }

    elements.resConfidencePct.textContent = `${pred.confidence}%`;
    elements.resProgressBar.style.width = `${Math.min(100, pred.confidence)}%`;

    if (pred.confidence >= 80) {
      elements.resProgressBar.style.background = 'linear-gradient(90deg, #10B981, #34D399)';
    } else if (pred.confidence >= 50) {
      elements.resProgressBar.style.background = 'linear-gradient(90deg, #F59E0B, #FBBF24)';
    } else {
      elements.resProgressBar.style.background = 'linear-gradient(90deg, #EF4444, #F87171)';
    }

    elements.resTop5List.innerHTML = '';
    if (data.top5 && data.top5.length > 0) {
      data.top5.forEach((item, index) => {
        const row = document.createElement('div');
        row.className = 'dist-row' + (isConfident ? '' : ' clickable');
        if (!isConfident) {
          row.title = `Klik untuk mengonfirmasi bahwa Anda bermaksud menulis Aksara ${item.latin || item.class_name}`;
        }

        row.innerHTML = `
          <div class="dist-glyph">${item.unicode_char || 'ꦄ'}</div>
          <div class="dist-info">
            <span class="dist-name">#${index + 1} ${item.latin || item.class_name} (${item.category_name || item.category})</span>
            <div class="dist-bar-track">
              <div class="dist-bar-fill" style="width: ${item.confidence}%;"></div>
            </div>
            ${!isConfident ? '<span class="dist-action-hint">👆 Klik untuk konfirmasi aksara ini</span>' : ''}
          </div>
          <span class="dist-pct">${item.confidence}%</span>
        `;

        if (!isConfident) {
          row.addEventListener('click', () => {
            selectAmbiguousCandidate(item);
          });
        }

        elements.resTop5List.appendChild(row);
      });
    }
  }

  function selectAmbiguousCandidate(item) {
    if (elements.predictionHeroCard) elements.predictionHeroCard.classList.remove('hidden');
    if (elements.ambiguityCard) elements.ambiguityCard.classList.add('hidden');
    if (elements.breakdownCard) elements.breakdownCard.classList.remove('hidden');

    elements.resUnicode.textContent = item.unicode_char || 'ꦄ';
    elements.resLatin.textContent = item.latin ? `Aksara ${item.latin}` : item.class_name;
    elements.resCategory.textContent = item.category_name || item.category;
    elements.resConfidenceBadge.textContent = 'Dikonfirmasi Pengguna';
    elements.resConfidenceBadge.style.background = 'rgba(16, 185, 129, 0.2)';
    elements.resConfidenceBadge.style.color = '#34D399';
    elements.resConfidenceBadge.style.border = '1px solid rgba(16, 185, 129, 0.4)';
    elements.resDesc.textContent = item.desc || `Aksara ${item.latin || item.class_name} yang Anda pilih dari kemungkinan teratas.`;

    if (elements.resBaseConsonant) elements.resBaseConsonant.textContent = item.base_consonant || 'Baku';
    if (elements.resSandhanganType) elements.resSandhanganType.textContent = item.category_name || item.category || '-';
    if (elements.resVowel) elements.resVowel.textContent = item.vowel ? `${item.vowel} (/${item.vowel}/)` : 'a (/a/)';
    if (elements.resPosition) elements.resPosition.textContent = item.position || 'Bentuk Baku';

    showToast(`✨ Anda memilih Aksara ${item.latin || item.class_name}!`, 'success');
  }

  // =========================================================
  // 6. Mode Kuis & Latihan Interaktif
  // =========================================================
  const quizState = {
    submode: 'write',
    isDrawing: false,
    lastPoints: [],
    brushSize: 6,
    isGhostOn: true,
    score: 0,
    streak: 0,
    target: null,
    mcqScore: 0,
    mcqStreak: 0,
    mcqTarget: null,
    mcqAnswered: false
  };

  let quizCtx = null;
  if (elements.quizPaintCanvas) {
    quizCtx = elements.quizPaintCanvas.getContext('2d', { willReadFrequently: true });
  }

  function initQuiz() {
    initQuizCanvas();
    shuffleQuizQuestion();
    initMcqQuiz();
    setupQuizSubnav();
  }

  function setupQuizSubnav() {
    if (elements.btnSubnavWrite && elements.btnSubnavMcq) {
      elements.btnSubnavWrite.addEventListener('click', () => switchQuizSubmode('write'));
      elements.btnSubnavMcq.addEventListener('click', () => switchQuizSubmode('mcq'));
    }
  }

  function switchQuizSubmode(mode) {
    quizState.submode = mode;
    const isWrite = mode === 'write';
    if (elements.btnSubnavWrite) elements.btnSubnavWrite.classList.toggle('active', isWrite);
    if (elements.btnSubnavMcq) elements.btnSubnavMcq.classList.toggle('active', !isWrite);
    if (elements.quizPanelWrite) elements.quizPanelWrite.classList.toggle('hidden', !isWrite);
    if (elements.quizPanelMcq) elements.quizPanelMcq.classList.toggle('hidden', isWrite);

    if (!isWrite && !quizState.mcqTarget) {
      shuffleMcqQuestion();
    }
  }

  function initQuizCanvas() {
    if (!elements.quizPaintCanvas || !quizCtx) return;
    quizCtx.fillStyle = '#FFFFFF';
    quizCtx.fillRect(0, 0, elements.quizPaintCanvas.width, elements.quizPaintCanvas.height);
    quizCtx.lineCap = 'round';
    quizCtx.lineJoin = 'round';
    quizCtx.strokeStyle = '#05070B';
    quizCtx.lineWidth = quizState.brushSize;

    elements.quizPaintCanvas.addEventListener('mousedown', startQuizDrawing);
    elements.quizPaintCanvas.addEventListener('mousemove', drawQuiz);
    window.addEventListener('mouseup', stopQuizDrawing);

    elements.quizPaintCanvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      startQuizDrawing(e.touches[0]);
    }, { passive: false });

    elements.quizPaintCanvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      drawQuiz(e.touches[0]);
    }, { passive: false });

    elements.quizPaintCanvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      stopQuizDrawing();
    });

    if (elements.quizBrushSize) {
      elements.quizBrushSize.addEventListener('input', (e) => {
        quizState.brushSize = parseInt(e.target.value, 10);
        if (elements.quizBrushVal) elements.quizBrushVal.textContent = `${quizState.brushSize}px`;
        quizCtx.lineWidth = quizState.brushSize;
      });
    }

    if (elements.btnQuizGhost) {
      elements.btnQuizGhost.addEventListener('click', toggleQuizGhost);
    }
    if (elements.btnQuizClear) {
      elements.btnQuizClear.addEventListener('click', clearQuizCanvas);
    }
    if (elements.btnQuizShuffle) {
      elements.btnQuizShuffle.addEventListener('click', shuffleQuizQuestion);
    }
    if (elements.btnQuizNext) {
      elements.btnQuizNext.addEventListener('click', shuffleQuizQuestion);
    }
    if (elements.btnQuizVerify) {
      elements.btnQuizVerify.addEventListener('click', verifyQuizWriting);
    }
  }

  function getQuizCanvasCoords(e) {
    const rect = elements.quizPaintCanvas.getBoundingClientRect();
    const scaleX = elements.quizPaintCanvas.width / rect.width;
    const scaleY = elements.quizPaintCanvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  function startQuizDrawing(e) {
    quizState.isDrawing = true;
    const { x, y } = getQuizCanvasCoords(e);
    quizState.lastPoints = [{ x, y }];
    quizCtx.beginPath();
    quizCtx.arc(x, y, quizState.brushSize / 2, 0, Math.PI * 2);
    quizCtx.fillStyle = '#05070B';
    quizCtx.fill();
    quizCtx.beginPath();
    quizCtx.moveTo(x, y);
  }

  function drawQuiz(e) {
    if (!quizState.isDrawing) return;
    const { x, y } = getQuizCanvasCoords(e);
    quizState.lastPoints.push({ x, y });
    if (quizState.lastPoints.length > 2) {
      const p1 = quizState.lastPoints[quizState.lastPoints.length - 2];
      const p2 = quizState.lastPoints[quizState.lastPoints.length - 1];
      quizCtx.quadraticCurveTo(p1.x, p1.y, (p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
      quizCtx.stroke();
    }
  }

  function stopQuizDrawing() {
    if (!quizState.isDrawing) return;
    quizState.isDrawing = false;
    quizCtx.closePath();
  }

  function clearQuizCanvas() {
    if (!quizCtx) return;
    quizCtx.fillStyle = '#FFFFFF';
    quizCtx.fillRect(0, 0, elements.quizPaintCanvas.width, elements.quizPaintCanvas.height);
    if (elements.quizFeedbackCard) elements.quizFeedbackCard.classList.add('hidden');
  }

  function toggleQuizGhost() {
    quizState.isGhostOn = !quizState.isGhostOn;
    if (elements.quizGhostOverlay) {
      elements.quizGhostOverlay.style.display = quizState.isGhostOn ? 'flex' : 'none';
    }
    if (elements.ghostBadgeIndicator) {
      elements.ghostBadgeIndicator.style.display = quizState.isGhostOn ? 'block' : 'none';
    }
    if (elements.quizGhostBtnText) {
      elements.quizGhostBtnText.textContent = quizState.isGhostOn ? 'Panduan Jiplak: ON' : 'Panduan Jiplak: OFF';
    }
  }

  function shuffleQuizQuestion() {
    if (!window.AKSARA_DATABASE) return;
    const keys = Object.keys(window.AKSARA_DATABASE);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    quizState.target = window.AKSARA_DATABASE[randomKey];

    if (elements.quizTargetCategory) elements.quizTargetCategory.textContent = quizState.target.category_name || quizState.target.category;
    if (elements.quizTargetChar) elements.quizTargetChar.textContent = quizState.target.unicode_char;
    if (elements.quizGhostOverlay) elements.quizGhostOverlay.textContent = quizState.target.unicode_char;
    if (elements.quizTargetLatin) elements.quizTargetLatin.textContent = `Tulis Aksara: "${quizState.target.latin}"`;
    if (elements.quizTargetDesc) elements.quizTargetDesc.textContent = quizState.target.desc;

    clearQuizCanvas();
  }

  async function verifyQuizWriting() {
    if (!elements.quizPaintCanvas || !quizState.target) return;
    setDiagnosticLoading(true);

    try {
      const result = await runClientInference(elements.quizPaintCanvas, false);
      setDiagnosticLoading(false);

      const isCorrect = (result.predicted.class_name === quizState.target.class_name);
      const isClose = result.top5.some(item => item.class_name === quizState.target.class_name);

      showQuizFeedback({
        is_correct: isCorrect,
        is_close: isClose,
        confidence: result.predicted.confidence,
        feedback: isCorrect
          ? `Luar biasa! Aksara ${quizState.target.latin} Anda tulis dengan sangat tepat (${result.predicted.confidence}%).`
          : isClose
          ? `Hampir tepat! Sandhangan atau goresan Anda terdeteksi mirip dengan Aksara ${result.predicted.latin}.`
          : `Goresan Anda terbaca sebagai Aksara ${result.predicted.latin} (${result.predicted.unicode_char}). Coba perhatikan kembali bentuk dasar aksara.`
      });

      renderDiagnosticResult(result);
    } catch (err) {
      setDiagnosticLoading(false);
      showToast(`Gagal: ${err.message}`, 'error');
    }
  }

  function showQuizFeedback(result) {
    if (!elements.quizFeedbackCard) return;
    elements.quizFeedbackCard.classList.remove('hidden', 'correct', 'close', 'incorrect');

    if (result.is_correct) {
      elements.quizFeedbackCard.classList.add('correct');
      if (elements.quizFeedbackBadge) elements.quizFeedbackBadge.textContent = '✅ BENAR (+10)';
      quizState.score += 10;
      quizState.streak += 1;
      showToast(`🎉 Tepat sekali! Aksara ${quizState.target.latin} ditulis dengan baik!`, 'success');
    } else if (result.is_close) {
      elements.quizFeedbackCard.classList.add('close');
      if (elements.quizFeedbackBadge) elements.quizFeedbackBadge.textContent = '⚠️ HAMPIR BENAR';
      quizState.streak = 0;
      showToast('Hampir benar! Periksa detail lengkungan/ekor aksara.', 'info');
    } else {
      elements.quizFeedbackCard.classList.add('incorrect');
      if (elements.quizFeedbackBadge) elements.quizFeedbackBadge.textContent = '❌ KURANG TEPAT';
      quizState.streak = 0;
      showToast('Kurang tepat. Coba ikuti siluet bayangan jiplak.', 'error');
    }

    if (elements.quizFeedbackConf) elements.quizFeedbackConf.textContent = `Tingkat Kepastian AI: ${result.confidence}%`;
    if (elements.quizFeedbackText) elements.quizFeedbackText.textContent = result.feedback;
    if (elements.quizScore) elements.quizScore.textContent = quizState.score;
    if (elements.quizStreak) elements.quizStreak.textContent = `${quizState.streak} 🔥`;
  }

  // --- MCQ Quiz ---
  function initMcqQuiz() {
    if (elements.btnMcqShuffle) {
      elements.btnMcqShuffle.addEventListener('click', shuffleMcqQuestion);
    }
  }

  function shuffleMcqQuestion() {
    if (!window.AKSARA_DATABASE || !elements.mcqOptionsGrid) return;
    quizState.mcqAnswered = false;

    const allKeys = Object.keys(window.AKSARA_DATABASE);
    const targetKey = allKeys[Math.floor(Math.random() * allKeys.length)];
    quizState.mcqTarget = window.AKSARA_DATABASE[targetKey];

    const distractors = [];
    while (distractors.length < 3) {
      const dKey = allKeys[Math.floor(Math.random() * allKeys.length)];
      if (dKey !== targetKey && !distractors.includes(dKey)) {
        distractors.push(dKey);
      }
    }

    const options = [quizState.mcqTarget, ...distractors.map(k => window.AKSARA_DATABASE[k])];
    options.sort(() => Math.random() - 0.5);

    if (elements.mcqTargetGlyph) elements.mcqTargetGlyph.textContent = quizState.mcqTarget.unicode_char;
    if (elements.mcqCategoryHint) elements.mcqCategoryHint.textContent = `Kategori: ${quizState.mcqTarget.category_name || quizState.mcqTarget.category}`;
    if (elements.mcqFeedbackHint) {
      elements.mcqFeedbackHint.textContent = 'Pilih salah satu jawaban yang paling tepat';
      elements.mcqFeedbackHint.style.color = '';
    }

    elements.mcqOptionsGrid.innerHTML = '';
    const letters = ['A', 'B', 'C', 'D'];

    options.forEach((opt, idx) => {
      const btn = document.createElement('button');
      btn.className = 'mcq-option-btn';
      btn.type = 'button';
      btn.innerHTML = `
        <span class="mcq-opt-letter">${letters[idx]}</span>
        <div class="mcq-opt-info">
          <span class="mcq-opt-text">${opt.latin}</span>
          <span class="mcq-opt-sub">${opt.category_name || opt.category}</span>
        </div>
      `;

      btn.addEventListener('click', () => {
        if (quizState.mcqAnswered) return;
        quizState.mcqAnswered = true;

        const isCorrect = (opt.class_name === quizState.mcqTarget.class_name);
        const allBtns = elements.mcqOptionsGrid.querySelectorAll('.mcq-option-btn');

        allBtns.forEach((b, i) => {
          b.disabled = true;
          if (options[i].class_name === quizState.mcqTarget.class_name) {
            b.classList.add('correct');
          }
        });

        if (isCorrect) {
          btn.classList.add('correct');
          quizState.mcqScore += 10;
          quizState.mcqStreak += 1;
          if (elements.mcqFeedbackHint) {
            elements.mcqFeedbackHint.textContent = `✨ Benar! Aksara ${quizState.mcqTarget.unicode_char} dibaca "${quizState.mcqTarget.latin}". (+10 Poin)`;
            elements.mcqFeedbackHint.style.color = '#34D399';
          }
          showToast(`🎯 Tepat! Aksara ${quizState.mcqTarget.latin} (+10 Poin)`, 'success');
        } else {
          btn.classList.add('incorrect');
          quizState.mcqStreak = 0;
          if (elements.mcqFeedbackHint) {
            elements.mcqFeedbackHint.textContent = `❌ Kurang tepat. Jawaban yang benar: "${quizState.mcqTarget.latin}" (${quizState.mcqTarget.unicode_char}).`;
            elements.mcqFeedbackHint.style.color = '#F87171';
          }
        }

        if (elements.mcqScore) elements.mcqScore.textContent = quizState.mcqScore;
        if (elements.mcqStreak) elements.mcqStreak.textContent = `${quizState.mcqStreak} 🔥`;

        setTimeout(() => {
          if (quizState.submode === 'mcq') shuffleMcqQuestion();
        }, 1600);
      });

      elements.mcqOptionsGrid.appendChild(btn);
    });
  }

  // =========================================================
  // 7. Kamus Aksara (120 Catalog)
  // =========================================================
  function renderCatalog(filterCat = 'all', searchQuery = '') {
    if (!window.AKSARA_DATABASE) return;
    elements.catalogGrid.innerHTML = '';
    const q = searchQuery.toLowerCase().trim();

    Object.values(window.AKSARA_DATABASE).forEach(item => {
      if (filterCat !== 'all' && item.category !== filterCat) return;
      if (q && !item.latin.toLowerCase().includes(q) && !item.class_name.toLowerCase().includes(q)) return;

      const card = document.createElement('div');
      card.className = 'catalog-item';
      card.title = `${item.latin} - Klik untuk memuat ke kanvas`;
      card.innerHTML = `
        <div class="catalog-glyph">${item.unicode_char}</div>
        <div class="catalog-label">${item.latin}</div>
      `;

      card.addEventListener('click', () => {
        switchTab('tab-canvas');
        loadSampleToCanvas(item.class_name);
      });

      elements.catalogGrid.appendChild(card);
    });
  }

  elements.catalogCategoryPills.forEach(pill => {
    pill.addEventListener('click', () => {
      elements.catalogCategoryPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      renderCatalog(pill.getAttribute('data-cat'), elements.catalogSearch.value);
    });
  });

  elements.catalogSearch.addEventListener('input', (e) => {
    const activePill = document.querySelector('.cat-pill.active');
    const cat = activePill ? activePill.getAttribute('data-cat') : 'all';
    renderCatalog(cat, e.target.value);
  });

  // =========================================================
  // 8. File Upload & Samples Loader
  // =========================================================
  elements.btnBrowse.addEventListener('click', () => elements.fileInput.click());
  elements.dropZone.addEventListener('click', () => elements.fileInput.click());

  elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('dragover');
  });

  elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('dragover');
  });

  elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.match('image.*')) {
      showToast('Harap pilih berkas gambar (PNG, JPG, WebP)!', 'error');
      return;
    }
    state.uploadedFile = file;
    elements.previewFilename.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      elements.imagePreview.src = e.target.result;
      state.uploadedImageElem = new Image();
      state.uploadedImageElem.src = e.target.result;

      elements.dropZone.classList.add('hidden');
      elements.previewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  elements.btnRemovePreview.addEventListener('click', () => {
    state.uploadedFile = null;
    state.uploadedImageElem = null;
    elements.fileInput.value = '';
    elements.imagePreview.src = '';
    elements.previewContainer.classList.add('hidden');
    elements.dropZone.classList.remove('hidden');
  });

  function loadSampleToCanvas(sampleKey) {
    clearCanvas();
    elements.canvasPlaceholder.classList.add('hidden');

    let key = sampleKey;
    if (!window.AKSARA_DATABASE[key]) {
      for (const prefix of ['aksara-dasar_', 'suku_', 'wulu_', 'taling_', 'pepet_', 'taling-tarung_']) {
        if (window.AKSARA_DATABASE[prefix + key]) {
          key = prefix + key;
          break;
        }
      }
    }

    const item = window.AKSARA_DATABASE[key];
    if (!item) return;

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);

    ctx.fillStyle = '#05070B';
    ctx.font = 'bold 220px "Noto Sans Javanese", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.unicode_char, elements.canvas.width / 2, elements.canvas.height / 2 + 10);

    saveCanvasState();
    showToast(`Contoh Aksara ${item.latin} (${item.unicode_char}) dimuat ke kanvas!`, 'info');
  }

  elements.sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const sample = chip.getAttribute('data-sample');
      loadSampleToCanvas(sample);
    });
  });

  // =========================================================
  // 9. Tab Switcher & Toast Helper
  // =========================================================
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.getAttribute('data-tab')));
  });

  function switchTab(tabId) {
    state.activeTab = tabId;
    elements.tabButtons.forEach(b => {
      const isActive = b.getAttribute('data-tab') === tabId;
      b.classList.toggle('active', isActive);
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    elements.tabContents.forEach(c => {
      c.classList.toggle('active', c.id === tabId);
    });
    if (tabId === 'tab-quiz' && !quizState.target) {
      initQuiz();
    }
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span>${type === 'success' ? '✨' : type === 'error' ? '⚠️' : 'ℹ️'}</span>
      <span>${message}</span>
    `;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // =========================================================
  // Inisialisasi Aplikasi
  // =========================================================
  initCanvas();
  renderCatalog();
  await initOnnxEngine();
});
