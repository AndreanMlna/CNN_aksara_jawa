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
      if (typeof self !== 'undefined' && self.crossOriginIsolated) {
        ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      } else {
        ort.env.wasm.numThreads = 1; // Single-thread murni jika isolasi origin tidak aktif
      }
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

  function getInkBoundingBox(sourceCanvas) {
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const imgData = sourceCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, width, height);
    const pixelData = imgData.data;

    let minX = width, maxX = -1, minY = height, maxY = -1;
    const inkPoints = [];

    // Perimeter guard: skip 4 border pixels
    for (let y = 4; y < height - 4; y++) {
      for (let x = 4; x < width - 4; x++) {
        const offset = (y * width + x) * 4;
        const r = pixelData[offset];
        const g = pixelData[offset + 1];
        const b = pixelData[offset + 2];
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
    const { minX, sw, inkPoints } = bbox;
    const xCutoff = minX + 0.55 * sw;

    let bodyMaxY = -1, tailMaxY = -1, bodyMinY = 9999;

    for (const point of inkPoints) {
      if (point.x <= xCutoff) {
        if (point.y > bodyMaxY) bodyMaxY = point.y;
        if (point.y < bodyMinY) bodyMinY = point.y;
      } else {
        if (point.y > tailMaxY) tailMaxY = point.y;
      }
    }

    if (bodyMaxY === -1 || tailMaxY === -1) return { hasSuku: false, ratio: 0.0 };

    const bodyHeight = Math.max(1, bodyMaxY - bodyMinY + 1);
    const descenderPx = tailMaxY - bodyMaxY;
    const ratio = descenderPx / bodyHeight;

    const hasSuku = (descenderPx >= 35) && (ratio >= 0.35);
    return { hasSuku, ratio: Math.max(0, ratio) };
  }

  function detectTalingAndTarungJS(bbox) {
    if (!bbox) return { hasTaling: false, hasTarung: false };
    const { minX, sw, sh, inkPoints } = bbox;
    const aspect = sw / Math.max(sh, 1);
    if (aspect < 0.95) return { hasTaling: false, hasTarung: false };

    // Taling (ꦺ) wajib memuat celah pemisah vertikal (valley) antara glif taling kiri dan konsonan kanan
    const valleyStart = minX + 0.20 * sw;
    const valleyEnd = minX + 0.48 * sw;
    const valleyColCounts = {};
    let leftCount = 0;
    let leftMinY = 9999, leftMaxY = -1;

    for (const point of inkPoints) {
      if (point.x >= valleyStart && point.x <= valleyEnd) {
        const col = Math.round(point.x);
        valleyColCounts[col] = (valleyColCounts[col] || 0) + 1;
      }
      if (point.x <= valleyStart) {
        leftCount++;
        if (point.y < leftMinY) leftMinY = point.y;
        if (point.y > leftMaxY) leftMaxY = point.y;
      }
    }

    const colVals = Object.values(valleyColCounts);
    const minColVal = colVals.length > 0 ? Math.min(...colVals) : 999;
    const leftHeight = leftMaxY - leftMinY + 1;

    const hasTaling = (minColVal <= 2) && (leftCount >= 15) && (leftHeight >= 0.55 * sh);

    // Deteksi Tarung kanan (Aksara 3 glif)
    let hasTarung = false;
    if (aspect >= 1.25) {
      const tarungStart = minX + 0.55 * sw;
      const tarungEnd = minX + 0.88 * sw;
      const tarungColCounts = {};
      let farRightCount = 0;

      for (const point of inkPoints) {
        if (point.x >= tarungStart && point.x <= tarungEnd) {
          const col = Math.round(point.x);
          tarungColCounts[col] = (tarungColCounts[col] || 0) + 1;
        }
        if (point.x >= minX + 0.85 * sw) {
          farRightCount++;
        }
      }

      const tarungMin = Object.keys(tarungColCounts).length > 0
        ? Math.min(...Object.values(tarungColCounts))
        : 999;

      if (tarungMin <= 3 && farRightCount >= 25) {
        hasTarung = true;
      }
    }

    return { hasTaling, hasTarung };
  }

  function renderStrokeToCard(sourceCanvas, bbox) {
    const { minX, minY, sw, sh } = bbox;
    const strokeCanvas = document.createElement('canvas');
    strokeCanvas.width = sw;
    strokeCanvas.height = sh;
    strokeCanvas.getContext('2d').drawImage(sourceCanvas, minX, minY, sw, sh, 0, 0, sw, sh);

    // View 0: Native bujur sangkar 500x500
    const canvasSquare = document.createElement('canvas');
    canvasSquare.width = 500;
    canvasSquare.height = 500;
    const ctxSquare = canvasSquare.getContext('2d');
    ctxSquare.fillStyle = '#FFFFFF';
    ctxSquare.fillRect(0, 0, 500, 500);
    const scale0 = Math.min(360 / Math.max(sw, 1), 360 / Math.max(sh, 1));
    const nw0 = Math.max(1, Math.round(sw * scale0));
    const nh0 = Math.max(1, Math.round(sh * scale0));
    ctxSquare.drawImage(strokeCanvas, 0, 0, sw, sh, Math.floor((500 - nw0) / 2), Math.floor((500 - nh0) / 2), nw0, nh0);

    // View Card: Kartu putih 600x500
    const canvasCard = document.createElement('canvas');
    canvasCard.width = 600;
    canvasCard.height = 500;
    const ctxCard = canvasCard.getContext('2d');
    ctxCard.fillStyle = '#FFFFFF';
    ctxCard.fillRect(0, 0, 600, 500);
    const scaleC = Math.min(360 / Math.max(sw, 1), 340 / Math.max(sh, 1));
    let nwc = Math.max(1, Math.round(sw * scaleC));
    const nhc = Math.max(1, Math.round(sh * scaleC));

    // Adaptive width relaxation: Cegah kolaps punuk ganda (misal Ga -> Gu) saat goresan memiliki ekor suku panjang
    if (nwc < 250 && sw / Math.max(sh, 1) < 0.85) {
      nwc = Math.min(360, Math.max(nwc, Math.min(255, Math.round(nwc * 1.25))));
    }

    const px = Math.max(15, Math.min(600 - nwc - 15, Math.floor(245 - nwc / 2)));
    const py = Math.max(15, Math.min(500 - nhc - 10, 430 - nhc));
    ctxCard.drawImage(strokeCanvas, 0, 0, sw, sh, px, py, nwc, nhc);

    // View 1: Ultra-wide 1528x540
    const canvasUltraWide = document.createElement('canvas');
    canvasUltraWide.width = 1528;
    canvasUltraWide.height = 540;
    const ctxUltraWide = canvasUltraWide.getContext('2d');
    ctxUltraWide.fillStyle = '#000000';
    ctxUltraWide.fillRect(0, 0, 1528, 540);
    ctxUltraWide.drawImage(canvasCard, 464, 20);

    // View 2: Medium-wide 882x540
    const canvasMediumWide = document.createElement('canvas');
    canvasMediumWide.width = 882;
    canvasMediumWide.height = 540;
    const ctxMediumWide = canvasMediumWide.getContext('2d');
    ctxMediumWide.fillStyle = '#000000';
    ctxMediumWide.fillRect(0, 0, 882, 540);
    ctxMediumWide.drawImage(canvasCard, 140, 20);

    // View 3: Native Taling 683x540
    const canvasTaling = document.createElement('canvas');
    canvasTaling.width = 683;
    canvasTaling.height = 540;
    const ctxTaling = canvasTaling.getContext('2d');
    ctxTaling.fillStyle = '#000000';
    ctxTaling.fillRect(0, 0, 683, 540);
    ctxTaling.drawImage(canvasCard, 41, 20);

    return { canvasSquare, canvasUltraWide, canvasMediumWide, canvasTaling };
  }

  function canvasToFloat32Tensor(sourceCanvas) {
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 224;
    resizedCanvas.height = 224;
    const ctxResized = resizedCanvas.getContext('2d');
    ctxResized.drawImage(sourceCanvas, 0, 0, sourceCanvas.width, sourceCanvas.height, 0, 0, 224, 224);

    const pixelData = ctxResized.getImageData(0, 0, 224, 224).data;
    const floatArray = new Float32Array(3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
      const r = pixelData[i * 4] / 255.0;
      const g = pixelData[i * 4 + 1] / 255.0;
      const b = pixelData[i * 4 + 2] / 255.0;

      // Planar RGB ordering [3, 224, 224]
      floatArray[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      floatArray[224 * 224 + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      floatArray[2 * 224 * 224 + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    return new ort.Tensor('float32', floatArray, [1, 3, 224, 224]);
  }

  function softmax(logits) {
    const maxVal = Math.max(...logits);
    const exps = logits.map(z => Math.exp(z - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
  }

  // =========================================================
  // =========================================================
  // 4. In-Browser Multi-View Consensus Inference
  // =========================================================
  const CONSONANTS_TO_SUKU_MAP = {
    ga: 'gu', ra: 'ru', ka: 'ku', ca: 'cu', ba: 'bu',
    ja: 'ju', da: 'du', sa: 'su', ta: 'tu', na: 'nu',
    pa: 'pu', la: 'lu', ma: 'mu', wa: 'wu', ya: 'yu',
    ha: 'hu', dha: 'dhu', tha: 'thu', nga: 'ngu', nya: 'nyu'
  };

  function blendSukuProbabilities(viewProbs, state) {
    const { probSquare, probUltraWide, probMediumWide } = viewProbs;
    const probabilities = new Array(state.totalClasses).fill(0.0);
    
    // Dataset-grounded: 16 kelas sandhangan suku dengan rasio ultra-wide (1528x540 / 1512x540)
    const wideSukuKeys = [
      'bu', 'dhu', 'du', 'gu', 'hu', 'ju', 'lu', 'mu',
      'ngu', 'nyu', 'pu', 'su', 'thu', 'tu', 'wu', 'yu'
    ];
    let wideSukuScore = 0;

    for (const key of wideSukuKeys) {
      const classIndex = state.classToIdx[`suku_${key}`];
      if (classIndex !== undefined) {
        wideSukuScore += probUltraWide[classIndex];
      }
    }

    const isWideSukuDominant = (wideSukuScore >= 0.20);

    // Bukti visual konsonan dasar dari View 0 (Square) untuk sinergi Bayesian
    const baseEvidence = {};
    for (const [baseC, sukuC] of Object.entries(CONSONANTS_TO_SUKU_MAP)) {
      const baseIdx = state.classToIdx[`aksara-dasar_${baseC}`];
      if (baseIdx !== undefined) {
        baseEvidence[`suku_${sukuC}`] = probSquare[baseIdx] || 0.0;
      }
    }

    for (let i = 0; i < state.totalClasses; i++) {
      const className = state.idxToClass[i] || '';
      if (className.startsWith('suku_')) {
        const baseBoost = baseEvidence[className] || 0.0;
        if (isWideSukuDominant) {
          // UltraWide dominant dipadukan dengan preservasi bentuk glif Square & sinergi konsonan dasar
          probabilities[i] = 0.70 * probUltraWide[i] + 0.15 * probMediumWide[i] + 0.15 * (probSquare[i] + baseBoost);
        } else {
          // Konsensus seimbang dengan kontribusi kuat Square & konsonan dasar
          probabilities[i] = 0.40 * probUltraWide[i] + 0.40 * probMediumWide[i] + 0.20 * (probSquare[i] + baseBoost);
        }
      } else if (className.startsWith('aksara-dasar_') || className.startsWith('pepet_')) {
        probabilities[i] = 0.20 * probSquare[i];
      } else if (className.startsWith('wulu_') || className.startsWith('taling-tarung_') || className.startsWith('taling_')) {
        probabilities[i] = 0.25 * probMediumWide[i];
      } else {
        probabilities[i] = 0.10 * probMediumWide[i];
      }
    }
    return probabilities;
  }

  function blendTalingProbabilities(viewProbs, state) {
    const { probSquare, probUltraWide, probMediumWide, probTaling } = viewProbs;
    const probabilities = new Array(state.totalClasses).fill(0.0);

    for (let i = 0; i < state.totalClasses; i++) {
      const className = state.idxToClass[i] || '';
      if (className.startsWith('taling_')) {
        probabilities[i] = 0.65 * probTaling[i] + 0.35 * probMediumWide[i];
      } else if (className.startsWith('taling-tarung_')) {
        probabilities[i] = 0.05 * probUltraWide[i];
      } else if (className.startsWith('aksara-dasar_') || className.startsWith('pepet_')) {
        probabilities[i] = 0.25 * probSquare[i];
      } else {
        probabilities[i] = 0.15 * probMediumWide[i];
      }
    }
    return probabilities;
  }

  function blendTalingTarungProbabilities(viewProbs, state) {
    const { probSquare, probUltraWide, probMediumWide, probTaling } = viewProbs;
    const probabilities = new Array(state.totalClasses).fill(0.0);

    for (let i = 0; i < state.totalClasses; i++) {
      const className = state.idxToClass[i] || '';
      if (className.startsWith('taling-tarung_')) {
        probabilities[i] = 0.70 * probUltraWide[i] + 0.30 * probMediumWide[i];
      } else if (className.startsWith('taling_')) {
        probabilities[i] = 0.10 * probTaling[i];
      } else if (className.startsWith('aksara-dasar_') || className.startsWith('pepet_')) {
        probabilities[i] = 0.15 * probSquare[i];
      } else {
        probabilities[i] = 0.10 * probMediumWide[i];
      }
    }
    return probabilities;
  }

  function blendBaseGlyphProbabilities(viewProbs, state) {
    const { probSquare, probMediumWide } = viewProbs;
    const probabilities = new Array(state.totalClasses).fill(0.0);

    for (let i = 0; i < state.totalClasses; i++) {
      const className = state.idxToClass[i] || '';
      if (className.startsWith('aksara-dasar_') || className.startsWith('pepet_')) {
        probabilities[i] = 0.80 * probSquare[i] + 0.20 * probMediumWide[i];
      } else if (className.startsWith('wulu_')) {
        probabilities[i] = 0.75 * probMediumWide[i] + 0.25 * probSquare[i];
      } else {
        probabilities[i] = 0.15 * probSquare[i];
      }
    }
    return probabilities;
  }

  function blendDomainProbabilities(morphology, viewProbs, state) {
    const { hasSuku, hasTaling, hasTarung } = morphology;
    let blended;

    if (hasSuku) {
      blended = blendSukuProbabilities(viewProbs, state);
    } else if (hasTaling && !hasTarung) {
      blended = blendTalingProbabilities(viewProbs, state);
    } else if (hasTaling && hasTarung) {
      blended = blendTalingTarungProbabilities(viewProbs, state);
    } else {
      blended = blendBaseGlyphProbabilities(viewProbs, state);
    }

    const totalSum = blended.reduce((acc, val) => acc + val, 0);
    return totalSum > 0 ? blended.map(val => val / totalSum) : blended;
  }

  function formatPredictionResults(finalProbs, latencyMs) {
    const candidates = finalProbs.map((prob, idx) => ({
      class_name: state.idxToClass[idx],
      confidence: parseFloat((prob * 100).toFixed(2))
    }));

    candidates.sort((a, b) => b.confidence - a.confidence);
    const top5 = candidates.slice(0, 5).map(candidate => {
      const meta = window.AKSARA_DATABASE ? window.AKSARA_DATABASE[candidate.class_name] : null;
      return {
        ...candidate,
        latin: meta ? meta.latin : candidate.class_name,
        unicode_char: meta ? meta.unicode_char : 'ꦄ',
        category: meta ? meta.category : 'Aksara',
        category_name: meta ? meta.category_name : 'Aksara Jawa',
        desc: meta ? meta.desc : '',
        base_consonant: meta ? meta.base_consonant : '',
        vowel: meta ? meta.vowel : '',
        position: meta ? meta.position : ''
      };
    });

    const predicted = top5[0] || { class_name: 'Unknown', confidence: 0.0 };
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
        'Goresan dinilai ambigu atau belum mencapai ambang batas keyakinan 50.0%. AI menemukan 5 kemungkinan terdekat berikut. Silakan klik aksara yang kamu maksud:'
    };
  }

  async function runClientInference(sourceCanvas, isUploaded = false) {
    if (!state.ortSession) {
      throw new Error('Model ONNX belum siap. Silakan tunggu beberapa detik.');
    }

    const startTime = performance.now();
    let finalProbs = [];

    if (isUploaded) {
      // Direct single-pass inference untuk citra yang diunggah
      const tensor = canvasToFloat32Tensor(sourceCanvas);
      const results = await state.ortSession.run({ input: tensor });
      finalProbs = softmax(Array.from(results.output.data));
    } else {
      // Kanvas Tulis: Domain-Aware Consensus Multi-View
      const bbox = getInkBoundingBox(sourceCanvas);
      if (!bbox) {
        throw new Error('Kanvas kosong. Silakan tulis karakter aksara terlebih dahulu.');
      }

      const morphology = {
        ...detectSukuDescenderJS(bbox),
        ...detectTalingAndTarungJS(bbox)
      };

      const views = renderStrokeToCard(sourceCanvas, bbox);
      const [resSquare, resUltraWide, resMediumWide, resTaling] = await Promise.all([
        state.ortSession.run({ input: canvasToFloat32Tensor(views.canvasSquare) }),
        state.ortSession.run({ input: canvasToFloat32Tensor(views.canvasUltraWide) }),
        state.ortSession.run({ input: canvasToFloat32Tensor(views.canvasMediumWide) }),
        state.ortSession.run({ input: canvasToFloat32Tensor(views.canvasTaling) })
      ]);

      const viewProbs = {
        probSquare: softmax(Array.from(resSquare.output.data)),
        probUltraWide: softmax(Array.from(resUltraWide.output.data)),
        probMediumWide: softmax(Array.from(resMediumWide.output.data)),
        probTaling: softmax(Array.from(resTaling.output.data))
      };

      finalProbs = blendDomainProbabilities(morphology, viewProbs, state);
    }

    const latencyMs = Math.round(performance.now() - startTime);
    return formatPredictionResults(finalProbs, latencyMs);
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
