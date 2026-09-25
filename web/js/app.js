/**
 * AksaraAI Studio - Frontend Application Engine
 * Menangani Kanvas Digital, Drag & Drop, API Communication, Kuis, dan Kamus Aksara.
 */

document.addEventListener('DOMContentLoaded', () => {
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
    quizTarget: null,
    quizScore: 0,
    quizStreak: 0,
    lastPoints: []
  };

  // =========================================================
  // DOM Elements Selector
  // =========================================================
  const elements = {
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
  // Inisialisasi Kanvas Gambar
  // =========================================================
  function initCanvas() {
    // Inisialisasi latar studio paper #F8FAFC (anti-glare & eye-friendly)
    ctx.fillStyle = '#F8FAFC';
    ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#05070B';
    ctx.lineWidth = state.brushSize;

    saveCanvasState();

    // Event Listeners Mouse
    elements.canvas.addEventListener('mousedown', startDrawing);
    elements.canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    // Event Listeners Touch (Mobile / Tablet)
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

    // Brush Slider
    elements.brushSlider.addEventListener('input', (e) => {
      state.brushSize = parseInt(e.target.value, 10);
      elements.brushIndicator.textContent = `${state.brushSize}px`;
      ctx.lineWidth = state.brushSize;
    });

    // Undo & Clear
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

    // Interpolasi Bézier agar goresan tulisan tangan terasa sangat mulus
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
      state.undoStack.pop(); // Pop current state
      const prevState = state.undoStack[state.undoStack.length - 1];
      ctx.putImageData(prevState, 0, 0);
    } else if (state.undoStack.length === 1) {
      clearCanvas();
    }
  }

  function clearCanvas() {
    ctx.fillStyle = '#F8FAFC';
    ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    elements.canvasPlaceholder.classList.remove('hidden');
    state.undoStack = [];
    saveCanvasState();
  }

  // =========================================================
  // Quick Sample Calligraphy Loader
  // =========================================================
  elements.sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const sampleKey = chip.getAttribute('data-sample');
      loadSampleToCanvas(sampleKey);
    });
  });

  function loadSampleToCanvas(sampleKey) {
    // Bersihkan kanvas
    clearCanvas();
    elements.canvasPlaceholder.classList.add('hidden');

    // Dapatkan data Unicode
    let sampleClass = sampleKey;
    if (!sampleClass.includes('_')) {
      sampleClass = `aksara-dasar_${sampleKey}`;
    }

    const info = window.AKSARA_DATABASE ? window.AKSARA_DATABASE[sampleClass] : null;
    const char = info ? info.unicode_char : 'ꦲ';

    // Gambar aksara di kanvas dengan font Javanese tebal untuk simulasi
    ctx.fillStyle = '#05070B';
    ctx.font = 'bold 240px "Noto Sans Javanese", serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(char, elements.canvas.width / 2, elements.canvas.height / 2 + 10);

    saveCanvasState();
    showToast(`Contoh ${info ? info.latin : sampleKey} dimuat ke kanvas!`, 'success');
  }

  // =========================================================
  // Tab Navigation Handling
  // =========================================================
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      switchTab(targetId);
    });
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

    if (tabId === 'tab-quiz' && !state.quizTarget) {
      initQuiz();
    }
  }

  // =========================================================
  // File Upload & Drag-and-Drop
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
      elements.dropZone.classList.add('hidden');
      elements.previewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  elements.btnRemovePreview.addEventListener('click', () => {
    state.uploadedFile = null;
    elements.fileInput.value = '';
    elements.imagePreview.src = '';
    elements.previewContainer.classList.add('hidden');
    elements.dropZone.classList.remove('hidden');
  });

  // =========================================================
  // Mode Kuis & Latihan Interaktif (Dual-Mode System)
  // =========================================================
  const quizState = {
    submode: 'write', // 'write' | 'mcq'
    isDrawing: false,
    lastPoints: [],
    brushSize: 6,
    isGhostOn: true,
    history: [],
    maxHistory: 15,
    score: 0,
    streak: 0,
    target: null,
    // MCQ State
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

  // --- SUB-NAV SWITCHER (Latihan Tulis vs Kuis Pilihan Ganda) ---
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

  // --- KANVAS GAMBAR DEDICATED UNTUK KUIS ---
  function initQuizCanvas() {
    if (!elements.quizPaintCanvas || !quizCtx) return;

    // Inisialisasi latar studio paper #F8FAFC (anti-glare & eye-friendly)
    quizCtx.fillStyle = '#F8FAFC';
    quizCtx.fillRect(0, 0, elements.quizPaintCanvas.width, elements.quizPaintCanvas.height);
    quizCtx.lineCap = 'round';
    quizCtx.lineJoin = 'round';
    quizCtx.strokeStyle = '#05070B';
    quizCtx.lineWidth = quizState.brushSize;

    saveQuizCanvasState();

    // Mouse Listeners
    elements.quizPaintCanvas.addEventListener('mousedown', startQuizDrawing);
    elements.quizPaintCanvas.addEventListener('mousemove', drawQuiz);
    window.addEventListener('mouseup', stopQuizDrawing);

    // Touch Listeners (Mobile / Tablet)
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

    // Brush Slider
    if (elements.quizBrushSize) {
      elements.quizBrushSize.addEventListener('input', (e) => {
        quizState.brushSize = parseInt(e.target.value, 10);
        if (elements.quizBrushVal) elements.quizBrushVal.textContent = `${quizState.brushSize}px`;
        quizCtx.lineWidth = quizState.brushSize;
      });
    }

    // Undo, Clear, & Ghost Guide Toggle
    if (elements.btnQuizUndo) elements.btnQuizUndo.addEventListener('click', undoQuizCanvas);
    if (elements.btnQuizClear) elements.btnQuizClear.addEventListener('click', clearQuizCanvas);
    if (elements.btnQuizGhost) elements.btnQuizGhost.addEventListener('click', toggleGhostGuide);
    if (elements.btnQuizShuffle) elements.btnQuizShuffle.addEventListener('click', shuffleQuizQuestion);
    if (elements.btnQuizVerify) elements.btnQuizVerify.addEventListener('click', verifyQuizAttempt);
    if (elements.btnQuizNext) elements.btnQuizNext.addEventListener('click', shuffleQuizQuestion);

    // Inisialisasi sinkronisasi awal status tombol dan bayangan jiplak
    if (elements.btnQuizGhost) elements.btnQuizGhost.classList.toggle('active', quizState.isGhostOn);
    if (elements.quizGhostOverlay) elements.quizGhostOverlay.classList.toggle('hidden', !quizState.isGhostOn);
    if (elements.ghostBadgeIndicator) elements.ghostBadgeIndicator.classList.toggle('hidden', !quizState.isGhostOn);
    if (elements.quizGhostBtnText) elements.quizGhostBtnText.textContent = quizState.isGhostOn ? 'Panduan Jiplak: ON' : 'Panduan Jiplak: OFF';
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
    if (!quizCtx) return;
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
    if (!quizState.isDrawing || !quizCtx) return;
    const { x, y } = getQuizCanvasCoords(e);
    quizState.lastPoints.push({ x, y });

    if (quizState.lastPoints.length > 2) {
      const p1 = quizState.lastPoints[quizState.lastPoints.length - 2];
      const p2 = quizState.lastPoints[quizState.lastPoints.length - 1];
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2;

      quizCtx.strokeStyle = '#05070B';
      quizCtx.lineWidth = quizState.brushSize;
      quizCtx.quadraticCurveTo(p1.x, p1.y, midX, midY);
      quizCtx.stroke();
      quizCtx.beginPath();
      quizCtx.moveTo(midX, midY);
    } else {
      quizCtx.lineTo(x, y);
      quizCtx.stroke();
    }
  }

  function stopQuizDrawing() {
    if (!quizState.isDrawing) return;
    quizState.isDrawing = false;
    quizState.lastPoints = [];
    saveQuizCanvasState();
  }

  function saveQuizCanvasState() {
    if (!elements.quizPaintCanvas || !quizCtx) return;
    const snapshot = quizCtx.getImageData(0, 0, elements.quizPaintCanvas.width, elements.quizPaintCanvas.height);
    quizState.history.push(snapshot);
    if (quizState.history.length > quizState.maxHistory) {
      quizState.history.shift();
    }
  }

  function undoQuizCanvas() {
    if (!quizCtx || quizState.history.length <= 1) return;
    quizState.history.pop();
    const prev = quizState.history[quizState.history.length - 1];
    quizCtx.putImageData(prev, 0, 0);
  }

  function clearQuizCanvas() {
    if (!elements.quizPaintCanvas || !quizCtx) return;
    quizCtx.fillStyle = '#F8FAFC';
    quizCtx.fillRect(0, 0, elements.quizPaintCanvas.width, elements.quizPaintCanvas.height);
    saveQuizCanvasState();
    if (elements.quizFeedbackCard) elements.quizFeedbackCard.classList.add('hidden');
  }

  function toggleGhostGuide() {
    quizState.isGhostOn = !quizState.isGhostOn;
    if (elements.quizGhostOverlay) {
      elements.quizGhostOverlay.classList.toggle('hidden', !quizState.isGhostOn);
    }
    if (elements.ghostBadgeIndicator) {
      elements.ghostBadgeIndicator.classList.toggle('hidden', !quizState.isGhostOn);
    }
    if (elements.quizGhostBtnText) {
      elements.quizGhostBtnText.textContent = quizState.isGhostOn ? 'Panduan Jiplak: ON' : 'Panduan Jiplak: OFF';
    }
    if (elements.btnQuizGhost) {
      elements.btnQuizGhost.classList.toggle('active', quizState.isGhostOn);
    }
  }

  // --- ACANG SOAL & VERIFIKASI LATIHAN TULIS ---
  function shuffleQuizQuestion() {
    if (elements.quizFeedbackCard) elements.quizFeedbackCard.classList.add('hidden');
    clearQuizCanvas();
    if (!window.AKSARA_DATABASE) return;

    const allKeys = Object.keys(window.AKSARA_DATABASE);
    const randomKey = allKeys[Math.floor(Math.random() * allKeys.length)];
    quizState.target = window.AKSARA_DATABASE[randomKey];
    state.quizTarget = quizState.target;

    if (elements.quizTargetChar) elements.quizTargetChar.textContent = quizState.target.unicode_char;
    if (elements.quizTargetLatin) elements.quizTargetLatin.textContent = `Tulis Aksara: "${quizState.target.latin}"`;
    if (elements.quizTargetDesc) elements.quizTargetDesc.textContent = quizState.target.desc;
    if (elements.quizTargetCategory) elements.quizTargetCategory.textContent = quizState.target.category_name || quizState.target.category;

    // Update Ghost Tracing Overlay
    if (elements.quizGhostOverlay) {
      elements.quizGhostOverlay.textContent = quizState.target.unicode_char;
    }
  }

  async function verifyQuizAttempt() {
    if (!quizState.target || !elements.quizPaintCanvas) return;

    elements.quizPaintCanvas.toBlob(async (blob) => {
      if (!blob) {
        showToast('Gagal memproses gambar kanvas.', 'error');
        return;
      }

      const formData = new FormData();
      formData.append('file', blob, 'quiz_writing.png');
      formData.append('target_class', quizState.target.class_name);

      setDiagnosticLoading(true);
      try {
        const response = await fetch('/api/verify', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();
        setDiagnosticLoading(false);

        if (data.status === 'success') {
          showQuizFeedback(data);
          // Sinkronkan juga ke panel diagnostik utama di kanan
          if (data.predicted) {
            renderDiagnosticResult({
              status: 'success',
              is_confident: data.is_correct || data.confidence >= 50.0,
              threshold: 50.0,
              predicted: {
                ...data.predicted,
                confidence: data.confidence
              },
              confidence: data.confidence,
              latency_ms: data.latency_ms,
              top5: [{
                ...data.predicted,
                confidence: data.confidence
              }]
            });
          }
        } else {
          showToast(data.message || 'Gagal melakukan verifikasi jawaban.', 'error');
        }
      } catch (err) {
        setDiagnosticLoading(false);
        showToast(`Gagal terhubung ke API backend: ${err.message}`, 'error');
      }
    }, 'image/png');
  }

  function showQuizFeedback(result) {
    if (!elements.quizFeedbackCard) return;
    elements.quizFeedbackCard.classList.remove('hidden', 'correct', 'close', 'incorrect');

    if (result.is_correct) {
      elements.quizFeedbackCard.classList.add('correct');
      if (elements.quizFeedbackBadge) elements.quizFeedbackBadge.textContent = '✅ BENAR (+10)';
      quizState.score += 10;
      quizState.streak += 1;
      showToast(`🎉 Luar biasa! Aksara ${quizState.target.latin} ditulis dengan sangat tepat!`, 'success');
    } else if (result.is_close) {
      elements.quizFeedbackCard.classList.add('close');
      if (elements.quizFeedbackBadge) elements.quizFeedbackBadge.textContent = '⚠️ HAMPIR BENAR';
      quizState.streak = 0;
      showToast('Hampir benar! Periksa kembali sandhangan aksara.', 'info');
    } else {
      elements.quizFeedbackCard.classList.add('incorrect');
      if (elements.quizFeedbackBadge) elements.quizFeedbackBadge.textContent = '❌ KURANG TEPAT';
      quizState.streak = 0;
      showToast('Kurang tepat. Coba ikuti bayangan panduan jiplak.', 'error');
    }

    if (elements.quizFeedbackConf) elements.quizFeedbackConf.textContent = `Tingkat Kepastian AI: ${result.confidence}%`;
    if (elements.quizFeedbackText) elements.quizFeedbackText.textContent = result.feedback;
    if (elements.quizScore) elements.quizScore.textContent = quizState.score;
    if (elements.quizStreak) elements.quizStreak.textContent = `${quizState.streak} 🔥`;
  }

  // --- SUB-PANEL 2: KUIS PILIHAN GANDA (TEBAK AKSARA FLASHCARD) ---
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

    // Ambil 3 pengecoh acak dari database
    const distractors = [];
    while (distractors.length < 3) {
      const dKey = allKeys[Math.floor(Math.random() * allKeys.length)];
      if (dKey !== targetKey && !distractors.includes(dKey)) {
        distractors.push(dKey);
      }
    }

    // Satukan 4 pilihan dan acak urutannya
    const options = [quizState.mcqTarget, ...distractors.map(k => window.AKSARA_DATABASE[k])];
    options.sort(() => Math.random() - 0.5);

    // Tampilkan aksara dan hint
    if (elements.mcqTargetGlyph) elements.mcqTargetGlyph.textContent = quizState.mcqTarget.unicode_char;
    if (elements.mcqCategoryHint) elements.mcqCategoryHint.textContent = `Kategori: ${quizState.mcqTarget.category_name || quizState.mcqTarget.category}`;
    if (elements.mcqFeedbackHint) elements.mcqFeedbackHint.textContent = 'Pilih salah satu jawaban yang paling tepat';

    // Render 4 Tombol Pilihan Ganda
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
        handleMcqChoice(btn, opt, options);
      });

      elements.mcqOptionsGrid.appendChild(btn);
    });
  }

  function handleMcqChoice(clickedBtn, chosenOption, allOptions) {
    if (quizState.mcqAnswered) return;
    quizState.mcqAnswered = true;

    const isCorrect = (chosenOption.class_name === quizState.mcqTarget.class_name);
    const allButtons = elements.mcqOptionsGrid.querySelectorAll('.mcq-option-btn');

    allButtons.forEach((btn, idx) => {
      btn.disabled = true;
      if (allOptions[idx].class_name === quizState.mcqTarget.class_name) {
        btn.classList.add('correct');
      }
    });

    if (isCorrect) {
      clickedBtn.classList.add('correct');
      quizState.mcqScore += 10;
      quizState.mcqStreak += 1;
      if (elements.mcqFeedbackHint) {
        elements.mcqFeedbackHint.textContent = `✨ Benar! Aksara ${quizState.mcqTarget.unicode_char} dibaca "${quizState.mcqTarget.latin}". (+10 Poin)`;
        elements.mcqFeedbackHint.style.color = '#34D399';
      }
      showToast(`🎯 Tepat! Aksara ${quizState.mcqTarget.latin} (+10 Poin)`, 'success');
    } else {
      clickedBtn.classList.add('incorrect');
      quizState.mcqStreak = 0;
      if (elements.mcqFeedbackHint) {
        elements.mcqFeedbackHint.textContent = `❌ Kurang tepat. Jawaban yang benar adalah "${quizState.mcqTarget.latin}" (${quizState.mcqTarget.unicode_char}).`;
        elements.mcqFeedbackHint.style.color = '#F87171';
      }
      showToast(`Kurang tepat. Jawaban benar: ${quizState.mcqTarget.latin}`, 'info');
    }

    if (elements.mcqScore) elements.mcqScore.textContent = quizState.mcqScore;
    if (elements.mcqStreak) elements.mcqStreak.textContent = `${quizState.mcqStreak} 🔥`;

    // Otomatis muat soal berikutnya setelah 1.5 detik
    setTimeout(() => {
      if (quizState.submode === 'mcq') {
        shuffleMcqQuestion();
      }
    }, 1500);
  }

  // =========================================================
  // Kamus Aksara (120 Characters Catalog)
  // =========================================================
  function renderCatalog(filterCat = 'all', searchQuery = '') {
    if (!window.AKSARA_DATABASE) return;

    elements.catalogGrid.innerHTML = '';
    const q = searchQuery.toLowerCase().trim();

    Object.values(window.AKSARA_DATABASE).forEach(item => {
      // Filter kategori
      if (filterCat !== 'all' && item.category !== filterCat) return;

      // Filter pencarian
      if (q && !item.latin.toLowerCase().includes(q) && !item.class_name.toLowerCase().includes(q)) {
        return;
      }

      const card = document.createElement('div');
      card.className = 'catalog-item';
      card.title = `${item.latin} (${item.desc}) - Klik untuk coba di kanvas`;
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
      const cat = pill.getAttribute('data-cat');
      renderCatalog(cat, elements.catalogSearch.value);
    });
  });

  elements.catalogSearch.addEventListener('input', (e) => {
    const activePill = document.querySelector('.cat-pill.active');
    const cat = activePill ? activePill.getAttribute('data-cat') : 'all';
    renderCatalog(cat, e.target.value);
  });

  // =========================================================
  // AI Inference & API Predictions
  // =========================================================
  elements.btnPredictCanvas.addEventListener('click', () => {
    elements.canvas.toBlob(blob => {
      if (!blob) return;
      predictImageFile(blob, 'canvas_handwriting.png');
    }, 'image/png');
  });

  elements.btnPredictUpload.addEventListener('click', () => {
    if (!state.uploadedFile) {
      showToast('Silakan pilih berkas gambar terlebih dahulu!', 'error');
      return;
    }
    predictImageFile(state.uploadedFile, state.uploadedFile.name);
  });

  async function predictImageFile(fileBlob, filename) {
    const formData = new FormData();
    formData.append('file', fileBlob, filename);

    setDiagnosticLoading(true);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setDiagnosticLoading(false);

      if (data.status === 'success') {
        renderDiagnosticResult(data);
        if (data.is_confident !== false && data.predicted.confidence >= 50.0) {
          showToast(`Klasifikasi berhasil: Aksara ${data.predicted.latin} (${data.predicted.confidence}%)`, 'success');
        } else {
          showToast(`⚠️ Tulisan kurang jelas (${data.predicted.confidence}%). Periksa opsi klarifikasi di bawah.`, 'info');
        }
      } else {
        showToast(data.message || 'Gagal melakukan prediksi citra.', 'error');
      }
    } catch (err) {
      setDiagnosticLoading(false);
      showToast(`Gagal terhubung ke API backend: ${err.message}`, 'error');
    }
  }

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

    // Tampilkan panel hasil
    elements.resultPlaceholder.classList.add('hidden');
    elements.resultLoading.classList.add('hidden');
    elements.resultContent.classList.remove('hidden');

    // Latency
    if (data.latency_ms !== undefined) {
      elements.latencyVal.textContent = `${data.latency_ms} ms`;
    }

    if (isConfident) {
      // 1. KONDISI TULISAN JELAS (Keyakinan >= 50%): Tampilkan Hero Box Definitif
      if (elements.predictionHeroCard) elements.predictionHeroCard.classList.remove('hidden');
      if (elements.ambiguityCard) elements.ambiguityCard.classList.add('hidden');
      if (elements.breakdownCard) elements.breakdownCard.classList.remove('hidden');

      if (elements.confMeterLabel) elements.confMeterLabel.textContent = 'Tingkat Keyakinan Tertinggi';
      if (elements.resTop5Title) elements.resTop5Title.textContent = '5 Kemungkinan Teratas (Top-5 Predictions)';

      // Isi Hero Card
      elements.resUnicode.textContent = pred.unicode_char || 'ꦄ';
      elements.resLatin.textContent = pred.latin ? `Aksara ${pred.latin}` : pred.class_name;
      elements.resCategory.textContent = pred.category_name || pred.category;
      elements.resConfidenceBadge.textContent = `${pred.confidence}% Pasti`;
      elements.resConfidenceBadge.style.background = '';
      elements.resConfidenceBadge.style.color = '';
      elements.resConfidenceBadge.style.border = '';
      elements.resDesc.textContent = pred.desc || 'Aksara Jawa';

      // Linguistic Breakdown
      elements.resBaseConsonant.textContent = `${pred.base_consonant || 'Baku'}`;
      elements.resSandhanganType.textContent = pred.category_name || '-';
      elements.resVowel.textContent = pred.vowel ? `${pred.vowel} (/${pred.vowel}/)` : 'a (/a/)';
      elements.resPosition.textContent = pred.position || 'Bentuk Baku';
    } else {
      // 2. KONDISI TULISAN KURANG JELAS / ANOMALI (Keyakinan < 50%):
      // Sembunyikan hasil definitif, tampilkan kartu klarifikasi "Apakah yang kamu maksud ini?"
      if (elements.predictionHeroCard) elements.predictionHeroCard.classList.add('hidden');
      if (elements.ambiguityCard) elements.ambiguityCard.classList.remove('hidden');
      if (elements.breakdownCard) elements.breakdownCard.classList.add('hidden');

      if (elements.ambiguityBadge) elements.ambiguityBadge.textContent = `Tulisan Kurang Jelas (${pred.confidence}% < ${CONFIDENCE_THRESHOLD}%)`;
      if (elements.ambiguityDesc) {
        elements.ambiguityDesc.textContent = data.clarification_message ||
          `Goresan dinilai ambigu atau kurang jelas sehingga belum mencapai ambang keyakinan ${CONFIDENCE_THRESHOLD}%. AI menemukan 5 kemungkinan terdekat berikut. Silakan klik aksara yang kamu maksud:`;
      }

      if (elements.confMeterLabel) elements.confMeterLabel.textContent = `Keyakinan Teratas (${pred.confidence}% - Di Bawah Ambang Batas ${CONFIDENCE_THRESHOLD}%)`;
      if (elements.resTop5Title) elements.resTop5Title.textContent = '🔍 Apakah yang Kamu Maksud Salah Satu Aksara Ini? (Klik untuk Memilih)';
    }

    // Confidence Progress Bar
    elements.resConfidencePct.textContent = `${pred.confidence}%`;
    elements.resProgressBar.style.width = `${Math.min(100, pred.confidence)}%`;

    // Dynamic Color berdasarkan confidence
    if (pred.confidence >= 80) {
      elements.resProgressBar.style.background = 'linear-gradient(90deg, #10B981, #34D399)';
    } else if (pred.confidence >= 50) {
      elements.resProgressBar.style.background = 'linear-gradient(90deg, #F59E0B, #FBBF24)';
    } else {
      elements.resProgressBar.style.background = 'linear-gradient(90deg, #EF4444, #F87171)';
    }

    // Top-5 Candidate List
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

    if (elements.resBaseConsonant) elements.resBaseConsonant.textContent = item.base_consonant || (item.latin ? item.latin.charAt(0).toUpperCase() + item.latin.slice(1) : 'Baku');
    if (elements.resSandhanganType) elements.resSandhanganType.textContent = item.category_name || item.category || '-';
    if (elements.resVowel) elements.resVowel.textContent = item.vowel ? `${item.vowel} (/${item.vowel}/)` : 'a (/a/)';
    if (elements.resPosition) elements.resPosition.textContent = item.position || 'Bentuk Baku';

    showToast(`✨ Anda memilih Aksara ${item.latin || item.class_name}!`, 'success');
  }

  // =========================================================
  // Toast Notifications
  // =========================================================
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
  // Helper: Memuat Contoh Aksara ke Kanvas Gambar
  // =========================================================
  function loadSampleToCanvas(className) {
    if (!window.AKSARA_DATABASE) return;

    let key = className;
    if (!window.AKSARA_DATABASE[key]) {
      if (window.AKSARA_DATABASE[`aksara-dasar_${key}`]) {
        key = `aksara-dasar_${key}`;
      } else if (window.AKSARA_DATABASE[`suku_${key}`]) {
        key = `suku_${key}`;
      } else if (window.AKSARA_DATABASE[`wulu_${key}`]) {
        key = `wulu_${key}`;
      } else if (window.AKSARA_DATABASE[`taling_${key}`]) {
        key = `taling_${key}`;
      }
    }

    const item = window.AKSARA_DATABASE[key];
    if (!item) return;

    // Bersihkan kanvas ke latar putih
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    elements.canvasPlaceholder.classList.add('hidden');

    // Gambar karakter aksara menggunakan Noto Sans Javanese
    ctx.fillStyle = '#05070B';
    ctx.font = 'bold 220px "Noto Sans Javanese", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.unicode_char, elements.canvas.width / 2, elements.canvas.height / 2 + 10);

    saveCanvasState();
    showToast(`Memuat contoh: ${item.latin} (${item.unicode_char})`, 'info');
  }

  // Event listener tombol chip contoh cepat
  document.querySelectorAll('.samples-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sample = chip.getAttribute('data-sample');
      loadSampleToCanvas(sample);
    });
  });

  // Inisialisasi awal
  initCanvas();
  initQuiz();
  renderCatalog();
});
