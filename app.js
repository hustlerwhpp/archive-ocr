/* Архив OCR — всё распознавание только в браузере, без сервера.
   Движок: Tesseract.js v6 (WebAssembly, Web Worker). PDF: pdf.js.
   Никаких ключей, паролей и отправки данных: фото не покидают компьютер. */
'use strict';

// ---------- Утилиты и состояние ----------
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let pages = [];            // очередь страниц: {id,name,src,knownText,text,words,confidence,timeMs,done}
let selectedId = null;
let pageCounter = 0;
let worker = null;         // один воркер Tesseract на сессию (ленивая загрузка)
let workerLang = null;
let stopRequested = false; // остановка пакетного распознавания
let benchStopRequested = false;
let benchRecommendation = null; // {quality, binarize, label}

// Режимы качества: ограничение длинной стороны фото (слабым ПК — меньше пикселей = быстрее).
const QUALITY = {
  economy: { maxSide: 1200, label: 'Экономный' },
  normal: { maxSide: 1800, label: 'Обычный' },
  max: { maxSide: 2400, label: 'Максимальное качество' },
};

// Известный текст образца — нужен и для кнопки «Пример для проверки», и для бенчмарка.
const SAMPLE_TEXT = [
  'СПРАВКА',
  'Выдана Иванову Ивану Ивановичу',
  'в том, что он работал в архиве',
  'с 12 марта 1954 года по 8 октября 1962 года',
  'в должности старшего архивиста.',
  'Основание: приказ № 47 от 12 марта 1954 года.',
  'Заведующий архивом: Петрова А. С.',
  '15 октября 1962 года.',
].join('\n');

function setStatus(text) { $('status').textContent = text; }
function setProgress(pct) { $('progressBar').style.width = Math.max(0, Math.min(100, pct)) + '%'; }
function currentQuality() {
  const checked = document.querySelector('input[name="quality"]:checked');
  return checked ? checked.value : 'normal';
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ---------- Загрузка файлов: выбор, drag-and-drop, вставка из буфера ----------
function addPage(name, src, knownText) {
  const page = { id: ++pageCounter, name, src, knownText: knownText || null, text: '', words: [], confidence: null, timeMs: null, done: false };
  pages.push(page);
  renderQueue();
  if (selectedId === null) selectPage(page.id);
  return page;
}

async function addImageFile(file) {
  const src = URL.createObjectURL(file);
  addPage(file.name || 'Фото', src);
  setStatus('Добавлено: ' + (file.name || 'фото из буфера'));
}

async function addPdfFile(file) {
  if (typeof pdfjsLib === 'undefined') {
    setStatus('Ошибка: pdf.js не загрузился (нужен интернет для CDN при первом запуске).');
    return;
  }
  setStatus('Читаю PDF: ' + file.name + ' …');
  const buf = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  for (let n = 1; n <= pdf.numPages; n++) {
    const pdfPage = await pdf.getPage(n);
    const viewport = pdfPage.getViewport({ scale: 2 });
    const canvas = document.createElement('canvas');
    canvas.width = viewport.width; canvas.height = viewport.height;
    await pdfPage.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
    addPage(file.name + ' — стр. ' + n, canvas.toDataURL('image/png'));
  }
  setStatus('PDF добавлен: ' + pdf.numPages + ' стр.');
}

async function addFiles(fileList) {
  for (const file of fileList) {
    if (/pdf$/i.test(file.type) || /\.pdf$/i.test(file.name)) await addPdfFile(file);
    else if (/^image\//i.test(file.type)) await addImageFile(file);
    else setStatus('Пропущен файл (нужны JPG/PNG/WebP или PDF): ' + file.name);
  }
}

// ---------- Предобработка на Canvas: серый, контраст, бинаризация по Отсу, апскейл, поворот ----------
function otsuThreshold(data) {
  // data — массив яркостей 0..255. Возвращает порог Отсу.
  const hist = new Array(256).fill(0);
  for (let i = 0; i < data.length; i++) hist[data[i]]++;
  const total = data.length;
  let sum = 0; for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, maxVar = -1, threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (wB === 0) continue;
    const wF = total - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; threshold = t; }
  }
  return threshold;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Не удалось прочитать изображение.'));
    img.src = src;
  });
}

// opts: {maxSide, binarize, grayscale, contrast, rotateDeg}
// Возвращает canvas, готовый для Tesseract, и его размеры.
async function preprocessImage(src, opts) {
  const img = await loadImage(src);
  const maxSide = opts.maxSide || 1800;
  const longest = Math.max(img.naturalWidth, img.naturalHeight);
  // Апскейл только в режиме максимального качества и только для маленьких фото;
  // в экономном режиме фото лишь уменьшаются — это и есть экономия для слабого ПК.
  let scale = longest > maxSide ? maxSide / longest : 1;
  if (opts.allowUpscale && longest < maxSide) scale = Math.min(1.5, maxSide / longest);
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
  if (opts.rotateDeg) {
    ctx.save(); ctx.translate(w / 2, h / 2); ctx.rotate(opts.rotateDeg * Math.PI / 180);
    ctx.drawImage(img, -w / 2, -h / 2, w, h); ctx.restore();
  } else {
    ctx.drawImage(img, 0, 0, w, h);
  }
  const imageData = ctx.getImageData(0, 0, w, h);
  const d = imageData.data;
  const gray = new Uint8Array(w * h);
  const contrast = opts.contrast || 1;
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    let g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    if (opts.grayscale === false) g = (d[i] + d[i + 1] + d[i + 2]) / 3; // запасной вариант
    g = Math.max(0, Math.min(255, (g - 128) * contrast + 128));
    gray[p] = g;
  }
  let threshold = 127;
  if (opts.binarize) threshold = otsuThreshold(gray);
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    const v = opts.binarize ? (gray[p] >= threshold ? 255 : 0) : gray[p];
    d[i] = v; d[i + 1] = v; d[i + 2] = v; d[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function readSettings(overrides) {
  const o = overrides || {};
  return {
    maxSide: o.maxSide || QUALITY[currentQuality()].maxSide,
    binarize: o.binarize !== undefined ? o.binarize : $('binarizeCheck').checked,
    grayscale: o.grayscale !== undefined ? o.grayscale : $('grayscaleCheck').checked,
    contrast: o.contrast || parseFloat($('contrastRange').value) || 1.2,
    rotateDeg: o.rotateDeg !== undefined ? o.rotateDeg : parseFloat($('rotateRange').value) || 0,
    allowUpscale: o.allowUpscale !== undefined ? o.allowUpscale : currentQuality() === 'max',
  };
}

// ---------- Tesseract.js v6: один воркер на сессию, ленивая загрузка ----------
async function getWorker(lang) {
  if (typeof Tesseract === 'undefined') {
    throw new Error('Движок Tesseract.js не загрузился. Проверьте интернет при первом запуске (CDN jsDelivr) и обновите страницу.');
  }
  if (worker && workerLang === lang) return worker;
  if (worker) { try { await worker.terminate(); } catch (e) { /* игнорируем */ } worker = null; }
  setStatus('Загружаю движок OCR и язык ' + lang + ' (первый раз — до минуты)…');
  // В v6 язык и OEM задаются прямо в createWorker, отдельные loadLanguage/initialize не нужны.
  worker = await Tesseract.createWorker(lang, 1, {
    logger: (m) => {
      if (m.status === 'recognizing text') setProgress(Math.round((m.progress || 0) * 100));
    },
  });
  workerLang = lang;
  const psm = $('psmSelect').value || '3';
  try { await worker.setParameters({ tessedit_pageseg_mode: psm }); } catch (e) { /* PSM по умолчанию */ }
  return worker;
}

// Вытаскивает слова с координатами и уверенностью из результата Tesseract (v6: data.words или блоки).
function extractWords(data) {
  if (data && Array.isArray(data.words) && data.words.length) return data.words;
  const out = [];
  const blocks = (data && data.blocks) || [];
  for (const b of blocks) for (const p of (b.paragraphs || [])) for (const l of (p.lines || [])) for (const w of (l.words || [])) out.push(w);
  return out;
}

// ---------- Распознавание ----------
async function recognizePage(page, settingsOverride) {
  const lang = $('langSelect').value || 'rus';
  const settings = readSettings(settingsOverride);
  const started = performance.now();
  setStatus('Предобработка: ' + page.name + ' …');
  const canvas = await preprocessImage(page.src, settings);
  page.preWidth = canvas.width; page.preHeight = canvas.height;
  const w = await getWorker(lang);
  setStatus('Распознаю: ' + page.name + ' …');
  // Третий аргумент включает блоки/слова в выдаче v6 (по умолчанию — только текст).
  const result = await w.recognize(canvas, {}, { blocks: true });
  const data = result.data || {};
  page.text = data.text || '';
  page.words = extractWords(data).map((wd) => ({
    text: wd.text || '', confidence: (typeof wd.confidence === 'number' ? wd.confidence : 100),
    bbox: wd.bbox || null,
  }));
  page.confidence = (typeof data.confidence === 'number' ? data.confidence : null);
  page.timeMs = Math.round(performance.now() - started);
  page.done = true;
  return page;
}

async function recognizeAll() {
  const todo = pages.filter((p) => !p.done);
  if (!todo.length) { setStatus('Нет страниц для распознавания: добавьте фото или нажмите «Пример для проверки».'); return; }
  stopRequested = false;
  $('stopBtn').disabled = false; $('recognizeAllBtn').disabled = true;
  try {
    for (let i = 0; i < todo.length; i++) {
      if (stopRequested) { setStatus('Остановлено пользователем. Распознано ' + i + ' из ' + todo.length + '.'); break; }
      const page = todo[i];
      setProgress(Math.round((i / todo.length) * 100));
      await recognizePage(page);
      renderQueue();
      if (page.id === selectedId) renderEditor(page);
      setStatus('Готово: ' + page.name + ' — ' + page.timeMs + ' мс, уверенность ' + (page.confidence === null ? '—' : Math.round(page.confidence) + '%'));
      if (i < todo.length - 1) await sleep(400); // пауза между страницами — не перегружаем слабый ПК
    }
  } catch (err) {
    setStatus('Ошибка: ' + (err && err.message ? err.message : err));
  } finally {
    setProgress(100);
    $('stopBtn').disabled = true; $('recognizeAllBtn').disabled = false;
    renderQueue();
  }
}

// ---------- Очередь и редактор ----------
function renderQueue() {
  const list = $('queueList');
  list.innerHTML = '';
  pages.forEach((p) => {
    const li = document.createElement('li');
    li.textContent = p.name + (p.done ? ' — готово (' + (p.confidence === null ? '—' : Math.round(p.confidence) + '%') + ')' : ' — не распознано');
    if (p.id === selectedId) li.classList.add('selected');
    if (p.done) li.classList.add('done');
    li.addEventListener('click', () => selectPage(p.id));
    list.appendChild(li);
  });
}

function selectedPage() { return pages.find((p) => p.id === selectedId) || null; }

function selectPage(id) {
  selectedId = id;
  const page = selectedPage();
  renderQueue();
  if (page) renderEditor(page);
}

function updateStats(page) {
  const text = $('textEditor').innerText || '';
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  $('stats').textContent = 'Символов: ' + text.length + ' · Слов: ' + words +
    ' · Уверенность: ' + (page && page.confidence !== null && page.confidence !== undefined ? Math.round(page.confidence) + '%' : '—') +
    ' · Время: ' + (page && page.timeMs ? (page.timeMs / 1000).toFixed(1) + ' сек' : '—');
}

// Рисует рамку выбранного слова на фото. Координаты слов — в системе предобработанного
// изображения, поэтому пересчитываем их в размер показанного фото.
function drawOverlay(page, selectedIndex) {
  const img = $('previewImg');
  const canvas = $('overlayCanvas');
  const ctx = canvas.getContext('2d');
  const w = img.clientWidth, h = img.clientHeight;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  if (!page || !page.preWidth || !w) return;
  const sx = w / page.preWidth, sy = h / page.preHeight;
  page.words.forEach((wd, i) => {
    if (!wd.bbox) return;
    ctx.strokeStyle = i === selectedIndex ? '#e6a800' : 'rgba(230,168,0,0.45)';
    ctx.lineWidth = i === selectedIndex ? 3 : 1;
    ctx.strokeRect(wd.bbox.x0 * sx, wd.bbox.y0 * sy, (wd.bbox.x1 - wd.bbox.x0) * sx, (wd.bbox.y1 - wd.bbox.y0) * sy);
  });
}

function selectWord(index) {
  const page = selectedPage(); if (!page) return;
  document.querySelectorAll('#textEditor .low').forEach((el) => el.classList.remove('selected-word'));
  const el = document.querySelector('#textEditor [data-word-index="' + index + '"]');
  if (el) { el.classList.add('selected-word'); el.scrollIntoView({ block: 'center' }); }
  drawOverlay(page, index);
}

function renderEditor(page) {
  const img = $('previewImg');
  $('previewEmpty').style.display = 'none';
  img.src = page.src;
  img.onload = () => drawOverlay(page, -1);
  const ed = $('textEditor');
  ed.innerHTML = '';
  if (!page.text) {
    ed.textContent = 'Текст ещё не распознан. Нажмите «Распознать всё» — или это черновик рукописи: сверяйте с фото вручную.';
    updateStats(page);
    return;
  }
  // Слова — отдельные span, неуверенные (<75%) подсвечены и кликабельны.
  page.words.forEach((wd, i) => {
    const span = document.createElement('span');
    span.textContent = wd.text;
    span.dataset.wordIndex = String(i);
    if (wd.confidence < 75) {
      span.className = 'low';
      span.title = 'Низкая уверенность: ' + Math.round(wd.confidence) + '% — сверьте с фото';
      span.addEventListener('click', () => selectWord(i));
    }
    ed.appendChild(span);
    ed.appendChild(document.createTextNode(' '));
  });
  updateStats(page);
}

// Клик по рамке на фото — найти слово по координатам и подсветить его в тексте.
$('overlayCanvas').addEventListener('click', (e) => {
  const page = selectedPage(); if (!page || !page.preWidth) return;
  const rect = e.target.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width * page.preWidth;
  const y = (e.clientY - rect.top) / rect.height * page.preHeight;
  const idx = page.words.findIndex((wd) => wd.bbox && x >= wd.bbox.x0 && x <= wd.bbox.x1 && y >= wd.bbox.y0 && y <= wd.bbox.y1);
  if (idx >= 0) selectWord(idx);
});

// Правка текста: не перестраиваем span при каждом нажатии (чтобы не терять курсор),
// просто обновляем статистику и сохраняем текст страницы.
$('textEditor').addEventListener('input', () => {
  const page = selectedPage(); if (!page) return;
  page.text = $('textEditor').innerText || '';
  updateStats(page);
});

// ---------- Экспорт: .txt, .docx, копирование ----------
function allText() {
  return pages.map((p) => '— ' + p.name + ' —\n' + (p.text || $('textEditor').innerText || '')).join('\n\n');
}
function downloadBlob(filename, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
function exportTxt() {
  const text = selectedPage() ? ($('textEditor').innerText || '') : allText();
  downloadBlob('archive-ocr.txt', new Blob([text], { type: 'text/plain;charset=utf-8' }));
  setStatus('Сохранён .txt');
}
async function exportDocx() {
  const text = selectedPage() ? ($('textEditor').innerText || '') : allText();
  if (typeof docx === 'undefined' || !docx.Document || !docx.Packer) {
    alert('Библиотека docx не загрузилась (нужен интернет для CDN). Сохраняю как .txt.');
    setStatus('docx недоступен — сохранён .txt вместо .docx');
    exportTxt();
    return;
  }
  const paras = text.split('\n').map((line) => new docx.Paragraph({ children: [new docx.TextRun(line || ' ')] }));
  const doc = new docx.Document({ sections: [{ children: paras }] });
  const blob = await docx.Packer.toBlob(doc);
  downloadBlob('archive-ocr.docx', blob);
  setStatus('Сохранён .docx');
}
async function copyText() {
  const text = $('textEditor').innerText || '';
  try { await navigator.clipboard.writeText(text); setStatus('Текст скопирован в буфер обмена.'); }
  catch (e) { $('textEditor').focus(); document.execCommand('selectAll'); setStatus('Выделите текст вручную (Ctrl+C) — доступ к буферу запрещён браузером.'); }
}

// ---------- Встроенный пример: страница-образец архивной справки на Canvas ----------
async function generateSample() {
  const canvas = document.createElement('canvas');
  canvas.width = 1200; canvas.height = 1700;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 1200, 1700);
  ctx.fillStyle = '#111'; ctx.textAlign = 'center';
  ctx.font = 'bold 48px serif'; ctx.fillText('СПРАВКА', 600, 130);
  ctx.font = '32px serif'; ctx.textAlign = 'left';
  const lines = SAMPLE_TEXT.split('\n').slice(1);
  lines.forEach((line, i) => ctx.fillText(line, 90, 230 + i * 62));
  ctx.font = '28px serif'; ctx.fillText('Печать архива', 90, 1500);
  const page = addPage('Пример — справка', canvas.toDataURL('image/png'), SAMPLE_TEXT);
  setStatus('Добавлен пример для проверки. Нажмите «Распознать всё».');
  return page;
}

// ---------- Бенчмарк: подобрать быстрый режим для этого компьютера ----------
// Точность считаем по известному тексту образца: расстояние Левенштейна.
function normalizeText(s) { return String(s || '').toLowerCase().replace(/[^a-zа-яё0-9]+/gi, ' ').replace(/\s+/g, ' ').trim(); }
function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (!m) return n; if (!n) return m;
  let prev = new Array(n + 1), cur = new Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    const t = prev; prev = cur; cur = t;
  }
  return prev[n];
}
function accuracyPct(recognized, expected) {
  const a = normalizeText(recognized), b = normalizeText(expected);
  if (!b) return 0;
  return Math.max(0, Math.round((1 - levenshtein(a, b) / Math.max(a.length, b.length)) * 100));
}

function benchRow(name, timeMs, accuracy, confidence, isBest) {
  const tr = document.createElement('tr');
  if (isBest) tr.className = 'best';
  // accuracy === null — замер без распознавания (экспресс-тест предобработки): точность неприменима.
  tr.innerHTML = '<td>' + esc(name) + (isBest ? ' — лучший' : '') + '</td><td>' + (timeMs / 1000).toFixed(1) + ' сек</td><td>' + (accuracy === null ? '—' : accuracy + '%') + '</td><td>' + (confidence === null ? '—' : Math.round(confidence) + '%') + '</td>';
  $('benchBody').appendChild(tr);
}

// Экспресс-тест (~2 сек): только предобработка, без движка — работает полностью офлайн.
async function expressTest() {
  $('benchStatus').textContent = 'Экспресс-тест: меряю предобработку…';
  const sample = pages.find((p) => p.knownText) || await generateSample();
  const sizes = [800, 1200, 1600, 1800, 2000];
  const times = [];
  for (const size of sizes) {
    const t0 = performance.now();
    await preprocessImage(sample.src, { maxSide: size, binarize: true, grayscale: true, contrast: 1.2, rotateDeg: 0 });
    times.push(Math.round(performance.now() - t0));
    await sleep(50);
  }
  $('benchTable').hidden = false; $('benchBody').innerHTML = '';
  sizes.forEach((size, i) => benchRow('Предобработка ' + size + ' px', times[i], null, null, i === 0));
  const avg = Math.round(times.reduce((a, b) => a + b, 0) / times.length);
  $('benchStatus').textContent = 'Экспресс-тест готов: предобработка одной страницы — в среднем ' + avg + ' мс. ' +
    (avg < 150 ? 'Компьютер быстрый для предобработки — можно брать обычный режим.' : 'Предобработка небыстрая — начните с экономного режима (1200 px).');
  $('benchRecommendation').hidden = true; $('applyBenchBtn').hidden = true;
}

// Полный бенчмарк: 3 прогона по одному, с паузами и кнопкой остановки.
async function fullBenchmark() {
  benchStopRequested = false;
  $('benchStopBtn').disabled = false; $('fullBenchBtn').disabled = true; $('expressBtn').disabled = true;
  $('benchTable').hidden = false; $('benchBody').innerHTML = '';
  $('benchRecommendation').hidden = true; $('applyBenchBtn').hidden = true;
  try {
    const sample = pages.find((p) => p.knownText) || await generateSample();
    // Проверяем движок заранее — если CDN недоступен, честно говорим и предлагаем экспресс-тест.
    try { await getWorker($('langSelect').value || 'rus'); }
    catch (err) {
      $('benchStatus').textContent = 'Полный бенчмарк невозможен: ' + err.message + ' Запустите экспресс-тест — он работает без движка.';
      return;
    }
    const configs = [
      { name: 'Экономный 1200 px + бинаризация', settings: { maxSide: 1200, binarize: true, allowUpscale: false }, quality: 'economy', binarize: true },
      { name: 'Обычный 1800 px + бинаризация', settings: { maxSide: 1800, binarize: true, allowUpscale: false }, quality: 'normal', binarize: true },
      { name: 'Без бинаризации 1800 px', settings: { maxSide: 1800, binarize: false, allowUpscale: false }, quality: 'normal', binarize: false },
    ];
    const results = [];
    for (const cfg of configs) {
      if (benchStopRequested) { $('benchStatus').textContent = 'Бенчмарк остановлен пользователем.'; break; }
      $('benchStatus').textContent = 'Бенчмарк: ' + cfg.name + ' …';
      const t0 = performance.now();
      const canvas = await preprocessImage(sample.src, { ...cfg.settings, grayscale: true, contrast: 1.2, rotateDeg: 0 });
      const w = await getWorker($('langSelect').value || 'rus');
      const res = await w.recognize(canvas, {}, { blocks: true });
      const timeMs = Math.round(performance.now() - t0);
      const acc = accuracyPct(res.data.text || '', SAMPLE_TEXT);
      const conf = typeof res.data.confidence === 'number' ? res.data.confidence : null;
      results.push({ ...cfg, timeMs, acc, conf });
      benchRow(cfg.name, timeMs, acc, conf, false);
      await sleep(600); // пауза между прогонами — не перегружаем слабый ПК
    }
    if (!results.length) return;
    const bestAcc = Math.max(...results.map((r) => r.acc));
    // Самый быстрый среди тех, чья точность не хуже лучшей более чем на 5 пунктов.
    const candidates = results.filter((r) => r.acc >= bestAcc - 5);
    const best = candidates.reduce((a, b) => (a.timeMs <= b.timeMs ? a : b));
    // Подсветить лучший ряд.
    [...$('benchBody').children].forEach((tr, i) => { if (results[i] === best) tr.classList.add('best'); });
    benchRecommendation = best;
    $('benchRecommendation').hidden = false;
    $('benchRecommendation').textContent = 'Рекомендация: ' + best.name + ' — ' + (best.timeMs / 1000).toFixed(1) + ' сек, точность ' + best.acc + '%. ' +
      'Это самый быстрый режим без заметной потери точности на этом компьютере.';
    $('applyBenchBtn').hidden = false;
    $('benchStatus').textContent = 'Полный бенчмарк готов: ' + results.length + ' прогона. Нажмите «Применить», чтобы включить рекомендованный режим.';
  } catch (err) {
    $('benchStatus').textContent = 'Бенчмарк прерван: ' + (err && err.message ? err.message : err) + ' Попробуйте экспресс-тест.';
  } finally {
    $('benchStopBtn').disabled = true; $('fullBenchBtn').disabled = false; $('expressBtn').disabled = false;
  }
}

function applyBenchmark() {
  if (!benchRecommendation) return;
  document.querySelector('input[name="quality"][value="' + benchRecommendation.quality + '"]').checked = true;
  $('binarizeCheck').checked = benchRecommendation.binarize;
  setStatus('Применён рекомендованный режим: ' + benchRecommendation.name + '. Настройки — в панели выше.');
  $('benchStatus').textContent = 'Применено: ' + benchRecommendation.name + '.';
}

// ---------- Привязка событий ----------
function init() {
  if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';
  }
  const drop = $('dropZone'), input = $('fileInput');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
  drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
  document.addEventListener('paste', (e) => {
    const items = (e.clipboardData || {}).items || [];
    const files = [];
    for (const it of items) if (it.type && it.type.indexOf('image/') === 0) { const f = it.getAsFile(); if (f) files.push(f); }
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  $('sampleBtn').addEventListener('click', generateSample);
  $('recognizeAllBtn').addEventListener('click', recognizeAll);
  $('stopBtn').addEventListener('click', () => { stopRequested = true; });
  $('copyBtn').addEventListener('click', copyText);
  $('exportTxtBtn').addEventListener('click', exportTxt);
  $('exportDocxBtn').addEventListener('click', exportDocx);
  $('expressBtn').addEventListener('click', expressTest);
  $('fullBenchBtn').addEventListener('click', fullBenchmark);
  $('benchStopBtn').addEventListener('click', () => { benchStopRequested = true; });
  $('applyBenchBtn').addEventListener('click', applyBenchmark);
  $('contrastRange').addEventListener('input', (e) => { $('contrastValue').textContent = e.target.value; });
  $('rotateRange').addEventListener('input', (e) => { $('rotateValue').textContent = e.target.value; });
  $('transcribeMode').addEventListener('change', (e) => { $('transcribeWarning').hidden = !e.target.checked; });
  window.addEventListener('resize', () => { const p = selectedPage(); if (p) drawOverlay(p, -1); });
}
document.addEventListener('DOMContentLoaded', init);
