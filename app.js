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
function currentEngine() {
  const checked = document.querySelector('input[name="engine"]:checked');
  return checked ? checked.value : 'tesseract';
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ---------- Загрузка файлов: выбор, drag-and-drop, вставка из буфера ----------
function addPage(name, src, knownText) {
  const page = { id: ++pageCounter, name, src, knownText: knownText || null, text: '', words: [], confidence: null, timeMs: null, done: false, engine: null };
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

// ---------- Режим «Рукопись»: PP-OCRv6 (ONNX, onnxruntime-web, всё локально) ----------
// Детектор строк: PP-OCRv6 tiny det (PaddlePaddle, ~1,7 МБ).
// Распознаватель строк: PP-OCRv6 tiny/medium rec Cyrillic (Metrika Labs, ~2,4 / ~60 МБ).
// Проверено Node-тестом 2026-11-11: выход det — карта вероятностей в ПОЛНОМ разрешении
// входа (а не 1/4, как у классического DBNet), выход rec — уже ВЕРОЯТНОСТИ
// [1, T, 852] (сумма по строке = 1.0, softmax НЕ применять): 0 = blank,
// 1..850 = словарь ниже, 851 = пробел (проверено: встречается между словами).
// Модели в репозиторий НЕ кладутся: скачиваются с HuggingFace при первом использовании
// и кэшируются в IndexedDB ('archive-ocr-models').
const PPOCR_DICT = "!\"#$%&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_`abcdefghijklmnopqrstuvwxyz©‥{}\\|@^~÷∕∙⋅·±∓∩∪□←↔⇒⇐⇔∀∃∄∴∵∝∞⊥∟∠∡∢′″∥⊾⊿∂∫∬∭∮∯∰∑∏√∛∜∱∲∳∶∷∼®℉Ω℧Å⌀ℏ⅀⍺⍵¢€£¥₿ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻ➀➁➂➃➄➅➆➇➈➉➊➋➌➍➎➏➐➑➒➓❶❷❸❹❺❻❼❽❾❿①②③④⑤⑥⑦⑧⑨⑩●▶𝑢︽–﹥𝜓•∋ƒ०⬆Ạ◀▫︾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ¡¤¦§¨ª«¬¯°²³´µ¶¸¹º»¼½¾¿×‐‑‒—―‖‗‘’‚‛“”„‟†‡‣․…‧‰‴‵‶‷‸‹›※‼‽‾₤₡₹−∖∗≈≠≡≤≥⊂⊃↑→↓↕™Ω℮∆✓✗✘▪◼✔✕☑☒№₽₴ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψωЀЁЂЃЄЅІЇЈЉЊЋЌЍЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяѐёђѓєѕіїјљњћќѝўџѠѡѢѣѤѥѦѧѨѩѪѫѬѭѮѯѰѱѲѳѴѵѶѷѸѹѺѻѼѽѾѿҀҁҊҋҌҍҎҏҐґҒғҔҕҖҗҘҙҚқҜҝҞҟҠҡҢңҦҧҨҩҪҫҬҭҮүҰұҲҳҶҷҸҹҺһҼҽҾҿӀӁӂӃӄӅӆӇӈӉӊӋӌӍӎӏӐӑӒӓӖӗӘәӚӛӜӝӞӟӠӡӢӣӤӥӦӧӨөӪӫӬӭӮӯӰӱӲӳӴӵӶӷӸӹӺӻӼӽӾӿԀԁԂԃԄԅԆԇԈԉԊԋԌԍԎԏԐԑԒԓԔԕԖԗԘԙԚԛԜԝԞԟԠԡԢԣԤԥԦԧԨԩԪԫԬԭԮԯⷠⷡⷢⷣⷤⷥⷦⷧⷨⷩⷪⷫⷬⷭⷮⷯⷰⷱⷲⷳⷴⷵⷶⷷⷸⷹⷺⷻⷼⷽⷾⷿꙀꙁꙂꙃꙄꙅꙆꙇꙈꙉꙊꙋꙌꙍꙎꙏꙐꙑꙒꙓꙔꙕꙖꙗꙘꙙꙚꙛꙜꙝꙞꙟꙠꙡꙢꙣꙤꙥꙦꙧꙨꙩꙪꙫꙬꙭꙮꙴꙵꙶꙷꙸꙹꙺꙻꚀꚁꚂꚃꚄꚅꚆꚇꚈꚉꚊꚋꚌꚍꚎꚏꚐꚑꚒꚓꚔꚕꚖꚗꚘꚙꚚꚛꚜꚝꚞꚟ";
const PPOCR_DICT_CHARS = Array.from(PPOCR_DICT); // именно Array.from: в словаре есть 2 символа вне BMP
const PPOCR_URLS = {
  ort: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.js',
  ortDist: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/',
  det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx',
  detSmall: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx',
  recTiny: 'https://huggingface.co/metrika-labs/PP-OCRv6_tiny_rec_cyrillic_onnx/resolve/main/inference.onnx',
  recMedium: 'https://huggingface.co/metrika-labs/PP-OCRv6_medium_rec_cyrillic_onnx/resolve/main/inference.onnx',
};
let ortLoading = null;           // промис загрузки onnxruntime-web (лениво, один раз)
let ppocrSessions = { det: null, detKey: null, recTiny: null, recMedium: null };

// ---------- Лимит памяти: 1–8 ГБ, шаг 1 ГБ (настройка на сайте) ----------
// Чем больше лимит, тем тяжелее модель. 1 ГБ — PP-OCRv6 Tiny (замерено),
// 2 ГБ — Small det + Medium rec, 3–8 ГБ — TrOCR (ПОКА НЕ ПОДКЛЮЧЁН:
// нужна переконвертация декодера в merged-формат с KV-кэшем; ручной путь без
// кэша даёт 5–20 мин/страницу и не годится). Размеры скачивания — десятичные МБ.
// RAM вкладки — оценка с учётом рантайма и активаций; для fp32 есть риск упереться
// в лимит кучи WASM ~4 ГБ и уронить вкладку (по данным исследования TrOCR).
const RAM_TIERS = [
  { gb: 1, key: 'tiny', label: 'PP-OCRv6 Tiny', det: 'tiny', rec: 'tiny', download: '~4 МБ моделей + ~14 МБ рантайм', ram: '~0,3–0,5 ГБ', status: 'ready', note: 'Замерено на курсиве: 46–47% точности, ~2 сек/стр.' },
  { gb: 2, key: 'medium', label: 'PP-OCRv6 Small det + Medium rec', det: 'small', rec: 'medium', download: '~72 МБ моделей + ~14 МБ рантайм', ram: '~1–1,5 ГБ', status: 'ready', note: 'Честно: на курсиве Medium замеренно хуже Tiny (27% против 46%). Medium лучше для аккуратного почерка и печатного текста.' },
  { gb: 3, key: 'trocr-q4', label: 'TrOCR q4f16', det: null, rec: null, download: '~294 МБ', ram: '~0,7–1 ГБ', status: 'soon', note: 'Скоро: режим ещё не подключён, нужна переконвертация декодера TrOCR.' },
  { gb: 4, key: 'trocr-int8', label: 'TrOCR int8 (рекомендуемый максимум)', det: null, rec: null, download: '~388 МБ', ram: '~1–1,5 ГБ', status: 'soon', note: 'Скоро: режим ещё не подключён, нужна переконвертация декодера TrOCR.' },
  { gb: 5, key: 'trocr-int8-beam', label: 'TrOCR int8 + beam search', det: null, rec: null, download: '~388 МБ', ram: '~2 ГБ', status: 'soon', note: 'Скоро: режим ещё не подключён, нужна переконвертация декодера TrOCR.' },
  { gb: 6, key: 'trocr-fp16', label: 'TrOCR fp16', det: null, rec: null, download: '~770 МБ', ram: '~2–3 ГБ', status: 'soon', note: 'Скоро: режим ещё не подключён, нужна переконвертация декодера TrOCR.' },
  { gb: 7, key: 'trocr-fp32', label: 'TrOCR fp32', det: null, rec: null, download: '~1,5 ГБ', ram: '~3,5–4,5 ГБ', status: 'soon', note: 'Скоро: режим ещё не подключён. Внимание: fp32 рискует упереться в лимит кучи WASM ~4 ГБ и уронить вкладку.' },
  { gb: 8, key: 'trocr-fp32-max', label: 'TrOCR fp32 + запас', det: null, rec: null, download: '~1,5 ГБ', ram: '~4–5 ГБ', status: 'soon', note: 'Скоро: режим ещё не подключён. Внимание: fp32 рискует упереться в лимит кучи WASM ~4 ГБ и уронить вкладку.' },
];

// TrOCR (Mievst/trocr-base-handwritten-ru-ONNX) — только константа с URL для
// будущего подключения. Кода TrOCR здесь намеренно нет (см. комментарий выше).
// Все URL проверены (HTTP 302 на CDN HuggingFace — файл существует).
const TROCR_URLS = {
  repo: 'https://huggingface.co/Mievst/trocr-base-handwritten-ru-ONNX',
  encoderInt8: 'https://huggingface.co/Mievst/trocr-base-handwritten-ru-ONNX/resolve/main/onnx/encoder_model_int8.onnx', // 87 942 932 Б
  decoderInt8: 'https://huggingface.co/Mievst/trocr-base-handwritten-ru-ONNX/resolve/main/onnx/decoder_model_int8.onnx', // 300 116 959 Б
  encoderQ4: 'https://huggingface.co/Mievst/trocr-base-handwritten-ru-ONNX/resolve/main/onnx/encoder_model_q4f16.onnx', // 50 218 204 Б
  decoderQ4: 'https://huggingface.co/Mievst/trocr-base-handwritten-ru-ONNX/resolve/main/onnx/decoder_model_q4f16.onnx', // 243 664 494 Б
};

const RAM_LIMIT_KEY = 'archiveOcrRamLimitGb';
const RAM_LIMIT_DEFAULT = 2;

function currentRamLimit() {
  const el = $('ramRange');
  const v = el ? parseInt(el.value, 10) : RAM_LIMIT_DEFAULT;
  return Math.min(8, Math.max(1, isNaN(v) ? RAM_LIMIT_DEFAULT : v));
}
function ramTier() {
  const gb = currentRamLimit();
  return RAM_TIERS.find((t) => t.gb === gb) || RAM_TIERS[1];
}
function ppocrAccuracy() {
  const checked = document.querySelector('input[name="ppocrAccuracy"]:checked');
  return checked ? checked.value : 'tiny';
}

// Эффективная конфигурация PP-OCR для текущего лимита и выбранной точности.
// 1 ГБ — всегда Tiny; 2 ГБ — Tiny или Small det + Medium rec (по радио «Точность»).
function effectivePpocrConfig() {
  const tier = ramTier();
  if (tier.status === 'soon') return { tier, status: 'soon', det: null, rec: null };
  const acc = ppocrAccuracy();
  return {
    tier, status: 'ready',
    det: (tier.gb >= 2 && acc === 'medium') ? 'small' : 'tiny',
    rec: acc,
  };
}

// Выгрузка сессий PP-OCR, которые не нужны новому лимиту (освободить память).
// ort.InferenceSession.release есть в onnxruntime-web; вызов — в try/catch,
// отсутствие метода не ломает работу.
async function releasePpocrSession(s) {
  if (!s) return;
  try { if (typeof s.release === 'function') await s.release(); } catch (e) { /* игнорируем */ }
}
async function releasePpocrSessionsForTier() {
  const cfg = effectivePpocrConfig();
  if (cfg.status === 'soon') {
    await releasePpocrSession(ppocrSessions.det);
    ppocrSessions.det = null; ppocrSessions.detKey = null;
    await releasePpocrSession(ppocrSessions.recTiny); ppocrSessions.recTiny = null;
    await releasePpocrSession(ppocrSessions.recMedium); ppocrSessions.recMedium = null;
    return;
  }
  if (ppocrSessions.det && ppocrSessions.detKey !== cfg.det) {
    await releasePpocrSession(ppocrSessions.det);
    ppocrSessions.det = null; ppocrSessions.detKey = null;
  }
  if (cfg.rec === 'tiny' && ppocrSessions.recMedium) {
    await releasePpocrSession(ppocrSessions.recMedium); ppocrSessions.recMedium = null;
  }
  if (cfg.rec === 'medium' && ppocrSessions.recTiny) {
    await releasePpocrSession(ppocrSessions.recTiny); ppocrSessions.recTiny = null;
  }
}

// Обновить подпись лимита памяти и состояние радио «Точность»:
// Medium недоступен при лимите 1 ГБ (нужно минимум 2 ГБ).
function updateRamUi() {
  const tier = ramTier();
  const cfg = effectivePpocrConfig();
  $('ramValue').textContent = tier.gb + ' ГБ';
  const info = $('ramTierInfo');
  if (cfg.status === 'soon') {
    info.textContent = tier.gb + ' ГБ → ' + tier.label + '. Скачивание при первом запуске: ' + tier.download + '. RAM вкладки: ' + tier.ram + '. ' + tier.note;
  } else {
    const detName = cfg.det === 'small' ? 'Small' : 'Tiny';
    const recName = cfg.rec === 'medium' ? 'Medium' : 'Tiny';
    info.textContent = tier.gb + ' ГБ → PP-OCRv6: детектор ' + detName + ' + распознаватель ' + recName + '. Скачивание при первом запуске: ' + tier.download + '. RAM вкладки: ' + tier.ram + '. ' + tier.note;
  }
  $('trocrSoonWarning').hidden = cfg.status !== 'soon';
  const mediumRadio = document.querySelector('input[name="ppocrAccuracy"][value="medium"]');
  const tinyRadio = document.querySelector('input[name="ppocrAccuracy"][value="tiny"]');
  if (mediumRadio) {
    mediumRadio.disabled = tier.gb < 2;
    if (tier.gb < 2 && tinyRadio) tinyRadio.checked = true;
  }
}

async function onRamLimitChange() {
  try { localStorage.setItem(RAM_LIMIT_KEY, String(currentRamLimit())); } catch (e) { /* localStorage может быть недоступен */ }
  updateRamUi();
  try { await releasePpocrSessionsForTier(); } catch (e) { /* выгрузка необязательна */ }
}

// Цветной кадр для PP-OCR: ресайз, поворот, контраст — но БЕЗ бинаризации и без
// перевода в серый (синие чернила и курсив от бинаризации страдают).
async function prepareColorCanvas(src, opts) {
  const img = await loadImage(src);
  const maxSide = opts.maxSide || 1800;
  const longest = Math.max(img.naturalWidth, img.naturalHeight);
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
  if (opts.contrast && opts.contrast !== 1) {
    const imageData = ctx.getImageData(0, 0, w, h);
    const d = imageData.data, c = opts.contrast;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = Math.max(0, Math.min(255, (d[i] - 128) * c + 128));
      d[i + 1] = Math.max(0, Math.min(255, (d[i + 1] - 128) * c + 128));
      d[i + 2] = Math.max(0, Math.min(255, (d[i + 2] - 128) * c + 128));
    }
    ctx.putImageData(imageData, 0, 0);
  }
  return canvas;
}

function resizeCanvasTo(srcCanvas, maxSide) {
  const longest = Math.max(srcCanvas.width, srcCanvas.height);
  if (longest <= maxSide) return srcCanvas;
  const scale = maxSide / longest;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(srcCanvas.width * scale));
  canvas.height = Math.max(1, Math.round(srcCanvas.height * scale));
  canvas.getContext('2d').drawImage(srcCanvas, 0, 0, canvas.width, canvas.height);
  return canvas;
}

// Ленивая загрузка onnxruntime-web. Один поток WASM: на GitHub Pages нет COOP/COEP,
// SharedArrayBuffer недоступен, многопоточность включить нельзя.
async function loadOrt() {
  if (typeof ort !== 'undefined') return ort;
  if (ortLoading) return ortLoading;
  ortLoading = new Promise((resolve, reject) => {
    setStatus('Загружаю движок рукописей (onnxruntime-web)…');
    const s = document.createElement('script');
    s.src = PPOCR_URLS.ort;
    s.onload = () => {
      try {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = PPOCR_URLS.ortDist;
        resolve(ort);
      } catch (e) { reject(e); }
    };
    s.onerror = () => reject(new Error('Не удалось загрузить onnxruntime-web с CDN jsDelivr. Проверьте интернет и обновите страницу.'));
    document.head.appendChild(s);
  });
  return ortLoading;
}

function ppocrIdb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('archive-ocr-models', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('models');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(new Error('IndexedDB недоступен в этом браузере — модели придётся скачивать при каждом запуске.'));
  });
}

// Скачать модель (с прогрессом в статусе) или взять из кэша IndexedDB.
async function ppocrCached(key, url, label) {
  let db = null;
  try {
    db = await ppocrIdb();
    const cached = await new Promise((resolve) => {
      const tx = db.transaction('models', 'readonly');
      const req = tx.objectStore('models').get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
    if (cached) return cached;
  } catch (e) { /* без кэша — просто скачаем */ }
  setStatus('Скачиваю модель рукописей: ' + label + '…');
  setProgress(0);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('Не удалось скачать модель рукописей (' + label + ', HTTP ' + resp.status + '). Проверьте интернет и попробуйте ещё раз.');
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  let buf;
  if (resp.body && resp.body.getReader && total > 0) {
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      setStatus('Скачиваю модель рукописей: ' + label + ' — ' + Math.round((received / total) * 100) + '%');
      setProgress(Math.round((received / total) * 100));
    }
    const all = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    buf = all.buffer;
  } else {
    buf = await resp.arrayBuffer();
  }
  if (db) {
    try {
      const tx = db.transaction('models', 'readwrite');
      tx.objectStore('models').put(buf, key);
    } catch (e) { /* кэш необязателен */ }
  }
  setProgress(100);
  return buf;
}

async function getPpocrSessions() {
  if (PPOCR_DICT_CHARS.length !== 850) throw new Error('Внутренняя ошибка: словарь распознавателя повреждён (ожидалось 850 символов).');
  await loadOrt();
  const cfg = effectivePpocrConfig();
  const acc = cfg.rec; // 'tiny' | 'medium' (для soon сюда не доходим — см. recognizePpocrPage)
  const detKey = cfg.det; // 'tiny' | 'small'
  if (!ppocrSessions.det || ppocrSessions.detKey !== detKey) {
    if (ppocrSessions.det) { await releasePpocrSession(ppocrSessions.det); ppocrSessions.det = null; }
    const detUrl = detKey === 'small' ? PPOCR_URLS.detSmall : PPOCR_URLS.det;
    const detLabel = detKey === 'small' ? 'детектор строк Small (~9,9 МБ)' : 'детектор строк (~1,7 МБ)';
    const buf = await ppocrCached(detKey === 'small' ? 'detSmall' : 'det', detUrl, detLabel);
    setStatus('Загружаю детектор строк…');
    ppocrSessions.det = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    ppocrSessions.detKey = detKey;
  }
  const recKey = acc === 'medium' ? 'recMedium' : 'recTiny';
  if (!ppocrSessions[recKey]) {
    const buf = await ppocrCached(recKey, acc === 'medium' ? PPOCR_URLS.recMedium : PPOCR_URLS.recTiny,
      acc === 'medium' ? 'распознаватель Medium (~60 МБ)' : 'распознаватель Tiny (~2,4 МБ)');
    setStatus('Загружаю распознаватель ' + (acc === 'medium' ? 'Medium' : 'Tiny') + '…');
    ppocrSessions[recKey] = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
  }
  return { det: ppocrSessions.det, rec: ppocrSessions[recKey] };
}

// Группировка рамок в строки: сверху вниз, внутри строки — слева направо.
function sortLineBoxes(boxes) {
  const sorted = boxes.slice().sort((a, b) => a.y0 - b.y0 || a.x0 - b.x0);
  const rows = [];
  for (const b of sorted) {
    const cy = (b.y0 + b.y1) / 2;
    let row = rows.find((r) => {
      const rcy = (r[0].y0 + r[0].y1) / 2;
      return Math.abs(cy - rcy) < Math.max(10, (b.y1 - b.y0) * 0.6);
    });
    if (!row) { row = []; rows.push(row); }
    row.push(b);
  }
  rows.sort((a, b) => a[0].y0 - b[0].y0);
  return rows.flatMap((r) => r.sort((a, b) => a.x0 - b.x0));
}

// Детекция DB (упрощённая, но честная): порог 0.2 по карте вероятностей,
// компоненты связности, отсев по средней вероятности 0.45, небольшое расширение рамки.
// Карта — в полном разрешении входа детектора, масштабировать не нужно.
function ppocrDetectDb(prob, mw, mh) {
  const THRESH = 0.2, BOX_THRESH = 0.45;
  const bin = new Uint8Array(mw * mh);
  for (let i = 0; i < prob.length; i++) bin[i] = prob[i] >= THRESH ? 1 : 0;
  const label = new Int32Array(mw * mh).fill(-1);
  const boxes = [];
  const stack = [];
  for (let s = 0; s < bin.length; s++) {
    if (!bin[s] || label[s] !== -1) continue;
    // label — только отметка «посещено» (значение не важно, проверяем !== -1)
    let minX = mw, minY = mh, maxX = -1, maxY = -1, area = 0, sumProb = 0;
    stack.push(s); label[s] = 1;
    while (stack.length) {
      const p = stack.pop();
      const x = p % mw, y = (p / mw) | 0;
      area++; sumProb += prob[p];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      const nb = [p - 1, p + 1, p - mw, p + mw];
      for (const q of nb) {
        if (q < 0 || q >= bin.length) continue;
        if (x === 0 && q === p - 1) continue;
        if (x === mw - 1 && q === p + 1) continue;
        if (bin[q] && label[q] === -1) { label[q] = 1; stack.push(q); }
      }
    }
    const meanProb = area ? sumProb / area : 0;
    if (area < 3 || meanProb < BOX_THRESH) continue;
    // Небольшое расширение: классический unclip 1.4 для широких строк даёт ~45 px
    // на сторону, и соседние строки пересекаются (проверено Node-тестом).
    const h = maxY - minY + 1;
    const d = Math.min(6, Math.round(h * 0.15));
    boxes.push({
      x0: Math.max(0, minX - d), y0: Math.max(0, minY - d),
      x1: Math.min(mw, maxX + 1 + d), y1: Math.min(mh, maxY + 1 + d),
      score: meanProb,
    });
  }
  return sortLineBoxes(boxes);
}

// Детекция «по строкам»: бинаризация Отсу + горизонтальный профиль.
// Ряд считается «с чернилами», если тёмных пикселей >= 3% ширины; разрывы до 4 px склеиваются.
// Для ровных страниц — быстро и без нейросети; на плотном курсиве строки могут слипаться.
function ppocrDetectLines(gray, w, h) {
  const thr = otsuThreshold(gray);
  const profile = new Array(h).fill(0);
  for (let y = 0; y < h; y++) {
    let c = 0;
    for (let x = 0; x < w; x++) if (gray[y * w + x] < thr) c++;
    profile[y] = c;
  }
  const minInk = Math.max(2, Math.round(w * 0.03));
  const boxes = [];
  let y = 0;
  while (y < h) {
    if (profile[y] < minInk) { y++; continue; }
    let y0 = y;
    while (y < h && profile[y] >= minInk) y++;
    let y1 = y;
    while (y < h) {
      let gap = 0;
      while (y < h && profile[y] < minInk && gap < 4) { y++; gap++; }
      if (gap < 4 && y < h && profile[y] >= minInk) { while (y < h && profile[y] >= minInk) y++; y1 = y; }
      else break;
    }
    if (y1 - y0 >= 6) boxes.push({ x0: 0, y0: Math.max(0, y0 - 2), x1: w, y1: Math.min(h, y1 + 2), score: 1 });
  }
  return sortLineBoxes(boxes);
}

// CTC-декодирование: argmax -> склеить повторы -> убрать blank (0).
// Выход модели — уже вероятности, поэтому уверенность строки — среднее максимумов
// по распознанным символам, без дополнительного softmax.
function ppocrCtcDecode(data, T, classes) {
  let prev = -1, text = '', probs = [];
  for (let t = 0; t < T; t++) {
    const off = t * classes;
    let best = 0, bestVal = -Infinity;
    for (let c = 0; c < classes; c++) { const v = data[off + c]; if (v > bestVal) { bestVal = v; best = c; } }
    if (best !== 0 && best !== prev) {
      if (best === 851 && classes === 852) text += ' '; // 851-й класс — пробел
      else if (best >= 1 && best <= 850) text += PPOCR_DICT_CHARS[best - 1] || '';
      probs.push(bestVal);
    }
    prev = best;
  }
  return { text, confidence: probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0 };
}

// Распознавание страницы в режиме «Рукопись»: детекция -> кропы строк -> rec построчно.
async function recognizePpocrPage(page) {
  const tier = ramTier();
  if (tier.status === 'soon') {
    // TrOCR пока не подключён: честно отказываемся, не ломая приложение
    // и не помечая страницу распознанной (done остаётся false).
    throw new Error('Режим «' + tier.label + '» (лимит ' + tier.gb + ' ГБ) ещё не подключён: для TrOCR нужна переконвертация декодера в формат с KV-кэшем. Выберите лимит 1–2 ГБ (PP-OCRv6) или дождитесь обновления.');
  }
  const started = performance.now();
  const settings = readSettings();
  setStatus('Готовлю фото: ' + page.name + ' …');
  const colorCanvas = await prepareColorCanvas(page.src, settings);
  page.preWidth = colorCanvas.width; page.preHeight = colorCanvas.height;
  const sessions = await getPpocrSessions();
  const detCanvas = resizeCanvasTo(colorCanvas, 960);
  const dw = detCanvas.width, dh = detCanvas.height;
  const dctx = detCanvas.getContext('2d', { willReadFrequently: true });
  const detImg = dctx.getImageData(0, 0, dw, dh);
  const detectMode = (document.querySelector('input[name="ppocrDetect"]:checked') || {}).value || 'auto';
  let boxes;
  if (detectMode === 'lines') {
    const gray = new Uint8Array(dw * dh);
    const d = detImg.data;
    for (let i = 0, p = 0; i < d.length; i += 4, p++) gray[p] = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    boxes = ppocrDetectLines(gray, dw, dh);
  } else {
    setStatus('Ищу строки: ' + page.name + ' …');
    const input = new Float32Array(3 * dh * dw);
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    const d = detImg.data;
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      const s = (y * dw + x) * 4;
      const r = d[s] / 255, g = d[s + 1] / 255, b = d[s + 2] / 255;
      input[(0 * dh + y) * dw + x] = (b - mean[2]) / std[2];
      input[(1 * dh + y) * dw + x] = (g - mean[1]) / std[1];
      input[(2 * dh + y) * dw + x] = (r - mean[0]) / std[0];
    }
    const out = await sessions.det.run({ [sessions.det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, dh, dw]) });
    const t = out[sessions.det.outputNames[0]];
    boxes = ppocrDetectDb(t.data, t.dims[3], t.dims[2]);
  }
  if (!boxes.length) {
    page.text = ''; page.words = []; page.confidence = 0;
    page.timeMs = Math.round(performance.now() - started);
    page.done = true; page.engine = 'ppocr';
    setStatus('Строк не найдено: ' + page.name + '. Попробуйте другой режим детекции или проверьте фото.');
    return page;
  }
  const sx = colorCanvas.width / dw, sy = colorCanvas.height / dh;
  const lines = [];
  for (let i = 0; i < boxes.length; i++) {
    if (stopRequested) break;
    const b = boxes[i];
    setStatus('Распознаю строку ' + (i + 1) + ' из ' + boxes.length + ': ' + page.name);
    setProgress(Math.round(((i + 1) / boxes.length) * 100));
    const x0 = Math.max(0, Math.round(b.x0 * sx)), y0 = Math.max(0, Math.round(b.y0 * sy));
    const x1 = Math.min(colorCanvas.width, Math.round(b.x1 * sx)), y1 = Math.min(colorCanvas.height, Math.round(b.y1 * sy));
    const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
    const rw = Math.min(640, Math.max(8, Math.round((cw / ch) * 48)));
    const lineCanvas = document.createElement('canvas');
    lineCanvas.width = rw; lineCanvas.height = 48;
    lineCanvas.getContext('2d').drawImage(colorCanvas, x0, y0, cw, ch, 0, 0, rw, 48);
    const lctx = lineCanvas.getContext('2d', { willReadFrequently: true });
    const ldata = lctx.getImageData(0, 0, rw, 48).data;
    const recInput = new Float32Array(3 * 48 * rw);
    for (let y = 0; y < 48; y++) for (let x = 0; x < rw; x++) {
      const s = (y * rw + x) * 4;
      recInput[(0 * 48 + y) * rw + x] = (ldata[s + 2] / 255 - 0.5) / 0.5; // B
      recInput[(1 * 48 + y) * rw + x] = (ldata[s + 1] / 255 - 0.5) / 0.5; // G
      recInput[(2 * 48 + y) * rw + x] = (ldata[s] / 255 - 0.5) / 0.5;     // R
    }
    const out = await sessions.rec.run({ [sessions.rec.inputNames[0]]: new ort.Tensor('float32', recInput, [1, 3, 48, rw]) });
    const t = out[sessions.rec.outputNames[0]];
    const decoded = ppocrCtcDecode(t.data, t.dims[1], t.dims[2]);
    lines.push({
      text: decoded.text, confidence: decoded.confidence,
      bbox: { x0, y0, x1, y1 },
    });
  }
  page.text = lines.map((l) => l.text).join('\n');
  // Слова для подсветки/оверлея — построчные: клик по строке работает как по слову.
  page.words = lines.map((l) => ({ text: l.text, confidence: l.confidence * 100, bbox: l.bbox }));
  page.confidence = lines.length ? (lines.reduce((a, l) => a + l.confidence, 0) / lines.length) * 100 : 0;
  page.timeMs = Math.round(performance.now() - started);
  page.done = true; page.engine = 'ppocr';
  return page;
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
  if (currentEngine() === 'ppocr') return recognizePpocrPage(page);
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
  page.engine = 'tesseract';
  return page;
}

async function recognizeAll() {
  const engine = currentEngine();
  // Повторно распознаём страницы, у которых движок отличается от текущего:
  // например, после переключения «Печать» -> «Рукопись» старые страницы прогоняются заново.
  const todo = pages.filter((p) => !p.done || p.engine !== engine);
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
    const engineTag = p.engine === 'ppocr' ? ', PP-OCR' : (p.engine === 'tesseract' ? ', Tesseract' : '');
    li.textContent = p.name + (p.done ? ' — готово (' + (p.confidence === null ? '—' : Math.round(p.confidence) + '%') + engineTag + ')' : ' — не распознано');
    if (p.id === selectedId) li.classList.add('selected');
    if (p.done) li.classList.add('done');
    li.addEventListener('click', () => selectPage(p.id));
    list.appendChild(li);
  });
  // «Распознать заново» — только для уже распознанной выбранной страницы.
  const sel = selectedPage();
  $('reRecognizeBtn').disabled = !(sel && sel.done);
}

function selectedPage() { return pages.find((p) => p.id === selectedId) || null; }

// Распознать выбранную страницу заново текущим движком
// (например, после смены настроек или переключения «Печать» <-> «Рукопись»).
async function reRecognizeSelected() {
  const page = selectedPage();
  if (!page) return;
  page.done = false; page.text = ''; page.words = []; page.confidence = null; page.timeMs = null; page.engine = null;
  $('reRecognizeBtn').disabled = true; $('recognizeAllBtn').disabled = true; $('stopBtn').disabled = false;
  try {
    await recognizePage(page);
    renderQueue();
    renderEditor(page);
    setStatus('Готово: ' + page.name + ' — ' + page.timeMs + ' мс, уверенность ' + (page.confidence === null ? '—' : Math.round(page.confidence) + '%'));
  } catch (err) {
    setStatus('Ошибка: ' + (err && err.message ? err.message : err));
  } finally {
    $('recognizeAllBtn').disabled = false; $('stopBtn').disabled = true;
    renderQueue();
  }
}

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
  // В режиме «Рукопись» слова — это целые строки, поэтому разделяем их переносами.
  const separator = page.engine === 'ppocr' ? '\n' : ' ';
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
    ed.appendChild(document.createTextNode(separator));
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

// ---------- PWA: service worker, установка, обновление ----------
// Регистрация — только если SW поддерживается и страница открыта по http(s),
// в try/catch: без SW приложение работает как обычный сайт (в т.ч. как file://
// и в однофайловом артефакте, где sw.js рядом нет).
let swRegistration = null;
let deferredInstallPrompt = null;

function registerServiceWorker() {
  try {
    if (!('serviceWorker' in navigator)) return;
    if (window.location.protocol === 'file:') return;
    navigator.serviceWorker.register('sw.js').then((reg) => {
      swRegistration = reg;
      reg.addEventListener('updatefound', () => {
        const nw = reg.installing;
        if (!nw) return;
        nw.addEventListener('statechange', () => {
          // Новый SW установлен, но страницей ещё управляет старый — предлагаем обновиться.
          if (nw.state === 'installed' && navigator.serviceWorker.controller) {
            $('updateBanner').hidden = false;
          }
        });
      });
    }).catch(() => { /* SW недоступен — приложение работает и без него */ });
  } catch (e) { /* ничего не ломаем */ }
}

function setupPwaInstall() {
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    $('installBtn').hidden = false;
  });
  $('installBtn').addEventListener('click', async () => {
    if (!deferredInstallPrompt) return;
    deferredInstallPrompt.prompt();
    try { await deferredInstallPrompt.userChoice; } catch (e) { /* игнорируем */ }
    deferredInstallPrompt = null;
    $('installBtn').hidden = true;
  });
  // iOS: своего beforeinstallprompt нет — показываем ручную подсказку.
  const isIos = /iphone|ipad|ipod/i.test(navigator.userAgent || '');
  if (isIos && !window.navigator.standalone) $('iosHint').hidden = false;
  $('updateBtn').addEventListener('click', () => {
    const waiting = swRegistration && swRegistration.waiting;
    if (waiting) waiting.postMessage({ type: 'SKIP_WAITING' });
    else window.location.reload();
  });
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('controllerchange', () => window.location.reload());
  }
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
  $('reRecognizeBtn').addEventListener('click', reRecognizeSelected);
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
  document.querySelectorAll('input[name="engine"]').forEach((r) => r.addEventListener('change', updateEngineUi));
  updateEngineUi();
  // Лимит памяти: восстановить из localStorage (по умолчанию 2 ГБ).
  try {
    const stored = parseInt(localStorage.getItem(RAM_LIMIT_KEY) || '', 10);
    if (stored >= 1 && stored <= 8) $('ramRange').value = String(stored);
  } catch (e) { /* localStorage может быть недоступен */ }
  updateRamUi();
  $('ramRange').addEventListener('input', () => { onRamLimitChange(); });
  document.querySelectorAll('input[name="ppocrAccuracy"]').forEach((r) => r.addEventListener('change', async () => {
    updateRamUi();
    try { await releasePpocrSessionsForTier(); } catch (e) { /* выгрузка необязательна */ }
  }));
  registerServiceWorker();
  setupPwaInstall();
  window.addEventListener('resize', () => { const p = selectedPage(); if (p) drawOverlay(p, -1); });
}
document.addEventListener('DOMContentLoaded', init);

// Показ/скрытие настроек в зависимости от выбранного движка.
function updateEngineUi() {
  const ppocr = currentEngine() === 'ppocr';
  $('ppocrSettings').hidden = !ppocr;
  $('ppocrWarning').hidden = !ppocr;
  $('tesseractSettings').style.display = ppocr ? 'none' : 'contents';
}
