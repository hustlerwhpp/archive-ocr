/* Архив OCR — service worker (PWA).
   Кэширует оболочку приложения и CDN-библиотеки, чтобы после первого запуска
   приложение открывалось и без интернета (установка: «Установить приложение»).
   Модели НЕ кэшируются здесь: они лежат в IndexedDB (база 'archive-ocr-models',
   см. app.js) — Cache Storage для файлов на сотни мегабайт не подходит.
   Версия кэша: при обновлении приложения меняйте CACHE_NAME на новое имя. */
'use strict';

const CACHE_NAME = 'archive-ocr-v1';

const SHELL_URLS = [
  '/',
 '/index.html', '/styles.css', '/app.js',
  '/manifest.webmanifest',
  '/icon-192.png', '/icon-512.png', '/icon-512-maskable.png',
];

const CDN_LIBS = [
  'https://cdn.jsdelivr.net/npm/tesseract.js@6/dist/tesseract.min.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.wasm',
  'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js',
  'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js',
  'https://cdn.jsdelivr.net/npm/docx@8.5.0/build/index.umd.js',
];

// Модели — только из сети, в кэш не кладём (они уже в IndexedDB).
function isModelRequest(url) {
  return url.hostname === 'huggingface.co' || url.pathname.endsWith('.onnx');
}

async function cacheOne(cache, url) {
  try { await cache.add(url); } catch (e) { /* офлайн или CDN недоступен — не роняем установку */ }
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => Promise.all([...SHELL_URLS, ...CDN_LIBS].map((u) => cacheOne(cache, u))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Сообщение от страницы: пропустить ожидание и сразу активировать новый SW.
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return; // POST и прочие — мимо кэша
  const url = new URL(req.url);

  // Модели: только сеть, никогда не кэшировать.
  if (isModelRequest(url)) return;

  // Навигация и index.html: сначала сеть, при неудаче — кэш (офлайн).
  if (req.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('/index.html')) {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE_NAME).then((c) => c.put('/index.html', copy)).catch(() => {});
          return resp;
        })
        .catch(() => caches.match('/index.html').then((r) => r || caches.match('/')))
    );
    return;
  }

  // Оболочка и CDN-библиотеки: сначала кэш, потом сеть (и докэшировать).
  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        if (resp && resp.ok) {
          const copy = resp.clone();
          caches.open(CACHE_NAME).then((c) => c.put(req, copy)).catch(() => {});
        }
        return resp;
      });
    })
  );
});
