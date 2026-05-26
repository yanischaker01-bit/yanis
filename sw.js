/* Service Worker – Carte LGV SEA */
const CACHE = 'lgv-sea-v22';
const TILE_CACHE = 'lgv-tiles-v1';

/* Ne jamais pré-cacher le HTML : il doit toujours venir du réseau */
const PRECACHE = [
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './data/rail.geojson',
  './data/pk.geojson',
  './data/pk_hecto.geojson',
  './data/oa.geojson',
  './data/oa_poly.geojson',
  './data/oh.geojson',
  './data/pam.geojson',
  './data/acces.geojson',
  './data/old.geojson',
  './data/mc.geojson',
  './data/n2.geojson',
  './data/eco.geojson',
  './data/bois.geojson',
  './data/oa_poly.pmtiles',
];

self.addEventListener('message', e => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(function(c) {
        return Promise.all(
          PRECACHE.map(function(url) {
            return c.add(url).catch(function() {
              console.warn('SW: précache ignoré:', url);
            });
          })
        );
      })
      .then(function() { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== CACHE && k !== TILE_CACHE).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = e.request.url;

  /* Navigation HTML → jamais interceptée, toujours réseau direct */
  if (e.request.mode === 'navigate') return;

  /* Tuiles OSM/Esri/IGN → réseau d'abord, cache en fallback */
  if (/openstreetmap\.org|arcgisonline\.com|geoportail/.test(url)) {
    e.respondWith(
      fetch(e.request)
        .then(r => {
          const clone = r.clone();
          caches.open(TILE_CACHE).then(c => c.put(e.request, clone));
          return r;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  /* Tout le reste → cache d'abord, réseau en fallback */
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(r => {
        if (r.ok) {
          const clone = r.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return r;
      });
    })
  );
});
