const CACHE_NAME = 'trading-agent-v4'

self.addEventListener('install', event => {
  self.skipWaiting()
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(['/trading-dashboard/']))
  )
})

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(key => caches.delete(key)))
    )
  )
})

self.addEventListener('fetch', event => {
  if (event.request.url.includes('railway.app')) {
    return
  }
  event.respondWith(fetch(event.request))
})