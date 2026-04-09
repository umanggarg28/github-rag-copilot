/**
 * Vercel Serverless Function — PostHog reverse proxy.
 *
 * Ad blockers match on "us.i.posthog.com" and drop requests. Routing through
 * our own Vercel domain (/api/ph/*) makes them look first-party.
 *
 * Uses Node.js stream piping (req.pipe → proxyRes.pipe → res) so the body
 * is forwarded byte-for-byte with no parsing — preserving PostHog's gzip/base64
 * compression without any re-encoding issues.
 */
import https from 'https';

export default function handler(req, res) {
  const segments = Array.isArray(req.query.path)
    ? req.query.path.join('/')
    : (req.query.path || '');

  const query = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
  const path  = `/${segments}${query}`;

  const options = {
    hostname: 'us.i.posthog.com',
    path,
    method:  req.method,
    headers: {
      ...req.headers,
      host: 'us.i.posthog.com',
    },
  };

  const proxyReq = https.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    res.status(500).json({ error: err.message });
  });

  // Pipe the incoming body directly — no buffering, no re-encoding
  req.pipe(proxyReq);
}
