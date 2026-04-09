/**
 * Vercel Serverless Function — PostHog reverse proxy.
 *
 * Why: ad blockers match on "us.i.posthog.com" and block analytics with
 * ERR_BLOCKED_BY_CLIENT. Routing through our own domain (/api/ph/*) makes
 * requests look first-party, bypassing the block.
 *
 * Using Node.js runtime (not Edge) because Edge runtime requires Next.js
 * for Vercel's file-based routing to work correctly. Node.js runtime
 * supports all HTTP methods and body types natively.
 */
const POSTHOG_HOST = 'https://us.i.posthog.com';

export default async function handler(req, res) {
  // Reconstruct the PostHog path from the wildcard [...path] segment
  const segments = req.query.path || [];
  const pathStr  = Array.isArray(segments) ? segments.join('/') : segments;
  const query    = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
  const targetUrl = `${POSTHOG_HOST}/${pathStr}${query}`;

  // Read the incoming body as a raw buffer (preserves gzip/base64 compression
  // that PostHog JS applies — do NOT parse or re-encode it)
  const bodyChunks = [];
  for await (const chunk of req) {
    bodyChunks.push(chunk);
  }
  const rawBody = bodyChunks.length > 0 ? Buffer.concat(bodyChunks) : undefined;

  // Forward only the headers PostHog needs; strip Vercel-specific ones
  const forwardedHeaders = {
    host:             'us.i.posthog.com',
    'content-type':   req.headers['content-type']   || 'text/plain',
    'user-agent':     req.headers['user-agent']      || 'posthog-proxy',
  };
  if (req.headers['content-encoding']) {
    forwardedHeaders['content-encoding'] = req.headers['content-encoding'];
  }

  const proxyResp = await fetch(targetUrl, {
    method:  req.method,
    headers: forwardedHeaders,
    body:    ['GET', 'HEAD'].includes(req.method) ? undefined : rawBody,
  });

  // Forward response status + headers back to the browser
  res.status(proxyResp.status);
  proxyResp.headers.forEach((value, key) => {
    // Skip headers that Node's http module manages itself
    if (!['transfer-encoding', 'connection'].includes(key)) {
      res.setHeader(key, value);
    }
  });

  const responseBody = await proxyResp.arrayBuffer();
  res.send(Buffer.from(responseBody));
}
