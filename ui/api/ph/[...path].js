/**
 * Vercel Serverless Function — PostHog reverse proxy.
 * Routes /api/ph/* → https://us.i.posthog.com/* so ad blockers
 * can't match on the PostHog hostname.
 */
import https from 'https';

export const config = {
  api: { bodyParser: false, responseLimit: false },
};

export default async function handler(req, res) {
  const segments = Array.isArray(req.query.path)
    ? req.query.path.join('/')
    : (req.query.path || '');
  const query   = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
  const path    = `/${segments}${query}`;

  // Buffer the entire body first so we know the exact byte length.
  // req.pipe() was sending an empty body because Vercel may pre-consume
  // the stream; reading it explicitly avoids that.
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const body = chunks.length > 0 ? Buffer.concat(chunks) : null;

  // Only forward headers PostHog needs — strip Vercel/proxy-specific ones
  // that can confuse the upstream server.
  const forwardHeaders = {
    host:           'us.i.posthog.com',
    'content-type': req.headers['content-type'] || 'text/plain',
    'user-agent':   req.headers['user-agent']   || 'posthog-proxy',
  };
  if (body) forwardHeaders['content-length'] = String(body.byteLength);

  const options = {
    hostname: 'us.i.posthog.com',
    path,
    method:  req.method,
    headers: forwardHeaders,
  };

  await new Promise((resolve, reject) => {
    const proxyReq = https.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
      proxyRes.on('end', resolve);
    });
    proxyReq.on('error', reject);

    if (body) proxyReq.write(body);
    proxyReq.end();
  });
}
