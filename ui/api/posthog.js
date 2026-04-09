/**
 * PostHog reverse proxy — all /api/ph/* traffic is rewritten here by vercel.json.
 * The original path is passed as ?path=... so we can reconstruct the full PostHog URL.
 */
import https from 'https';

export const config = { api: { bodyParser: false, responseLimit: false } };

export default async function handler(req, res) {
  const { path: phPath = '/', ...rest } = req.query;

  // Reconstruct the PostHog target URL: prepend / if missing, forward all
  // original query params (ip, ver, compression, etc.) except our injected 'path'.
  const qs = new URLSearchParams(rest).toString();
  const targetPath = `/${phPath}${qs ? '?' + qs : ''}`;

  // Buffer the full body so content-length is accurate when forwarded.
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks);

  const headers = {
    host:           'us.i.posthog.com',
    'content-type': req.headers['content-type'] || 'text/plain',
    'user-agent':   req.headers['user-agent']   || 'posthog-proxy',
  };
  if (body.length) headers['content-length'] = String(body.length);

  await new Promise((resolve, reject) => {
    const proxyReq = https.request(
      { hostname: 'us.i.posthog.com', path: targetPath, method: req.method, headers },
      (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);
        proxyRes.on('end', resolve);
      }
    );
    proxyReq.on('error', reject);
    if (body.length) proxyReq.write(body);
    proxyReq.end();
  });
}
