/**
 * Vercel Edge Function — PostHog reverse proxy.
 *
 * Why this exists: ad blockers match on the hostname "us.i.posthog.com" and
 * block analytics requests with ERR_BLOCKED_BY_CLIENT. By routing through our
 * own domain (/api/ph/*) the requests look first-party and slip past blockers.
 *
 * Vercel `rewrites` in vercel.json only handle GET routing; POST bodies are
 * not forwarded to external hosts. An Edge Function gives us full control to
 * proxy every method (GET, POST) with the correct headers and body.
 */
export const config = { runtime: 'edge' };

const POSTHOG_HOST = 'https://us.i.posthog.com';

export default async function handler(req) {
  const url = new URL(req.url);

  // Strip the /api/ph prefix — keep path + query string for PostHog
  const targetPath = url.pathname.replace(/^\/api\/ph/, '') || '/';
  const targetUrl  = `${POSTHOG_HOST}${targetPath}${url.search}`;

  // Build forwarded headers, replacing host so PostHog accepts the request
  const headers = new Headers(req.headers);
  headers.set('host', 'us.i.posthog.com');

  const proxyResp = await fetch(targetUrl, {
    method:  req.method,
    headers,
    body:    ['GET', 'HEAD'].includes(req.method) ? undefined : req.body,
    redirect: 'follow',
  });

  return new Response(proxyResp.body, {
    status:     proxyResp.status,
    statusText: proxyResp.statusText,
    headers:    proxyResp.headers,
  });
}
