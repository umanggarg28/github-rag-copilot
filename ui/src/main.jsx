import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Analytics } from '@vercel/analytics/react'
import posthog from 'posthog-js'
import './index.css'
import App from './App.jsx'

// Routing — the repo and view live in the URL rather than React state, so:
//   • Refreshing the page preserves where the user was
//   • Links are shareable (paste in Slack, others land directly on that view)
//   • Browser back/forward works without bespoke history shims
//   • Pre-indexed repos load instantly for any user (Qdrant is shared backend)
//
// All routes mount the same <App />; inside, App reads useParams +
// useLocation to derive `activeRepo` and the active view. Keeping the
// route definitions tiny avoids a forest of wrapper components.
//
//   /                              → landing
//   /r/:owner/:repo                → repo view (defaults to graph)
//   /r/:owner/:repo/diagram        → diagram (graph) view
//   /r/:owner/:repo/chat           → chat view
//   anything else                  → redirect to landing
//
// Conversation-level URLs (Tier 2) and concept deep links (Tier 3) attach
// later under the same /r/:owner/:repo prefix without touching this file.

posthog.init('phc_B4VarKaWfNc3u7vMcsUPRDbNgSyVxaBqtYT3ZwP6FshM', {
  // Route through /api/ph (Vercel serverless proxy) so ad blockers
  // don't block requests going directly to us.i.posthog.com.
  api_host: '/api/ph',
  ui_host: 'https://us.posthog.com',
  capture_pageview: true,
  capture_pageleave: true,
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/"                                element={<App />} />
        <Route path="/r/:owner/:repo"                  element={<App />} />
        <Route path="/r/:owner/:repo/diagram"          element={<App />} />
        <Route path="/r/:owner/:repo/chat"             element={<App />} />
        <Route path="/r/:owner/:repo/c/:sessionId"     element={<App />} />
        <Route path="*"                                element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
    <Analytics />
  </StrictMode>,
)
