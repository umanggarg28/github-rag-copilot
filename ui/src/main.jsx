import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Analytics } from '@vercel/analytics/react'
import posthog from 'posthog-js'
import './index.css'
import App from './App.jsx'

posthog.init('phc_B4VarKaWfNc3u7vMcsUPRDbNgSyVxaBqtYT3ZwP6FshM', {
  // Route through Vercel proxy (/ph/*) so ad blockers don't block requests
  // to us.i.posthog.com directly. The proxy is configured in vercel.json.
  api_host: '/ph',
  ui_host: 'https://us.posthog.com',
  capture_pageview: true,
  capture_pageleave: true,
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <Analytics />
  </StrictMode>,
)
