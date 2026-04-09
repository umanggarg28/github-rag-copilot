export default function handler(req, res) {
  res.json({ method: req.method, ok: true });
}
