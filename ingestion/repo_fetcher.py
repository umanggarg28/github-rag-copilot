"""
repo_fetcher.py — Fetch source files from a public GitHub repository.

Strategy: download the repo as a zip archive (one API call) and extract
files in memory — no local clone, no disk space for the repo itself.

Why not clone with git?
  - Cloning requires git to be installed on the host
  - Full clone history is large; we only need the current file contents
  - On free-tier hosting (Render), disk space is limited
  - The zip approach works anywhere that can make an HTTP request

GitHub API used:
  GET /repos/{owner}/{repo}/zipball/{ref}
    → 302 redirect to an S3 URL with the zip archive

  Rate limits (no auth): 60 requests/hour  ← enough for demos
  Rate limits (GITHUB_TOKEN set): 5,000 requests/hour

The function returns a list of dicts:
  [{"path": "src/main.py", "content": "def main(): ...", "size": 1234}, ...]

Only text files whose paths pass the file_filter are included.
"""

import io
import zipfile
import base64
import re
from pathlib import Path
from typing import Optional
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings


def parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract owner and repo name from a GitHub URL.

    Handles formats:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/main
      github.com/owner/repo
    """
    # Strip protocol and trailing slashes
    url = url.strip().rstrip("/")
    url = re.sub(r"^https?://", "", url)

    parts = url.split("/")
    if len(parts) < 2 or parts[0] != "github.com":
        raise ValueError(
            f"Invalid GitHub URL: {url!r}\n"
            "Expected format: https://github.com/owner/repo"
        )

    owner = parts[1]
    repo  = parts[2].removesuffix(".git")
    return owner, repo


def fetch_repo_files(
    github_url: str,
    file_filter_fn,          # callable(path: str) -> bool
    ref: str = "HEAD",
    max_file_size: int = 200_000,   # bytes — skip files larger than this
) -> list[dict]:
    """
    Download a GitHub repository as a zip and return filtered file contents.

    Args:
        github_url:     e.g. "https://github.com/openai/tiktoken"
        file_filter_fn: function that takes a file path and returns True if
                        we should index it (from file_filter.py)
        ref:            branch/tag/commit to fetch (default: HEAD = default branch)
        max_file_size:  skip files larger than this many bytes (avoids
                        indexing minified JS bundles, huge generated files, etc.)

    Returns:
        List of file dicts:
        [{"path": "src/encoder.py", "content": "...", "size": 4200}, ...]
    """
    owner, repo = parse_github_url(github_url)
    print(f"Fetching {owner}/{repo} @ {ref}...")

    # ── Step 1: Download zip archive ─────────────────────────────────────────
    # GitHub returns a 302 redirect to an S3 URL. requests follows it automatically.
    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{ref}"
    headers = _auth_headers()

    response = requests.get(zip_url, headers=headers, stream=True, timeout=60)
    if response.status_code == 404:
        raise ValueError(f"Repository not found: {owner}/{repo}. Is it public?")
    response.raise_for_status()

    print(f"  Downloaded {len(response.content) / 1024 / 1024:.1f} MB archive")

    # ── Step 2: Extract zip in memory ────────────────────────────────────────
    # The zip contains a top-level directory like "owner-repo-abc1234/" that we strip.
    zip_bytes = io.BytesIO(response.content)
    results = []
    skipped_binary = 0
    skipped_filter = 0
    skipped_size   = 0

    with zipfile.ZipFile(zip_bytes) as zf:
        entries = zf.infolist()
        # Determine the top-level prefix to strip (e.g. "openai-tiktoken-abc1234/")
        prefix = entries[0].filename if entries else ""

        for entry in entries:
            if entry.is_dir():
                continue

            # Strip the top-level prefix to get a clean relative path
            rel_path = entry.filename.removeprefix(prefix)
            if not rel_path:
                continue

            # Apply the file filter (language rules, excluded directories, etc.)
            if not file_filter_fn(rel_path):
                skipped_filter += 1
                continue

            # Skip large files
            if entry.file_size > max_file_size:
                skipped_size += 1
                print(f"  [skip — too large] {rel_path} ({entry.file_size / 1024:.0f} KB)")
                continue

            # Read file bytes
            try:
                raw_bytes = zf.read(entry.filename)
            except Exception:
                skipped_binary += 1
                continue

            # Decode as UTF-8 — skip files with binary content
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                skipped_binary += 1
                continue

            results.append({
                "path":    rel_path,
                "content": content,
                "size":    entry.file_size,
                "repo":    f"{owner}/{repo}",
            })

    print(
        f"  → {len(results)} files fetched | "
        f"{skipped_filter} filtered | "
        f"{skipped_binary} binary | "
        f"{skipped_size} too large"
    )
    return results


def get_repo_metadata(github_url: str) -> dict:
    """
    Fetch basic repo metadata (description, stars, default branch, language).
    Used to populate the UI and give the LLM context about what it's answering from.
    """
    owner, repo = parse_github_url(github_url)
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers=_auth_headers(), timeout=15)
    response.raise_for_status()
    data = response.json()
    return {
        "repo":          f"{owner}/{repo}",
        "description":   data.get("description", ""),
        "stars":         data.get("stargazers_count", 0),
        "language":      data.get("language", ""),
        "default_branch": data.get("default_branch", "main"),
        "url":           data.get("html_url", github_url),
    }


def _auth_headers() -> dict:
    """Build request headers, including auth token if GITHUB_TOKEN is set."""
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers
