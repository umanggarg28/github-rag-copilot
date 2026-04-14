"""
file_filter.py — Decide which files in a repo should be indexed.

The goal: index only files that carry meaningful semantic content about the
codebase. Exclude everything that is auto-generated, binary, or noise.

Why this matters for RAG quality:
  If you index node_modules/ (100k+ files) or package-lock.json (50k lines),
  queries like "how does authentication work?" return dependency internals
  instead of your actual code. The signal-to-noise ratio collapses.

Two-layer filtering:
  1. Directory exclusion — skip entire subtrees (node_modules/, .venv/, etc.)
  2. Extension inclusion — only index known text file types

This is intentionally conservative: it's better to miss a niche file type
than to flood the index with garbage.
"""

from pathlib import Path


# ── Directories to always exclude ────────────────────────────────────────────
# These are subtrees that contain dependencies, build output, or caches.
# Checked against every path component, so "a/node_modules/b" is caught.
EXCLUDED_DIRS = {
    # JS/TS ecosystem
    "node_modules", ".npm", ".yarn",
    # Python ecosystem
    ".venv", "venv", "env", "__pycache__", ".eggs", "*.egg-info",
    # Build output
    "dist", "build", "out", "target", ".next", ".nuxt",
    # Version control / tooling
    ".git", ".svn", ".hg",
    # IDE
    ".idea", ".vscode",
    # Dependencies (other languages)
    "vendor",       # Go, PHP
    "Pods",         # iOS CocoaPods
    ".gradle",      # Java/Kotlin
    # Misc
    ".cache", "coverage", ".nyc_output",
}

# ── File extensions to include ────────────────────────────────────────────────
# Only files with these extensions are indexed.
# Grouped by language/category for clarity.
INCLUDED_EXTENSIONS = {
    # Python
    ".py",
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Systems languages
    ".go", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    # JVM
    ".java", ".kt", ".scala",
    # Ruby / PHP / other scripting
    ".rb", ".php", ".swift",
    # Shell
    ".sh", ".bash", ".zsh",
    # Markup / docs
    ".md", ".mdx", ".rst", ".txt",
    # Config / data (selective — not lock files)
    ".yaml", ".yml", ".toml", ".json",
    # SQL
    ".sql",
    # HTML / templates (worth indexing for web projects)
    ".html", ".jinja", ".jinja2",
}

# ── Path patterns to always exclude ──────────────────────────────────────────
# These are path prefixes/substrings for directories that contain generated
# artifacts rather than source code, regardless of what repo is being indexed.
# Checked against the full relative path so "backend/diagrams/foo.json" matches.
EXCLUDED_PATH_PATTERNS = (
    # Generated artifact directories — runtime caches with no source value
    "diagrams/",       # cached diagram/tour JSON (Cartographer-style caches;
                       # image assets in other repos aren't indexed by extension anyway)
    "repo_maps/",      # cached repo map JSON
    ".cache/",         # generic cache directories
    "checkpoints/",    # ML model weight checkpoints (binary metadata, not code)
    # NOTE: migrations/ is intentionally NOT excluded — migration files contain
    # real schema SQL and are valuable for answering "how is the DB structured?"
    # NOTE: snapshots/ is intentionally NOT excluded — could be test fixtures,
    # API response recordings, or other useful data (Jest uses __snapshots__ which
    # is caught by the dot-prefix dir check, not this pattern)
)


# ── Specific filenames to always exclude ─────────────────────────────────────
# These are auto-generated or noisy even if they have an included extension.
EXCLUDED_FILENAMES = {
    # Lock files (auto-generated, massive, no semantic value)
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
    # Bundler output
    "bundle.js", "bundle.min.js",
    # Misc auto-generated
    ".DS_Store",
    "thumbs.db",
}

# ── Filename suffixes to exclude ──────────────────────────────────────────────
# Matches filenames ending with these patterns.
EXCLUDED_SUFFIXES = (
    ".min.js",      # minified JS (unreadable, noise)
    ".min.css",
    ".map",         # source maps
    ".pyc",         # compiled Python
    ".pb",          # protobuf binaries
    "_pb2.py",      # protobuf generated Python
    ".generated.ts", ".generated.js",  # code-gen output
)


def should_index(path: str) -> bool:
    """
    Return True if this file should be indexed.

    Args:
        path: relative path from repo root, e.g. "src/auth/middleware.py"

    Decision logic:
      1. Any path component matches an excluded dir → False
      2. Filename is in excluded filenames → False
      3. Filename ends with an excluded suffix → False
      4. Extension not in included extensions → False
      5. Otherwise → True
    """
    p = Path(path)

    # Check against excluded path patterns (generated-artifact directories).
    # Use forward slashes for consistency across platforms.
    normalized = path.replace("\\", "/")
    for pattern in EXCLUDED_PATH_PATTERNS:
        if pattern in normalized:
            return False

    # Check every component (handles "a/node_modules/b.js")
    for part in p.parts[:-1]:   # exclude filename itself
        if part in EXCLUDED_DIRS or part.startswith("."):
            return False

    filename  = p.name
    extension = p.suffix.lower()

    if filename in EXCLUDED_FILENAMES:
        return False

    if filename.endswith(EXCLUDED_SUFFIXES):
        return False

    if extension not in INCLUDED_EXTENSIONS:
        return False

    return True


def filter_files(paths: list[str]) -> list[str]:
    """Apply should_index to a list of paths and return only those that pass."""
    return [p for p in paths if should_index(p)]


def language_from_path(path: str) -> str:
    """
    Map a file extension to a human-readable language name.
    Used to populate chunk metadata (stored in Qdrant) for display and filtering.
    """
    ext_map = {
        ".py":    "python",
        ".js":    "javascript", ".jsx": "javascript",
        ".mjs":   "javascript", ".cjs": "javascript",
        ".ts":    "typescript", ".tsx": "typescript",
        ".go":    "go",
        ".rs":    "rust",
        ".java":  "java",
        ".kt":    "kotlin",
        ".rb":    "ruby",
        ".php":   "php",
        ".c":     "c",  ".h":  "c",
        ".cpp":   "cpp", ".cc": "cpp", ".hpp": "cpp",
        ".sh":    "shell", ".bash": "shell", ".zsh": "shell",
        ".md":    "markdown", ".mdx": "markdown",
        ".rst":   "rst",
        ".yaml":  "yaml", ".yml": "yaml",
        ".toml":  "toml",
        ".json":  "json",
        ".sql":   "sql",
        ".html":  "html",
        ".swift": "swift",
        ".scala": "scala",
    }
    return ext_map.get(Path(path).suffix.lower(), "text")
