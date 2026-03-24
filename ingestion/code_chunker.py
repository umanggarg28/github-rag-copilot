"""
code_chunker.py — Split source files into semantically meaningful chunks.

Two strategies depending on file type:

1. AST chunking (Python only)
   Parse the file into an Abstract Syntax Tree. Extract each top-level
   function and class as its own chunk. Classes include all their methods.

   Why: A function is the natural unit of code — it has a name, inputs,
   outputs, and a single responsibility. Splitting mid-function loses context.

2. Character-window chunking (everything else)
   Split by character count with overlap — same approach used for prose.
   Works for markdown, YAML, config files, and languages without AST support.

   Why not AST for all languages? Python's `ast` module is in the stdlib.
   Multi-language AST (tree-sitter) adds complexity. For a learning project,
   Python AST + fallback covers 80% of cases cleanly.

Chunk shape (returned by both strategies):
  {
    "text":          str,        # the actual code/text content
    "language":      str,        # "python", "typescript", etc.
    "filepath":      str,        # "src/auth/middleware.py"
    "chunk_type":    str,        # "function", "class", "module", "text"
    "name":          str,        # function/class name (or "" for text chunks)
    "start_line":    int,        # 1-indexed line where chunk starts
    "end_line":      int,        # 1-indexed line where chunk ends
    "calls":         list[str],  # names called by this function (AST only)
  }

The `calls` field is used to build the Code Knowledge Graph — an interactive
D3 visualization of how functions call each other across files. It's extracted
by the CallExtractor visitor which walks ast.Call nodes inside each function body.
"""

import ast
import textwrap
from pathlib import Path


# ── Call extractor ────────────────────────────────────────────────────────────

class _CallExtractor(ast.NodeVisitor):
    """
    AST visitor that collects the names of all functions/methods called
    inside a function or class body.

    How ast.NodeVisitor works:
      - Subclass it and define visit_<NodeType> methods.
      - Call self.visit(node) to start traversal from any node.
      - self.generic_visit(node) continues the walk into child nodes.

    Two kinds of calls in Python's AST:
      ast.Name:      direct calls — foo(), bar()
                     → node.func is an ast.Name, name is node.func.id
      ast.Attribute: method/attr calls — self.foo(), obj.method()
                     → node.func is an ast.Attribute, name is node.func.attr

    We collect only the leaf name (not the full dotted path) because we match
    against function names in the index, not fully-qualified paths.
    """
    def __init__(self):
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)          # self.embed() → "embed"
        elif isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)            # embed() → "embed"
        self.generic_visit(node)                       # recurse into nested calls


def _extract_calls(node: ast.AST) -> list[str]:
    """Extract unique called names from an AST node (function or class)."""
    extractor = _CallExtractor()
    extractor.visit(node)
    # Deduplicate while preserving order; filter builtins that add noise
    _NOISE = {"print", "len", "range", "isinstance", "str", "int", "list",
               "dict", "set", "tuple", "super", "hasattr", "getattr", "setattr",
               "append", "extend", "format", "join", "split", "strip", "get",
               "items", "keys", "values", "zip", "enumerate", "map", "filter"}
    seen = set()
    result = []
    for name in extractor.calls:
        if name not in seen and name not in _NOISE:
            seen.add(name)
            result.append(name)
    return result


# ── AST Chunking (Python) ─────────────────────────────────────────────────────

def chunk_python(content: str, filepath: str) -> list[dict]:
    """
    Parse Python source and extract functions and classes as individual chunks.

    Algorithm:
      1. Parse content into an AST with ast.parse()
      2. Walk top-level nodes looking for FunctionDef, AsyncFunctionDef, ClassDef
      3. For each, extract the source lines using node.lineno / node.end_lineno
      4. If a node is too large (>60 lines), split it further into sub-chunks

    What about module-level code (imports, constants, global statements)?
    We collect it as a single "module" chunk. It's useful context for
    understanding what a file imports and configures.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        # Fall back to character-window if the file can't be parsed
        # (e.g. Python 2 syntax, encoding issues)
        print(f"  [ast parse failed for {filepath}: {e}] → fallback chunking")
        return chunk_by_window(content, filepath, language="python")

    lines = content.splitlines()
    chunks = []

    # Collect line numbers of all top-level definitions
    definition_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "lineno"):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    definition_lines.add(ln)

    # ── Module-level chunk ────────────────────────────────────────────────────
    # Lines not covered by any function/class (imports, constants, etc.)
    module_lines = [
        line for i, line in enumerate(lines, 1)
        if i not in definition_lines
    ]
    module_text = "\n".join(module_lines).strip()
    if module_text:
        chunks.append({
            "text":       f"# {filepath}\n{module_text}",
            "language":   "python",
            "filepath":   filepath,
            "chunk_type": "module",
            "name":       "",
            "start_line": 1,
            "end_line":   len(lines),
            "calls":      [],
        })

    # ── Function and class chunks ─────────────────────────────────────────────
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno
        end   = node.end_lineno or node.lineno
        node_lines = lines[start - 1 : end]
        node_text  = "\n".join(node_lines)

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        name = node.name

        # If the chunk is large, split into sub-chunks by method (for classes)
        # or by logical blocks (for large functions)
        if len(node_lines) > 80 and chunk_type == "class":
            sub_chunks = _split_class(node, lines, filepath)
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "text":       f"# {filepath}\n{node_text}",
                "language":   "python",
                "filepath":   filepath,
                "chunk_type": chunk_type,
                "name":       name,
                "start_line": start,
                "end_line":   end,
                "calls":      _extract_calls(node),
            })

    return chunks if chunks else chunk_by_window(content, filepath, language="python")


def _split_class(class_node: ast.ClassDef, lines: list[str], filepath: str) -> list[dict]:
    """
    Split a large class into per-method chunks.

    Each method gets the class signature as a header so the LLM knows
    which class the method belongs to:

      class MyClass:
          def __init__(self): ...
          ↓
      Chunk: "class MyClass:\n    def __init__(self): ..."
    """
    chunks = []
    class_start = class_node.lineno
    class_header = lines[class_start - 1]   # "class MyClass(Base):"

    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.lineno
        end   = node.end_lineno or node.lineno
        method_lines = lines[start - 1 : end]
        # Indent method lines if not already indented (should be)
        method_text  = "\n".join(method_lines)

        chunks.append({
            "text":       f"# {filepath}\n{class_header}\n{method_text}",
            "language":   "python",
            "filepath":   filepath,
            "chunk_type": "function",
            "name":       f"{class_node.name}.{node.name}",
            "start_line": start,
            "end_line":   end,
            "calls":      _extract_calls(node),
        })

    # Also include the class-level code (class variables, docstring)
    class_end = class_node.end_lineno or class_node.lineno
    class_text = "\n".join(lines[class_start - 1 : class_end])
    chunks.insert(0, {
        "text":       f"# {filepath}\n{class_text[:800]}",   # truncated overview
        "language":   "python",
        "filepath":   filepath,
        "chunk_type": "class",
        "name":       class_node.name,
        "start_line": class_start,
        "end_line":   class_end,
        "calls":      _extract_calls(class_node),
    })

    return chunks


# ── Character-window chunking (fallback) ──────────────────────────────────────

def chunk_by_window(
    content:   str,
    filepath:  str,
    language:  str = "text",
    chunk_size:  int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split text into overlapping fixed-size character windows.

    Used for:
      - Markdown documentation (.md, .rst)
      - Config files (.yaml, .toml, .json)
      - Languages without AST support (TypeScript, Go, Rust, etc.)
      - Python files that failed to parse

    The overlap ensures that a concept spanning a chunk boundary isn't lost.
    With overlap=200, the last 200 chars of chunk N are the first 200 chars
    of chunk N+1.
    """
    if not content.strip():
        return []

    lines  = content.splitlines()
    chunks = []
    start  = 0

    while start < len(content):
        end  = min(start + chunk_size, len(content))
        text = content[start:end]

        # Find approximate start/end line numbers for this character range
        start_line = content[:start].count("\n") + 1
        end_line   = content[:end].count("\n") + 1

        chunks.append({
            "text":       f"# {filepath}\n{text}",
            "language":   language,
            "filepath":   filepath,
            "chunk_type": "text",
            "name":       "",
            "start_line": start_line,
            "end_line":   end_line,
            "calls":      [],
        })

        if end == len(content):
            break
        start = end - chunk_overlap

    return chunks


# ── Main entry point ──────────────────────────────────────────────────────────

def chunk_file(file: dict) -> list[dict]:
    """
    Chunk a single file dict (as returned by repo_fetcher).

    Args:
        file: {"path": str, "content": str, "size": int, "repo": str}

    Returns:
        List of chunk dicts with text + metadata.
    """
    from ingestion.file_filter import language_from_path

    filepath = file.get("path") or file.get("filepath", "")
    content  = file["content"]
    language = language_from_path(filepath)
    repo     = file.get("repo", "")

    if language == "python":
        chunks = chunk_python(content, filepath)
    else:
        chunks = chunk_by_window(content, filepath, language=language)

    # Attach repo to every chunk
    for chunk in chunks:
        chunk["repo"] = repo

    return chunks


def chunk_files(files: list[dict]) -> list[dict]:
    """Chunk all files and return a flat list of all chunks."""
    all_chunks = []
    for file in files:
        file_chunks = chunk_file(file)
        all_chunks.extend(file_chunks)
        print(f"  {file.get('path') or file.get('filepath', '?')} → {len(file_chunks)} chunks")
    print(f"Total: {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks
