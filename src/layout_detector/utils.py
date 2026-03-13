"""Compatibility helpers for PaddleOCR / paddlex.

Purpose
- Provide minimal legacy LangChain modules expected by some `paddlex` builds
    that are imported transitively by `paddleocr`.

Why
- Older `paddlex` versions reference `langchain.docstore.document.Document` and
    `langchain.text_splitter.RecursiveCharacterTextSplitter`, which moved or were
    removed in modern LangChain.

Usage
- Call `apply_paddlex_langchain_shims()` once before importing `paddleocr` so
    its transitive `paddlex` imports succeed without ModuleNotFoundError.
"""

from __future__ import annotations

import sys
import types
from typing import Any, List
import os


def _shim_langchain_docstore_document() -> None:
    """Install a minimal `langchain.docstore.document.Document`.

    Modern LangChain (>= 0.2) relocates or removes this symbol. We provide a
    compatible `Document` class by trying modern locations first and falling
    back to a simple stub if necessary.
    """
    if "langchain.docstore.document" in sys.modules:
        return
    DocumentCls = None
    try:  # LangChain >=0.2
        from langchain_core.documents import Document as DocumentCls  # type: ignore
    except Exception:
        try:  # Older path
            from langchain.schema import Document as DocumentCls  # type: ignore
        except Exception:
            pass
    if DocumentCls is None:
        class DocumentCls:  # minimal, last-resort stub
            def __init__(self, page_content: str = "", metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}
    # Ensure parent package exists and register module
    if "langchain.docstore" not in sys.modules:
        sys.modules["langchain.docstore"] = types.ModuleType("langchain.docstore")
    mod = types.ModuleType("langchain.docstore.document")
    mod.Document = DocumentCls
    sys.modules["langchain.docstore.document"] = mod


def _shim_langchain_text_splitter() -> None:
    """Install a minimal `langchain.text_splitter.RecursiveCharacterTextSplitter`.

    Tries to import from `langchain_text_splitters` (the modern location) or
    from legacy `langchain.text_splitter`. If neither exists, defines a tiny
    fallback that splits text into overlapping chunks.
    """
    if "langchain.text_splitter" in sys.modules:
        return
    Splitter = None
    try:
        from langchain_text_splitters import (  # type: ignore
            RecursiveCharacterTextSplitter as Splitter,
        )
    except Exception:
        try:
            from langchain.text_splitter import (  # type: ignore
                RecursiveCharacterTextSplitter as Splitter,
            )
        except Exception:
            pass
    if Splitter is None:
        class Splitter:  # fallback splitter
            def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **_: Any):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text: str) -> List[str]:
                pieces: List[str] = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    pieces.append(text[start:end])
                    start = max(end - self.chunk_overlap, end)
                return pieces
    mod = types.ModuleType("langchain.text_splitter")
    mod.RecursiveCharacterTextSplitter = Splitter
    sys.modules["langchain.text_splitter"] = mod


def apply_paddlex_langchain_shims() -> None:
    """
    Install legacy LangChain symbols so older paddlex imports do not fail.

    This helper pre-installs minimal shims for Document and RecursiveCharacterTextSplitter
    so that transitive imports inside `paddlex` and `paddleocr` succeed on modern
    LangChain versions without crashing.

    Processing order:
        1. Ensure langchain.docstore.document.Document is available
        2. Ensure langchain.text_splitter.RecursiveCharacterTextSplitter is available

    """
    _shim_langchain_docstore_document()
    _shim_langchain_text_splitter()


def resolve_layout_output_dir(image_path: str) -> tuple[str, str]:
    """
    Compute the standardized output directory for a layout-detector run.

    This helper centralizes the base-name extraction and path normalization so
    that all layout-detector tools write artifacts under `data/layout_detector/<base>`.

    Processing order:
        1. Derive the base name from the input path
        2. Build the fixed output path under data/layout_detector
        3. Normalize the path for portability

    Args:
        image_path: Absolute or relative path to the image being processed.

    Returns:
        tuple[str, str]: (base_name, normalized_output_dir) for the image.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    fixed_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "layout_detector", base)
    return base, os.path.normpath(fixed_output_dir)
