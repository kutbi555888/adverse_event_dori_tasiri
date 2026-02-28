from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None, marker_dir: str = "Data") -> Path:
    """
    Project root'ni topadi: joriy papka yoki uning parentlarida 'Data/' bo‘lsa, o‘shani root deb oladi.
    Aks holda start papkani qaytaradi.
    """
    start = start or Path.cwd()
    start = start.resolve()

    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / marker_dir).exists():
            return p
    return start