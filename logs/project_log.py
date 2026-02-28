# logs/project_log.py
from __future__ import annotations

import logging
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from logging.handlers import RotatingFileHandler
from uuid import uuid4

LOG_DIR = Path(__file__).resolve().parent
PROJECT_LOG_FILE = LOG_DIR / "project_log.log"   # << project log
ERROR_LOG_FILE   = LOG_DIR / "errors.log"        # << error log

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FMT = "%(asctime)s | %(levelname)s | %(message)s"

_project_logger: logging.Logger | None = None
_error_logger: logging.Logger | None = None


def _make_logger(name: str, filepath: Path, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    filepath.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=str(filepath),
        mode="a",                 # ✅ append (ulanib ketadi)
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FMT, datefmt=DATE_FMT))

    logger.addHandler(handler)
    return logger


def project_logger() -> logging.Logger:
    global _project_logger
    if _project_logger is None:
        _project_logger = _make_logger("PROJECT_LOG", PROJECT_LOG_FILE, logging.INFO)
    return _project_logger


def error_logger() -> logging.Logger:
    global _error_logger
    if _error_logger is None:
        _error_logger = _make_logger("ERROR_LOG", ERROR_LOG_FILE, logging.ERROR)
    return _error_logger


def _fmt_duration(sec: float) -> str:
    sec = int(sec)
    m = sec // 60
    s = sec % 60
    h = m // 60
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class StageCtx:
    stage_name: str
    run_id: str
    t0: float
    finished: bool = False


def stage_start(stage_name: str) -> StageCtx:
    """
    Bosqich boshlanishi:
    project_log.log ga 'BOSHLANDI' yozadi.
    """
    ctx = StageCtx(stage_name=stage_name, run_id=uuid4().hex[:8], t0=time.time())
    project_logger().info(f"[RUN {ctx.run_id}] ✅ BOSHLANDI: {ctx.stage_name}")
    return ctx


def stage_end(ctx: StageCtx) -> None:
    """
    Bosqich tugashi:
    project_log.log ga 'TUGADI' yozadi.
    """
    if ctx.finished:
        return
    ctx.finished = True
    dur = _fmt_duration(time.time() - ctx.t0)
    project_logger().info(f"[RUN {ctx.run_id}] ✅ TUGADI: {ctx.stage_name} | davomiyligi: {dur}")


def stage_error(ctx: StageCtx, exc: BaseException) -> None:
    """
    Xato bo'lsa:
    errors.log ga 'XATO YUZ BERDI: <type>: <msg>' + traceback yozadi.
    """
    etype = type(exc).__name__
    emsg = str(exc)

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    error_logger().error(
        f"[RUN {ctx.run_id}] ❌ XATO YUZ BERDI: {ctx.stage_name}\n"
        f"Xato turi: {etype}\n"
        f"Xato matni: {emsg}\n"
        f"Traceback:\n{tb}"
    )


class stage:
    """
    with stage("..."): blok ichida ishlatish uchun.
    - start -> boshlanganini yozadi
    - exception -> errors.log ga yozadi
    - end -> tugaganini yozadi
    """
    def __init__(self, stage_name: str):
        self.ctx = stage_start(stage_name)

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            stage_error(self.ctx, exc)
        stage_end(self.ctx)
        return False  # xatoni yashirmaydi (Jupyter ham ko‘rsatadi)