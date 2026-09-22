"""Synchronized CUDA timing and memory snapshots for the efficiency MVE."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterator

import torch


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return float(ordered[index])


def summarize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("kind") != "timing":
            continue
        groups.setdefault(str(event["name"]), []).append(event)
    rows = []
    for name, members in sorted(groups.items()):
        wall = [float(item["wall_ms"]) for item in members]
        cuda = [float(item["cuda_ms"]) for item in members if item.get("cuda_ms") is not None]
        rows.append(
            {
                "name": name,
                "count": len(members),
                "wall_median_ms": statistics.median(wall),
                "wall_p90_ms": _percentile(wall, 0.90),
                "wall_p95_ms": _percentile(wall, 0.95),
                "cuda_median_ms": statistics.median(cuda) if cuda else None,
                "cuda_p90_ms": _percentile(cuda, 0.90) if cuda else None,
                "cuda_p95_ms": _percentile(cuda, 0.95) if cuda else None,
            }
        )
    return rows


class EfficiencyRecorder:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.events: list[dict[str, Any]] = []
        self._step_state: dict[str, Any] | None = None

    def mark_step(self, *, phase: str, step_index: int, **metadata: Any) -> None:
        key = (str(phase), int(step_index))
        if self._step_state is not None and self._step_state["key"] == key:
            return
        self.finish_step()
        start_event = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        self._step_state = {
            "key": key,
            "phase": str(phase),
            "step_index": int(step_index),
            "metadata": dict(metadata),
            "wall_start": time.perf_counter(),
            "cuda_start": start_event,
        }

    def finish_step(self) -> None:
        state = self._step_state
        if state is None:
            return
        cuda_ms = None
        if torch.cuda.is_available():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            cuda_ms = float(state["cuda_start"].elapsed_time(end_event))
        self.events.append(
            {
                "kind": "timing",
                "name": f"step.{state['phase']}",
                "phase": state["phase"],
                "step_index": state["step_index"],
                "wall_ms": (time.perf_counter() - state["wall_start"]) * 1000.0,
                "cuda_ms": cuda_ms,
                **state["metadata"],
            }
        )
        self._step_state = None

    @contextmanager
    def measure(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        is_cuda = torch.cuda.is_available()
        start_event = end_event = None
        if is_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        wall_start = time.perf_counter()
        try:
            yield
        finally:
            if is_cuda:
                end_event.record()
                torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            cuda_ms = float(start_event.elapsed_time(end_event)) if is_cuda else None
            self.events.append(
                {
                    "kind": "timing",
                    "name": str(name),
                    "wall_ms": wall_ms,
                    "cuda_ms": cuda_ms,
                    **metadata,
                }
            )

    def memory_snapshot(self, name: str, **metadata: Any) -> dict[str, Any]:
        snapshot = {
            "kind": "memory",
            "name": str(name),
            "allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
            "reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            **metadata,
        }
        self.events.append(snapshot)
        return snapshot

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in self.events),
            encoding="utf-8",
        )
