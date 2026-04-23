"""Simple timer for measuring MTP overhead in a running server.

Usage: import and call `timer.enter(label)` before and `timer.exit(label)` after.
Uses torch.cuda.Event to avoid CPU-GPU syncs.
Prints a summary every `SUMMARY_EVERY` timer.flush() calls.
"""
import os
import time
from collections import defaultdict

import torch


class MTPTimer:
    def __init__(self):
        self.enabled = os.getenv("MTP_TIMING", "0") == "1"
        self.summary_every = int(os.getenv("MTP_TIMING_EVERY", "50"))
        self._events = defaultdict(list)  # label -> list of (start_evt, end_evt)
        self._open_stack = []
        self._iter = 0
        self._cpu_counts = defaultdict(int)
        self._cpu_times = defaultdict(float)

    def enter(self, label: str):
        if not self.enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        self._open_stack.append((label, start, t0))

    def exit(self, label: str):
        if not self.enabled:
            return
        top_label, start, t0 = self._open_stack.pop()
        assert top_label == label, f"timer mismatch: {top_label} vs {label}"
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events[label].append((start, end))
        self._cpu_times[label] += time.perf_counter() - t0
        self._cpu_counts[label] += 1

    def tick(self, ctx_max_seq_len: int = 0):
        """Call once per propose()-level iteration. Periodically summarize."""
        if not self.enabled:
            return
        self._iter += 1
        if self._iter % self.summary_every != 0:
            return
        # Flush: for each label, compute total / avg GPU time
        torch.cuda.synchronize()
        rows = []
        for label, evts in self._events.items():
            totals = 0.0
            for s, e in evts:
                totals += s.elapsed_time(e)
            n = len(evts)
            cpu_ms = self._cpu_times[label] * 1000.0
            rows.append(
                (
                    label,
                    n,
                    totals,
                    totals / max(n, 1),
                    cpu_ms,
                    cpu_ms / max(n, 1),
                )
            )
        rows.sort(key=lambda r: -r[2])
        print(
            f"[MTP_TIMING iter={self._iter} ctx_max={ctx_max_seq_len}] "
            f"GPU and CPU (perf_counter) ms/iter:",
            flush=True,
        )
        print(
            f"  {'label':<40s}  {'n':>6s}  {'gpu_total':>9s}  {'gpu_avg':>9s}  "
            f"{'cpu_total':>9s}  {'cpu_avg':>9s}",
            flush=True,
        )
        for label, n, gtot, gavg, ctot, cavg in rows:
            print(
                f"  {label:<40s}  {n:>6d}  {gtot:>9.2f}  {gavg:>9.3f}  "
                f"{ctot:>9.2f}  {cavg:>9.3f}",
                flush=True,
            )
        # Reset
        self._events.clear()
        self._cpu_times.clear()
        self._cpu_counts.clear()


_GLOBAL_TIMER = MTPTimer()


def enter(label):
    _GLOBAL_TIMER.enter(label)


def exit(label):
    _GLOBAL_TIMER.exit(label)


def tick(ctx_max_seq_len=0):
    _GLOBAL_TIMER.tick(ctx_max_seq_len)
