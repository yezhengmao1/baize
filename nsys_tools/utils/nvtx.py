"""
NVTX event utilities.
Author: yezhengmaolove@gmail.com
"""

import heapq
import sqlite3
import numpy as np
from typing import Iterable, Iterator, NamedTuple


# =============================================================================
# SQL
# =============================================================================

NVTX_EVENTS_SQL = """
SELECT n.start, n.end,
       COALESCE(n.text, s.value) AS name
FROM NVTX_EVENTS n
LEFT JOIN StringIds s ON s.id = n.textId
WHERE n.end IS NOT NULL
  AND COALESCE(n.text, s.value) IS NOT NULL
"""


class NvtxEvent(NamedTuple):
    start: int
    end: int
    rank: int | None
    name: str


class NvtxIndex:
    def __init__(self, conn: sqlite3.Connection, rank: int | None = None):
        """
        Build the index. This is a list of NVTX events, sorted by start time.
        """

        self.rank = rank
        self._needle_cache: dict[str, list[int]] = {}

        rows = conn.execute(NVTX_EVENTS_SQL).fetchall()
        if not rows:
            self.starts = np.empty(0, dtype=np.int64)
            self.ends = np.empty(0, dtype=np.int64)
            self.names: list[str] = []
            return

        # rows is a list of dicts, each with 'start', 'end', and 'name' keys.
        starts = np.fromiter(
            (r["start"] for r in rows), dtype=np.int64, count=len(rows)
        )
        ends = np.fromiter((r["end"] for r in rows), dtype=np.int64, count=len(rows))
        names = [r["name"] for r in rows]

        # get the order by start time, then sort the names by the order
        order = np.argsort(starts, kind="stable")
        self.starts = starts[order]
        self.ends = ends[order]
        self.names = [names[i] for i in order]

    def _needle_indices(self, needle: str) -> list[int]:
        """Indices into the sorted index of all frames whose name contains needle."""
        idx = self._needle_cache.get(needle)
        if idx is None:
            idx = [i for i, n in enumerate(self.names) if needle in n]
            self._needle_cache[needle] = idx
        return idx

    # ---- public API ---------------------------------------------------------

    def matches(self, needle: str) -> list[tuple[int, int, str]]:
        """Frames in the index whose name contains needle, as (start, end, name).

        Intended for callers that need to partition kernels by NVTX context
        (e.g. stepping through training iterations).
        """
        return [
            (int(self.starts[i]), int(self.ends[i]), self.names[i])
            for i in self._needle_indices(needle)
        ]

    def iter_stacks(
        self,
        api_intervals: Iterable[tuple[int, int]],
    ) -> Iterator[tuple[int, int, list[NvtxEvent]]]:
        """
        Yield (api_start, api_end, stack) for each interval using a sweep-line.

        Example — index holds these NVTX ranges (start, end, name):

            (0, 100, "Optimizer.step")
            (10,  90, "mlp.forward")
            (20,  50, "nccl:all_reduce")

        A kernel launched over the API interval (30, 40) is enclosed by all three,

            iter_stacks([(30, 40)])
            -> (30, 40, ["Optimizer.step", "mlp.forward", "nccl:all_reduce"])
               #          outermost (dur 100) ........... innermost (dur 30)
        """
        active: list[tuple[int, int]] = []  # (end, idx)
        ev_ptr = 0
        M = len(self.names)

        for api_s, api_e in api_intervals:
            while ev_ptr < M and int(self.starts[ev_ptr]) <= api_s:
                heapq.heappush(active, (int(self.ends[ev_ptr]), ev_ptr))
                ev_ptr += 1
            while active and active[0][0] < api_s:
                heapq.heappop(active)
            frames = [
                NvtxEvent(int(self.starts[i]), e, self.rank, self.names[i])
                for (e, i) in active
                if e >= api_e
            ]
            frames.sort(key=lambda f: f.end - f.start, reverse=True)
            yield api_s, api_e, frames
