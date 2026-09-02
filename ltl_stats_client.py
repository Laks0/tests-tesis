"""Client for MTSA's ltsa.lts.ltl.LTLStatsService.

Send an LTL formula (MTSA / LTSA syntax, e.g. ``[](a -> <>b)``) and get back the
size of the Buchi automata that ``AssertDefinition.compile`` builds for it:

* ``gba``   -- the generalized Buchi automaton right after ``gba.translate()``
               (before super-set reduction / degeneralisation / minimisation),
* ``buchi`` -- the final automaton (``CompositeState.composition``) after the
               whole compilation pipeline.

Each is reported as ``states`` and ``transitions``.

Two transports, same API:

    # A. spawn the Java service as a child process (pipe mode) -- no ports
    from ltl_stats_client import LTLStats
    with LTLStats.spawn("/path/to/maven-root/mtsa") as svc:
        print(svc.stats("[](a -> <>b)"))
        print(svc.stats("<>[]a"))

    # B. talk to an already-running TCP server
    #    (started with:  ./ltl-stats-server.sh --server 5747)
    with LTLStats.connect("localhost", 5747) as svc:
        print(svc.stats("a U b"))

``stats()`` returns a dict, e.g.::

    {"formula": "[](a -> <>b)", "ok": True, "isProperty": False,
     "gba":   {"states": 9, "transitions": 21},
     "buchi": {"states": 5, "transitions": 15}}

On a compilation error ``ok`` is False and an ``error`` key holds the message
(``stats()`` raises LTLStatsError unless ``raise_on_error=False``).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from typing import Optional


class LTLStatsError(RuntimeError):
    """The service could not compile a formula."""


class AutomatonSize:
    __slots__ = ("states", "transitions")

    def __init__(self, states: int, transitions: int) -> None:
        self.states = states
        self.transitions = transitions

    def __repr__(self) -> str:
        return f"AutomatonSize(states={self.states}, transitions={self.transitions})"


class LTLStatsResult:
    """Convenience wrapper around the raw JSON dict returned by the service."""

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.formula: str = raw.get("formula", "")
        self.ok: bool = bool(raw.get("ok"))
        self.error: Optional[str] = raw.get("error")
        self.is_property: Optional[bool] = raw.get("isProperty")
        self.gba = _size(raw.get("gba"))
        self.buchi = _size(raw.get("buchi"))

    def __repr__(self) -> str:
        if not self.ok:
            return f"LTLStatsResult(formula={self.formula!r}, error={self.error!r})"
        return (
            f"LTLStatsResult(formula={self.formula!r}, is_property={self.is_property}, "
            f"gba={self.gba!r}, buchi={self.buchi!r})"
        )


def _size(d: Optional[dict]) -> Optional[AutomatonSize]:
    if not d:
        return None
    return AutomatonSize(d["states"], d["transitions"])


class LTLStats:
    """A line-based JSON connection to LTLStatsService (subprocess or socket)."""

    def __init__(self, reader, writer, closer=None) -> None:
        self._reader = reader          # file-like, text mode, readline()
        self._writer = writer          # file-like, text mode, write()/flush()
        self._closer = closer          # optional callable for teardown

    # ----------------------------------------------------------------- factories
    @classmethod
    def spawn(
        cls,
        mtsa_project_dir: str,
        server_script: Optional[str] = None,
        java_ready_timeout: Optional[float] = None,  # kept for API symmetry; unused
    ) -> "LTLStats":
        """Start the Java service in pipe mode as a child process.

        ``mtsa_project_dir`` is the ``maven-root/mtsa`` directory of this repo.
        The first call triggers a Maven build, so it can take a while.
        """
        if server_script is None:
            server_script = os.path.join(
                mtsa_project_dir, "tools", "ltl-stats", "ltl-stats-server.sh"
            )
        proc = subprocess.Popen(
            ["bash", server_script],
            cwd=mtsa_project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let build/progress noise go to our stderr
            text=True,
            bufsize=1,
        )

        def _close():
            try:
                if proc.stdin:
                    proc.stdin.close()
            finally:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        return cls(proc.stdout, proc.stdin, _close)

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 5747,
                timeout: Optional[float] = None) -> "LTLStats":
        """Connect to a running ``LTLStatsService --server`` instance."""
        sock = socket.create_connection((host, port), timeout=timeout)
        # A single socket file object in r+ mode shares one fd fine here.
        rw = sock.makefile("rw", encoding="utf-8", newline="\n", buffering=1)
        return cls(rw, rw, lambda: (rw.close(), sock.close()))

    # -------------------------------------------------------------------- query
    def stats(self, formula: str, raise_on_error: bool = True) -> LTLStatsResult:
        """Compile one formula and return its automaton sizes."""
        if "\n" in formula or "\r" in formula:
            raise ValueError("formula must be a single line")
        self._writer.write(formula + "\n")
        self._writer.flush()

        line = self._reader.readline()
        if not line:
            raise LTLStatsError("LTLStatsService closed the stream (no response)")
        result = LTLStatsResult(json.loads(line))
        if raise_on_error and not result.ok:
            raise LTLStatsError(f"{formula!r}: {result.error}")
        return result

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._closer is not None:
            self._closer()
            self._closer = None

    def __enter__(self) -> "LTLStats":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Query MTSA LTLStatsService.")
    ap.add_argument("formula", nargs="*", help="LTL formula(s); omit to read stdin")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mtsa", metavar="DIR",
                   help="maven-root/mtsa dir; spawns the service as a subprocess")
    g.add_argument("--tcp", metavar="HOST:PORT",
                   help="connect to a running --server instance")
    args = ap.parse_args()

    if args.tcp:
        host, _, port = args.tcp.partition(":")
        svc = LTLStats.connect(host or "localhost", int(port or 5747))
    else:
        svc = LTLStats.spawn(args.mtsa)

    with svc:
        formulas = args.formula or (l.strip() for l in sys.stdin if l.strip())
        for f in formulas:
            res = svc.stats(f, raise_on_error=False)
            print(json.dumps(res.raw))

