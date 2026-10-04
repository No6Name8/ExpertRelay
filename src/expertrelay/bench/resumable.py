"""A set of benchmark runs that survives interruptions: each run is saved
the moment it finishes, a rerun does only the runs not yet saved, and
every start is a recorded session (machine profile, free RAM), so results
finished in a second session say so.

Used by bench.fair_test; bench.gptq_benchmark and bench.manager_validation
predate it and keep their own copy of the same logic (their recorded
results were produced by that code).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import base_record
from expertrelay.manager.profile import collect_machine_profile


def portable(v):
    """Paths inside the repo as repo-relative strings, recursively."""
    if isinstance(v, Path):
        v = str(v)
    if isinstance(v, str) and v.startswith(str(REPO_ROOT)):
        return Path(v).relative_to(REPO_ROOT).as_posix()
    if isinstance(v, list):
        return [portable(x) for x in v]
    if isinstance(v, dict):
        return {k: portable(x) for k, x in v.items()}
    return v


def write_json(path: Path, obj: dict) -> None:
    """Complete or absent: written to a temporary file, then renamed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class RunSet:
    work: Path
    settings: dict
    session: int = -1

    def start(self, label: str) -> int:
        """Check the saved settings match, record a new session, return its index."""
        self.work.mkdir(parents=True, exist_ok=True)
        saved = self.work / "settings.json"
        if saved.exists():
            if json.loads(saved.read_text()) != self.settings:
                raise SystemExit(f"{saved} is from other code/settings; move {self.work} away to start over")
        else:
            saved.write_text(json.dumps(self.settings, indent=1))
        self.session = len(list(self.work.glob("session_*.json")))
        machine = collect_machine_profile(measure_disk=False)
        write_json(
            self.work / f"session_{self.session}.json",
            base_record(
                label=f"{label} session {self.session}", seed=0, model=None, config={}, machine=machine
            ),
        )
        print(f"session {self.session}: {machine.memory.available_bytes / 1e9:.2f} GB RAM free", flush=True)
        return self.session

    def path(self, run_id: str) -> Path:
        return self.work / "runs" / f"{run_id}.json"

    def done(self, run_id: str) -> bool:
        p = self.path(run_id)
        return p.exists() and json.loads(p.read_text(encoding="utf-8"))["status"] == "ok"

    def prepare(self, run_id: str) -> None:
        """A saved failed attempt is kept under failed/ before the run is redone."""
        p = self.path(run_id)
        if p.exists():
            failed = self.work / "failed" / f"{p.stem}.session{self.session}.json"
            failed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, failed)

    def save(self, run_id: str, result: dict) -> None:
        write_json(self.path(run_id), {**portable(result), "run_id": run_id, "session": self.session})

    def load(self, run_id: str) -> dict:
        return json.loads(self.path(run_id).read_text(encoding="utf-8"))

    def sessions(self) -> list[dict]:
        out = []
        for p in sorted(self.work.glob("session_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            x = json.loads(p.read_text())
            out.append(
                {
                    "index": int(p.stem.split("_")[1]),
                    "timestamp": x["timestamp"],
                    "git_commit": x["git_commit"],
                    "git_dirty": x["git_dirty"],
                    "ram_available_bytes": x["machine"]["memory"]["available_bytes"],
                }
            )
        return out

    def failed_attempts(self) -> list[dict]:
        return [
            {k: x.get(k) for k in ("run_id", "session", "status", "stderr_tail")}
            for x in (
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((self.work / "failed").glob("*.json"))
            )
        ]
