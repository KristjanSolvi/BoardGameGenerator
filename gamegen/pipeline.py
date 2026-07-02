"""Orchestrator: runs the full multi-agent loop for one or more games.

Flow per run (see README for the diagram):

  inspiration sampler (no LLM)
    -> designer  ->  spec JSON (strict schema)
    -> rules engineer -> engine.py + test_engine.py
         -> pytest + validator hard checks; failures feed back for repair
            (max limits.repair_rounds)
    -> playtest harness (no LLM) -> playtest_report.json
    -> critic -> ACCEPT, or numbered revisions back to the designer
       (max limits.revision_cycles, each revision re-runs engineering,
        validation and playtesting on the new spec)
    -> novelty checker (logged, never auto-fails)
    -> rulebook writer -> rulebook.md
"""

from __future__ import annotations

import random
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from . import agents
from .backend import make_backend
from .config import Config
from .inspiration import sample_inspiration
from .playtest import run_playtests
from .runlog import RunLog
from .validator import load_engine_class, validate_engine


class RunFailed(Exception):
    """This run could not produce an accepted game (logged, not fatal to
    the batch)."""


def _engineer_until_valid(backend, runlog: RunLog, spec: dict, cfg: Config,
                          revision: int) -> tuple[Path, dict]:
    """Rules engineer + repair loop. Returns (engine_path, validation
    report dict) or raises RunFailed."""
    limits = cfg.limits
    engine_code: Optional[str] = None
    test_code: Optional[str] = None
    feedback: Optional[str] = None

    for attempt in range(int(limits["repair_rounds"]) + 1):
        files = agents.run_rules_engineer(
            backend, runlog, spec,
            format_retries=int(limits["format_retries"]),
            repair_feedback=feedback,
            previous_engine=engine_code,
            previous_tests=test_code,
        )
        engine_code, test_code = files["ENGINE"], files["TESTS"]
        # each attempt gets its own dir so tests can always `import engine`
        stem = f"rev{revision}_attempt{attempt}"
        engine_path = runlog.save_text(f"engine/{stem}/engine.py", engine_code)
        test_path = runlog.save_text(f"engine/{stem}/test_engine.py", test_code)

        report = validate_engine(
            engine_path=engine_path,
            test_path=test_path,
            spec=spec,
            seed=cfg.seed,
            random_playouts=int(cfg.validation["random_playouts"]),
            move_cap=int(cfg.validation["move_cap"]),
            illegal_move_samples=int(cfg.validation["illegal_move_samples"]),
        )
        runlog.save_json(f"engine/{stem}/validation.json", report.as_dict())
        runlog.event("validation", revision=revision, attempt=attempt,
                     ok=report.ok, failed_check=report.failed_check)

        if report.ok:
            return engine_path, report.as_dict()

        feedback = (
            f"Validation failed at check '{report.failed_check}'.\n\n"
            f"{report.failure_message}"
        )
    raise RunFailed(
        f"engine for revision {revision} still failing after "
        f"{limits['repair_rounds']} repair rounds "
        f"(last failure: {report.failed_check})"
    )


def run_one(cfg: Config, run_seed: int, root: Path) -> dict[str, Any]:
    """Execute one full generation run. Returns the run summary dict."""
    runlog = RunLog(root, run_seed)
    backend = make_backend(cfg, observer=runlog.observe_call)
    rng = random.Random(run_seed)
    started = time.time()
    summary: dict[str, Any] = {
        "run_dir": str(runlog.dir),
        "run_seed": run_seed,
        "backend": backend.name,
        "model": cfg.model,
        "status": "failed",
        "revisions": 0,
    }
    runlog.save_json("config_used.json", {**cfg.as_dict(), "run_seed": run_seed})

    try:
        # 1. inspiration (no LLM)
        inspiration = sample_inspiration(rng)
        runlog.save_json("seeds.json", inspiration.as_dict())
        runlog.event("inspiration", **inspiration.as_dict())

        # 2. designer
        spec = agents.run_designer(
            backend, runlog, inspiration,
            format_retries=int(cfg.limits["format_retries"]),
        )
        runlog.save_json("spec_rev0.json", spec)
        summary["game_name"] = spec["name"]

        accepted = False
        engine_path: Optional[Path] = None
        playtest_report: Optional[dict] = None

        for revision in range(int(cfg.limits["revision_cycles"]) + 1):
            summary["revisions"] = revision
            # 3-4. engineer + validate (with repair loop)
            engine_path, validation = _engineer_until_valid(
                backend, runlog, spec, cfg, revision
            )

            # 5. playtest (no LLM)
            engine = load_engine_class(engine_path)()
            playtest_report = run_playtests(engine, cfg.seed, cfg.playtest)
            runlog.save_json(f"playtest_report_rev{revision}.json",
                             playtest_report)

            # 6. critic
            verdict = agents.run_critic(
                backend, runlog, spec, playtest_report,
                format_retries=int(cfg.limits["format_retries"]),
            )
            runlog.save_json(f"critic_rev{revision}.json", verdict)
            runlog.event("critic", revision=revision,
                         verdict=verdict["verdict"],
                         scores={d: verdict["scores"][d]["score"]
                                 for d in agents.CRITIC_DIMENSIONS})
            if verdict["verdict"] == "ACCEPT":
                accepted = True
                break
            if revision == int(cfg.limits["revision_cycles"]):
                break  # out of budget; marked failed below

            # back to the designer with the critic's numbered revisions
            feedback = "\n".join(
                f"{i + 1}. {r}" for i, r in enumerate(verdict["revisions"])
            )
            spec = agents.run_designer(
                backend, runlog, inspiration,
                format_retries=int(cfg.limits["format_retries"]),
                revision_feedback=feedback,
                previous_spec=spec,
            )
            runlog.save_json(f"spec_rev{revision + 1}.json", spec)
            summary["game_name"] = spec["name"]

        if not accepted:
            raise RunFailed(
                f"critic did not accept the game within "
                f"{cfg.limits['revision_cycles']} revision cycles"
            )

        # canonical final artifacts
        runlog.save_json("spec_final.json", spec)
        runlog.save_text("engine/engine.py", Path(engine_path).read_text())
        runlog.save_text("engine/test_engine.py",
                         (Path(engine_path).parent / "test_engine.py").read_text())
        runlog.save_json("playtest_report.json", playtest_report)

        # 7. novelty checker (logged only; never auto-fails the run)
        novelty = agents.run_novelty_checker(
            backend, runlog, spec,
            format_retries=int(cfg.limits["format_retries"]),
        )
        runlog.save_json("novelty_report.json", novelty)

        # 8. rulebook writer
        rulebook = agents.run_rulebook_writer(
            backend, runlog, spec,
            format_retries=int(cfg.limits["format_retries"]),
        )
        runlog.save_text("rulebook.md", rulebook)

        summary["status"] = "accepted"
        summary["novelty_judgment"] = novelty["overall_judgment"]
        summary["headline_metrics"] = playtest_report["headline"]
    except RunFailed as exc:
        summary["status"] = "failed"
        summary["failure_reason"] = str(exc)
    except Exception as exc:
        summary["status"] = "error"
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        runlog.save_text("error_traceback.txt", traceback.format_exc())
    finally:
        summary["duration_seconds"] = round(time.time() - started, 1)
        runlog.save_json("run_summary.json", summary)
    return summary


def generate(cfg: Config, n_runs: int, root: Path) -> list[dict[str, Any]]:
    """Run n_runs independent generations. Run i uses seed cfg.seed + i so
    a batch is reproducible and each run is distinct."""
    summaries = []
    for i in range(n_runs):
        run_seed = cfg.seed + i
        print(f"=== run {i + 1}/{n_runs} (seed {run_seed}) ===", flush=True)
        summary = run_one(cfg, run_seed, root)
        print(f"    {summary['status']}: "
              f"{summary.get('game_name', '<no game>')} "
              f"({summary.get('failure_reason', 'ok')})", flush=True)
        summaries.append(summary)
    return summaries


def replay(run_dir: Path, cfg: Config) -> dict[str, Any]:
    """Re-run validation + playtests on an existing run's final engine.
    Writes replay_playtest_report.json next to the original."""
    engine_path = run_dir / "engine" / "engine.py"
    if not engine_path.exists():
        candidates = sorted((run_dir / "engine").glob("rev*_attempt*/engine.py"))
        if not candidates:
            raise FileNotFoundError(f"no engine found under {run_dir}/engine/")
        engine_path = candidates[-1]
    engine = load_engine_class(engine_path)()
    report = run_playtests(engine, cfg.seed, cfg.playtest)
    out = run_dir / "replay_playtest_report.json"
    import json as _json
    out.write_text(_json.dumps(report, indent=2) + "\n")
    print(f"replay report written to {out}")
    return report
