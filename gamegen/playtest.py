"""Playtest harness (no LLM).

Runs matches against the generated engine through the fixed interface —
no per-game code anywhere here. Two agents:

  * RandomAgent — uniform random legal move.
  * FlatMCAgent — flat Monte Carlo: for each legal move, run K random
    playouts from the successor state and pick the move with the best
    mean outcome. Fully generic (needs no evaluation function), and
    reliably stronger than random on games with any strategic signal,
    which is exactly what the decisiveness metric needs.

Matchups: random vs random, MC vs random (both seats), MC vs MC. The
games are asymmetric, so seat = role: player 0 is always the spec's
role "0" (and moves first). MC-vs-random is run from both seats so each
role's responsiveness to skill is measured separately — an asymmetric
design where only one role rewards skill is broken even if the overall
win rates look fine.
Metrics: game length distribution, per-role win rates, draw rate,
decisiveness (MC win rate vs random, overall and per role), branching
factor stats, termination reasons. Everything lands in
playtest_report.json.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Optional


class RandomAgent:
    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, engine, state, player) -> Any:
        return self.rng.choice(engine.legal_moves(state, player))


class FlatMCAgent:
    name = "flat_mc"

    def __init__(self, rng: random.Random, rollout_budget: int = 96,
                 simulation_move_cap: int = 200):
        # rollout_budget is the TOTAL rollouts per decision, split evenly
        # across candidate moves (min 1 each), so the cost of a decision
        # is bounded regardless of branching factor
        self.rng = rng
        self.rollout_budget = rollout_budget
        self.simulation_move_cap = simulation_move_cap

    def _rollout_value(self, engine, state, player) -> float:
        """Random rollout; +1 win / 0 draw-or-cap / -1 loss for `player`."""
        for _ in range(self.simulation_move_cap):
            if engine.is_terminal(state):
                winner = engine.result(state)["winner"]
                if winner is None:
                    return 0.0
                return 1.0 if winner == player else -1.0
            moves = engine.legal_moves(state, engine.current_player(state))
            state = engine.apply(state, self.rng.choice(moves))
        return 0.0

    def choose(self, engine, state, player) -> Any:
        moves = engine.legal_moves(state, player)
        if len(moves) == 1:
            return moves[0]
        per_move = max(1, self.rollout_budget // len(moves))
        best_move, best_value = None, float("-inf")
        for move in moves:
            succ = engine.apply(state, move)
            total = 0.0
            for _ in range(per_move):
                total += self._rollout_value(engine, succ, player)
            value = total / per_move
            if value > best_value:
                best_move, best_value = move, value
        return best_move


# ----------------------------------------------------------------------
def play_game(engine, agents: dict[int, Any], move_cap: int,
              branching: Optional[list[int]] = None) -> dict[str, Any]:
    """Play one game; agents maps player index -> agent."""
    state = engine.initial_state()
    plies = 0
    while not engine.is_terminal(state):
        if plies >= move_cap:
            return {"winner": None, "reason": "playtest_move_cap", "plies": plies}
        player = engine.current_player(state)
        moves = engine.legal_moves(state, player)
        if branching is not None:
            branching.append(len(moves))
        move = agents[player].choose(engine, state, player)
        state = engine.apply(state, move)
        plies += 1
    result = engine.result(state)
    return {"winner": result["winner"], "reason": str(result.get("reason")),
            "plies": plies}


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    return {
        "mean": round(statistics.mean(lengths_sorted), 2),
        "stdev": round(statistics.stdev(lengths_sorted), 2) if n > 1 else 0.0,
        "min": lengths_sorted[0],
        "p25": lengths_sorted[n // 4],
        "median": lengths_sorted[n // 2],
        "p75": lengths_sorted[(3 * n) // 4],
        "max": lengths_sorted[-1],
    }


def _tally(games: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(games)
    p0 = sum(1 for g in games if g["winner"] == 0)
    p1 = sum(1 for g in games if g["winner"] == 1)
    draws = n - p0 - p1
    reasons: dict[str, int] = {}
    for g in games:
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
    return {
        "games": n,
        "p0_wins": p0,
        "p1_wins": p1,
        "draws": draws,
        "p0_win_rate": round(p0 / n, 3),
        "p1_win_rate": round(p1 / n, 3),
        "draw_rate": round(draws / n, 3),
        "termination_reasons": reasons,
        "length": _length_stats([g["plies"] for g in games]),
    }


def run_playtests(engine, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run all matchups and return the playtest report dict."""
    move_cap = int(cfg["move_cap"])
    mc_kwargs = dict(
        rollout_budget=int(cfg["mc_rollout_budget"]),
        simulation_move_cap=int(cfg["mc_simulation_move_cap"]),
    )

    # -- random vs random ------------------------------------------------
    rng = random.Random(seed ^ 0x9A7E57)
    branching: list[int] = []
    rvr = [
        play_game(engine,
                  {0: RandomAgent(rng), 1: RandomAgent(rng)},
                  move_cap, branching)
        for _ in range(int(cfg["random_vs_random_games"]))
    ]

    # -- MC vs random, both seats (each seat is a distinct role) ----------
    mc_as_p0, mc_as_p1 = [], []
    for i in range(int(cfg["mc_vs_random_games"])):
        rng_g = random.Random(seed ^ 0x3C0001 ^ (i * 2654435761))
        mc_as_p0.append(play_game(
            engine,
            {0: FlatMCAgent(rng_g, **mc_kwargs), 1: RandomAgent(rng_g)},
            move_cap))
        rng_g2 = random.Random(seed ^ 0x3C0002 ^ (i * 2654435761))
        mc_as_p1.append(play_game(
            engine,
            {0: RandomAgent(rng_g2), 1: FlatMCAgent(rng_g2, **mc_kwargs)},
            move_cap))

    # -- MC vs MC ---------------------------------------------------------
    mvm = []
    for i in range(int(cfg["mc_vs_mc_games"])):
        rng_g = random.Random(seed ^ 0x3C0003 ^ (i * 2654435761))
        mvm.append(play_game(
            engine,
            {0: FlatMCAgent(rng_g, **mc_kwargs),
             1: FlatMCAgent(rng_g, **mc_kwargs)},
            move_cap))

    mc_wins = (sum(1 for g in mc_as_p0 if g["winner"] == 0)
               + sum(1 for g in mc_as_p1 if g["winner"] == 1))
    mc_games = len(mc_as_p0) + len(mc_as_p1)

    report = {
        "config": {**cfg, "seed": seed},
        "random_vs_random": _tally(rvr),
        "mc_vs_random": {
            "mc_as_p0": _tally(mc_as_p0),
            "mc_as_p1": _tally(mc_as_p1),
            "decisiveness_mc_win_rate": round(mc_wins / mc_games, 3),
        },
        "mc_vs_mc": _tally(mvm),
        "branching_factor": {
            "mean": round(statistics.mean(branching), 2),
            "max": max(branching),
            "min": min(branching),
            "p90": sorted(branching)[int(len(branching) * 0.9)],
        },
        "headline": {
            "first_player_win_rate_random": _tally(rvr)["p0_win_rate"],
            "first_player_win_rate_mc": _tally(mvm)["p0_win_rate"],
            "draw_rate_random": _tally(rvr)["draw_rate"],
            "draw_rate_mc": _tally(mvm)["draw_rate"],
            "decisiveness_mc_vs_random": round(mc_wins / mc_games, 3),
            "avg_game_length_random": _tally(rvr)["length"]["mean"],
        },
    }
    return report
