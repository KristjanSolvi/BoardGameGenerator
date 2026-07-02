"""Inspiration sampler (no LLM).

Samples 2-3 mechanic seeds from a curated list to force diversity across
runs, plus the forbidden list of famous abstract games that every agent
receives. Deterministic given the run RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

MECHANIC_SEEDS: dict[str, list[str]] = {
    "movement_style": [
        "single-step moves to adjacent cells",
        "sliding any distance along a line until blocked",
        "fixed-length leaps that may pass over pieces",
        "movement distance determined by the number of friendly pieces in the moving piece's line or group",
        "pieces move as connected groups rather than individually",
        "movement that grows or shrinks with the piece's distance from its starting side",
        "mandatory continued movement (momentum) once a piece starts moving",
        "pieces may only move toward or away from specific board features",
    ],
    "capture_style": [
        "no captures at all - pieces are never removed",
        "replacement capture by moving onto an enemy piece",
        "custodial capture by flanking an enemy piece on two opposite sides",
        "leap capture by jumping over an enemy piece",
        "capture by enclosure of a whole group",
        "displacement: enemy pieces are pushed rather than removed",
        "conversion: enemy pieces change ownership instead of leaving the board",
        "self-capture allowed as a deliberate sacrifice mechanism",
    ],
    "action_type": [
        "pure placement - pieces enter but never move",
        "pure movement - all pieces start on the board",
        "each turn choose: place a new piece OR move an existing one",
        "placement phase followed by a movement phase",
        "each turn: move one piece, then optionally trigger its special power",
        "pieces are placed onto stacks and only stack tops act",
    ],
    "board_topology": [
        "square grid with orthogonal adjacency",
        "square grid with orthogonal and diagonal adjacency",
        "hexagonal grid",
        "a graph of nodes and edges that is not a regular grid",
        "square grid where some cells are special (marked) and behave differently",
        "a board whose cells can be permanently removed or blocked during play",
    ],
    "goal_type": [
        "connection: link two opposite sides or regions",
        "alignment: form a specific pattern or line of your pieces",
        "elimination: reduce the opponent below a threshold of pieces",
        "territory: control the majority of cells or regions at the end",
        "race: be first to bring pieces to a target zone",
        "immobilization: leave the opponent without a legal move",
        "capture a fixed number of enemy pieces",
        "be the first to build a specified static structure",
    ],
}

FORBIDDEN_GAMES: list[str] = [
    "Chess", "Shogi", "Xiangqi", "Go", "Gomoku", "Renju", "Connect Four",
    "Checkers/Draughts", "International Draughts", "Othello/Reversi", "Hex",
    "Havannah", "TwixT", "Y", "Nine Men's Morris", "Tic-tac-toe",
    "Pente", "Tafl games (Hnefatafl)", "Amazons", "Breakthrough",
    "Lines of Action", "Onitama", "Santorini", "Hive", "GIPF", "TZAAR",
    "DVONN", "YINSH", "ZERTZ", "PUNCT", "Abalone", "Fanorona", "Konane",
    "Camelot", "Halma/Chinese Checkers", "Quoridor", "Isolation",
    "Quarto", "Pentago", "Tak", "Arimaa", "Backgammon", "Mancala/Oware",
]


@dataclass
class Inspiration:
    seeds: dict[str, str]          # category -> sampled seed
    forbidden_games: list[str]

    def seeds_text(self) -> str:
        return "\n".join(f"- {cat}: {seed}" for cat, seed in self.seeds.items())

    def forbidden_text(self) -> str:
        return ", ".join(self.forbidden_games)

    def as_dict(self) -> dict:
        return {"seeds": self.seeds, "forbidden_games": self.forbidden_games}


def sample_inspiration(rng: random.Random) -> Inspiration:
    """Sample 2-3 mechanic seeds from distinct categories.

    goal_type is always included (a game needs a goal); the remaining 1-2
    seeds come from other categories, so different runs are pushed toward
    genuinely different design regions.
    """
    n_extra = rng.choice([1, 2])
    other_categories = [c for c in MECHANIC_SEEDS if c != "goal_type"]
    chosen = ["goal_type"] + rng.sample(other_categories, n_extra)
    seeds = {cat: rng.choice(MECHANIC_SEEDS[cat]) for cat in sorted(chosen)}
    return Inspiration(seeds=seeds, forbidden_games=list(FORBIDDEN_GAMES))
