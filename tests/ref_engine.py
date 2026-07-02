"""Hand-written reference engine (3x3 alignment placement game) used to
test the validator and playtest harness without any LLM. Deliberately a
known game — it is a test fixture, not pipeline output."""

from __future__ import annotations

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


class IllegalMoveError(ValueError):
    pass


class RefEngine:
    def initial_state(self):
        return (None,) * 9 + (0,)  # 9 cells + side to move

    def current_player(self, state):
        return state[9]

    def legal_moves(self, state, player):
        if self.is_terminal(state) or player != state[9]:
            return []
        return [("place", i) for i in range(9) if state[i] is None]

    def apply(self, state, move):
        if move not in self.legal_moves(state, state[9]):
            raise IllegalMoveError(move)
        _, cell = move
        board = list(state[:9])
        board[cell] = state[9]
        return tuple(board) + (1 - state[9],)

    def _winner(self, state):
        for a, b, c in LINES:
            if state[a] is not None and state[a] == state[b] == state[c]:
                return state[a]
        return None

    def is_terminal(self, state):
        return self._winner(state) is not None or all(
            cell is not None for cell in state[:9]
        )

    def result(self, state):
        winner = self._winner(state)
        if winner is not None:
            return {"winner": winner, "reason": "alignment"}
        return {"winner": None, "reason": "board_full"}

    def mirror_state(self, state):
        swap = {None: None, 0: 1, 1: 0}
        return tuple(swap[c] for c in state[:9]) + (1 - state[9],)

    def render(self, state):
        symbols = {None: ".", 0: "X", 1: "O"}
        rows = [
            " ".join(symbols[state[r * 3 + c]] for c in range(3))
            for r in range(3)
        ]
        return "\n".join(rows) + f"\nto move: {state[9]}"
