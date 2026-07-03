# Game Mechanisms Overview (Focus: 2-Player Games)

A research overview of the core mechanisms behind playable/generated games,
organized by category and then synthesized across all of them, with a focus on
2-player play.

## Cross-Cutting Mechanisms (the "atoms" most games are built from)

Almost every game combines a handful of these primitives. This is the vocabulary
a generator should reason in.

| Mechanism | What it does | Typical use |
|---|---|---|
| Turn structure | Alternating / simultaneous / real-time | Defines pacing; 2-player is usually strict alternation |
| Movement | Piece relocation on a space (grid, track, graph) | Chess, Ludo, Checkers |
| Capture / elimination | Remove opponent's pieces or reduce a resource | Chess, Checkers, Go |
| Set collection | Gather combinations for value | Rummy, Poker hands |
| Pattern building | Form a target configuration (line, area) | Tic-tac-toe, Connect Four, Gomoku |
| Area control / territory | Own more of the board than opponent | Go, Reversi/Othello |
| Resource management | Acquire / spend limited assets | Card economies, worker placement |
| Hidden information | Some state known to only one player | Card hands, Battleship, Stratego |
| Randomness | Dice, shuffled deck, spinner | Ludo, Backgammon, Solitaire |
| Push-your-luck | Optional risk for reward | Blackjack, dice games |
| Bluffing / deduction | Infer or misrepresent hidden state | Poker, Mastermind, social deduction |
| Tempo / initiative | Who is forced to react | Chess, combat games |
| Race | First to a goal / end position | Ludo, Backgammon |
| Tableau / engine building | Cards/tiles that generate ongoing effects | Card engine games |

Two dimensions matter most for classifying a *2-player* game:

1. **Information** — perfect (chess, Go) vs. hidden (cards, Battleship).
2. **Chance** — deterministic (chess) vs. stochastic (Ludo, card draw).

Those two axes give a clean 2x2 that most 2-player games fall into.

---

## By Category

### 1. Playing Cards (deck of cards)

Core substrate: a shuffled deck = built-in **randomness + hidden information**.

Primary mechanisms:

- **Trick-taking** — each player plays a card, highest/trump wins the "trick"
  (2-player Euchre, Piquet, German Whist). Uses follow-suit rules, trump
  hierarchy, trick counting.
- **Set collection / melding** — form runs and sets (Gin Rummy, Cribbage).
- **Shedding** — be first to empty your hand (Speed, 2-player Crazy Eights).
- **Matching / building** — sequence cards onto foundations (Solitaire/Patience;
  Spit/Nertz is a racing 2-player version).
- **Comparing / betting** — hand rankings + wagering (heads-up Poker).
- **Draw/discard economy** — the near-universal loop: draw, play/meld, discard.

Signature 2-player card mechanisms: **hand management**, **hidden hands**,
**card counting/memory**, and **drafting** (offer/pick from a shared row).

### 2. Board Games (dedicated board + pieces)

Substrate: a spatial board (grid, track, or graph) + tokens.

- **Grid movement + capture** (Chess, Checkers/Draughts): perfect information,
  deterministic. Piece-specific movement, capture, promotion, check/checkmate.
- **Territory / area control** (Go, Othello/Reversi): claim space, win by
  majority. Surrounding, flipping, liberties.
- **Alignment / connection** (Connect Four, Gomoku, Hex): build a line or a
  connected path; Hex is a pure 2-player connection game with no draws.
- **Roll-and-move racing** (Ludo, Backgammon, Snakes & Ladders): dice-driven
  progress along a track. Backgammon adds bearing off, hitting/blocking, and the
  doubling cube (a stake-raising bluff mechanism).
- **Blocking / spatial denial** (Blokus Duo, Quoridor): restrict opponent's
  options via placement.

Signature mechanisms: **movement**, **capture**, **area majority**,
**pattern/line building**, **blocking**.

### 3. Simple Video Games (grid/arcade logic)

The same primitives as above plus:

- **Real-time input** instead of turns (reaction, timing).
- **Collision detection** (Snake, Pong — Pong being the archetypal 2-player
  video game: physics + reaction).
- **State machines** (enemy/AI behavior).
- **Score maximization** vs. **survival** win conditions.
- **Procedural/random spawning** (the video-game analog of a shuffle).

Grid-based simple video games (2048, Minesweeper, match-3) reuse **pattern
matching**, **cascading**, and **hidden information + deduction** (Minesweeper is
pure logical deduction).

### 4. Social / Strategy Games (2-player negotiation & deduction)

Here the mechanism *is* the psychology:

- **Bluffing** — represent false state (Poker, Liar's Dice, Coup-style).
- **Deduction** — narrow down hidden truth (Mastermind, Battleship, Guess Who?).
- **Simultaneous selection / mind-reading** — both choose secretly, reveal
  together (Rock-Paper-Scissors, Morra, matching pennies).
- **Negotiation / trading** — less central in 2-player (no coalitions), but
  threats and commitments matter.
- **Yomi (reading the opponent)** — anticipating and countering intent; the core
  of RPS-style and fighting-game metagames.

Signature 2-player social mechanisms: **hidden role/state, bluff, deduction, and
simultaneous reveal** — the mechanisms that turn a solvable game into a
psychological one.

---

## Overall Synthesis (what actually drives 2-player games)

Stripped down, **2-player games are generated by choosing points along a few
axes and then attaching a win condition:**

1. **Board/space type** — none · grid · track · graph · shared card row
2. **Information** — perfect vs. hidden
3. **Chance** — deterministic vs. random (dice/deck)
4. **Core verb** — move · capture · place · match · collect · race · guess · bet
5. **Win condition** — eliminate · align N-in-a-row · control majority · reach
   goal first · empty hand · reach score · deduce secret

Most common **primitive combinations** in 2-player design:

- **Move + capture + perfect info + deterministic** → abstract strategy family
  (chess, checkers).
- **Place + align/connect** → tic-tac-toe → Connect Four → Gomoku → Hex (a
  naturally scalable difficulty ladder, useful for a generator).
- **Roll + race + track** → Ludo, Backgammon.
- **Draw/discard + set collection + hidden hand** → the card-game family.
- **Secret state + guess/deduce** → Battleship, Mastermind, Guess Who.
- **Simultaneous secret choice + counter** → RPS and its extensions.

**Design levers that recur everywhere:** catch-up/comeback mechanisms,
tempo/initiative, the risk–reward (push-your-luck) dial, and the
information-vs-chance balance that determines how much *skill* vs. *luck* a game
expresses.

---

## Practical Takeaway for a Game Generator

Game creation can be treated as **sampling the 5-axis space**
(space × information × chance × verb × win condition), because the vast majority
of the games above are recombinations of the same ~15 primitive mechanisms.
