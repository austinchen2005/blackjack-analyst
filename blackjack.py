"""blackjack_sim.py – Monte‑Carlo learning *plus* evaluation utilities
=======================================================================
* Learns EVs for hit/stand with ε‑greedy exploration.
* Provides **TableStrategy** that freezes the best actions after learning.
* `analyze_strategy()` runs any Strategy object with casino‑style verbose
  output and returns a stats dict (wins / losses / pushes / net PnL).

Run this file directly to:
1. Learn for 200 k rounds (takes a few seconds).
2. Evaluate the learned table for 100 verbose rounds.
3. Print summary stats.
4. Plot the combined EV table.
"""

from __future__ import annotations

import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Card / Deck helpers
# ----------------------------------------------------------------------------
Card = namedtuple("Card", ["rank", "suit"])
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]


def card_value(card: Card) -> int:
    if card.rank == "A":
        return 1
    if card.rank in {"T", "J", "Q", "K"}:
        return 10
    return int(card.rank)


class Deck:
    """6‑deck shoe reshuffling at 25 % penetration."""

    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self._reshuffle()

    def _reshuffle(self):
        self.cards = [Card(r, s) for r in RANKS for s in SUITS] * self.num_decks
        random.shuffle(self.cards)

    def maybe_reshuffle(self):
        if len(self.cards) < 0.25 * 52 * self.num_decks:
            self._reshuffle()

    def deal(self) -> Card:
        if not self.cards:
            self._reshuffle()
        return self.cards.pop()


class Hand:
    """Hard‑total hand helper (aces upgraded when safe)."""

    def __init__(self):
        self.cards: List[Card] = []

    def add(self, card: Card):
        self.cards.append(card)

    def total(self) -> int:
        total = sum(card_value(c) for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == "A")
        while aces and total + 10 <= 21:
            total += 10
            aces -= 1
        return total

    def is_soft(self) -> bool:
        low = sum(card_value(c) for c in self.cards)
        return any(c.rank == "A" for c in self.cards) and self.total() != low

    def is_bust(self) -> bool:
        return self.total() > 21

    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.total() == 21

    def ranks(self):
        return "".join(c.rank for c in self.cards)


# ----------------------------------------------------------------------------
# Dealer – stands on soft‑17
# ----------------------------------------------------------------------------
class Dealer:
    def __init__(self):
        self.hand = Hand()

    def play(self, deck: Deck):
        while True:
            val, soft = self.hand.total(), self.hand.is_soft()
            if val < 17 or (val == 17 and soft):
                self.hand.add(deck.deal())
            else:
                break


# ----------------------------------------------------------------------------
# EV table + learning strategy (ε‑greedy)
# ----------------------------------------------------------------------------
Action = Literal["hit", "stand"]
State = Tuple[int, str]  # (hard total, dealer up‑rank)


@dataclass
class EVStats:
    count: int = 0
    pnl_total: float = 0.0

    def update(self, pnl: float):
        self.count += 1
        self.pnl_total += pnl

    def ev(self) -> float:
        return self.pnl_total / self.count if self.count else 0.0


class EVTable:
    def __init__(self):
        self.data: Dict[State, Dict[Action, EVStats]] = {}

    def _stats(self, state: State, action: Action) -> EVStats:
        return self.data.setdefault(state, {"hit": EVStats(), "stand": EVStats()})[action]

    def update(self, state: State, action: Action, pnl: float):
        self._stats(state, action).update(pnl)

    def ev(self, state: State, action: Action) -> float:
        return self._stats(state, action).ev()

    def best_action(self, state: State) -> Action:
        return "hit" if self.ev(state, "hit") >= self.ev(state, "stand") else "stand"


class LearningStrategy:
    """ε‑greedy strategy that refers to EVTable while learning."""

    def __init__(self, ev_table: EVTable, episodes: int, eps_start: float = 1.0, eps_end: float = 0.05):
        self.ev_table = ev_table
        self.episodes = episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.episode = 0

    def epsilon(self) -> float:
        frac = self.episode / self.episodes
        return self.eps_start * (1 - frac) + self.eps_end * frac

    def decide(self, hand: Hand, up: Card) -> Action:  # noqa: D401
        state = (hand.total(), up.rank)
        if random.random() < self.epsilon() or state not in self.ev_table.data:
            return random.choice(["hit", "stand"])
        return self.ev_table.best_action(state)

    def advance(self):
        self.episode += 1


# ----------------------------------------------------------------------------
# Single learning round (returns trajectory, pnl)
# ----------------------------------------------------------------------------

def play_learning_round(deck: Deck, strat: LearningStrategy, bet: float = 1.0):
    deck.maybe_reshuffle()
    dealer, player = Dealer(), Hand()
    player.add(deck.deal()); dealer.hand.add(deck.deal())
    player.add(deck.deal()); dealer.hand.add(deck.deal())
    upcard = dealer.hand.cards[0]

    # naturals
    if player.is_blackjack() or dealer.hand.is_blackjack():
        if player.is_blackjack() and dealer.hand.is_blackjack():
            pnl = 0.0
        elif player.is_blackjack():
            pnl = 1.5 * bet
        else:
            pnl = -bet
        strat.advance(); return [], pnl

    traj: List[Tuple[State, Action]] = []
    while True:
        if player.is_bust():
            pnl = -bet
            strat.advance(); break
        state: State = (player.total(), upcard.rank)
        action = strat.decide(player, upcard)
        traj.append((state, action))
        if action == "stand":
            dealer.play(deck)
            pt, dt = player.total(), dealer.hand.total()
            if dealer.hand.is_bust() or pt > dt:
                pnl = bet
            elif pt < dt:
                pnl = -bet
            else:
                pnl = 0.0
            strat.advance(); break
        else:
            player.add(deck.deal())

    return traj, pnl


# ----------------------------------------------------------------------------
# Learning driver
# ----------------------------------------------------------------------------

def simulate_learning(rounds: int = 200_000, seed: int | None = None) -> EVTable:
    if seed is not None:
        random.seed(seed)
    deck, table = Deck(), EVTable()
    strat = LearningStrategy(table, episodes=rounds)
    for _ in range(rounds):
        traj, pnl = play_learning_round(deck, strat)
        for st, act in traj:
            table.update(st, act, pnl)
    return table


# ----------------------------------------------------------------------------
# Strategy derived from learned table
# ----------------------------------------------------------------------------
class TableStrategy:
    """Uses a frozen mapping from EVTable (hard total 4‑20)."""

    def __init__(self, ev_table: EVTable):
        self.best = {state: ev_table.best_action(state) for state in ev_table.data}

    def decide(self, hand: Hand, up: Card) -> Action:  # noqa: D401
        state = (hand.total(), up.rank)
        return self.best.get(state, "stand")  # default to stand if unseen


# ----------------------------------------------------------------------------
# Analysis / evaluation with verbose play‑by‑play
# ----------------------------------------------------------------------------

def analyze_strategy(strategy, *, num_games: int = 100, bet: float = 1.0, seed: int | None = None, verbose: bool = True):
    if seed is not None:
        random.seed(seed)
    deck = Deck()

    wins = losses = pushes = 0
    net_pnl = 0.0

    for g in range(1, num_games + 1):
        deck.maybe_reshuffle()
        dealer, player = Dealer(), Hand()
        player.add(deck.deal()); dealer.hand.add(deck.deal())
        player.add(deck.deal()); dealer.hand.add(deck.deal())
        upcard = dealer.hand.cards[0]

        if verbose:
            print(f"--- Hand {g} ---")
            print(f"Dealer upcard: {upcard.rank}")
            print(f"Player cards: {player.ranks()}")

        # naturals
        if player.is_blackjack() or dealer.hand.is_blackjack():
            if player.is_blackjack() and dealer.hand.is_blackjack():
                outcome, pnl = 0, 0.0
                result_msg = "Both blackjack – Push"
            elif player.is_blackjack():
                outcome, pnl = 1, 1.5 * bet
                result_msg = "Player blackjack – Win 3:2"
            else:
                outcome, pnl = -1, -bet
                result_msg = "Dealer blackjack – Player loses"
            if verbose:
                print(result_msg, "\n")
            
        else:
            # player decisions
            while True:
                if player.is_bust():
                    outcome, pnl = -1, -bet
                    if verbose:
                        print("Player busts\n")
                    break
                action = strategy.decide(player, upcard)
                if action == "hit":
                    if verbose:
                        print("Hit")
                    player.add(deck.deal())
                    if verbose:
                        print(f"Player cards: {player.ranks()}")
                    continue
                else:
                    if verbose:
                        print("Stand")
                    break

            if not player.is_bust():
                dealer.play(deck)
                pt, dt = player.total(), dealer.hand.total()
                if verbose:
                    print(f"Dealer cards: {dealer.hand.ranks()}")
                    print(f"{dt} vs {pt}")

                if dealer.hand.is_bust() or pt > dt:
                    outcome, pnl = 1, bet
                    if verbose:
                        print("Player wins\n")
                elif pt < dt:
                    outcome, pnl = -1, -bet
                    if verbose:
                        print("Player loses\n")
                else:
                    outcome, pnl = 0, 0.0
                    if verbose:
                        print("Push\n")

        net_pnl += pnl
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            pushes += 1

    if verbose:
        print("==== Final Stats ====")
        print(f"Games: {num_games}")
        print(f"Wins: {wins} ({wins/num_games:.2%})")
        print(f"Losses: {losses} ({losses/num_games:.2%})")
        print(f"Pushes: {pushes} ({pushes/num_games:.2%})")
        print(f"Net PnL: {net_pnl:+.2f}\n")

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "net_pnl": net_pnl,
        "win_rate": wins / num_games,
        "loss_rate": losses / num_games,
        "push_rate": pushes / num_games,
    }


# ----------------------------------------------------------------------------
# EV table plotting for inspection
# ----------------------------------------------------------------------------

def plot_ev_table(ev_table: EVTable, *, figsize: Tuple[int, int] = (14, 8), font_size: int = 10):
    rows = list(range(20, 3, -1))
    cols = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]

    cell_text = []
    for total in rows:
        sub = []
        for up in cols:
            state = (total, up)
            evh, evs = ev_table.ev(state, "hit"), ev_table.ev(state, "stand")
            best = "H" if evh >= evs else "S"
            sub.append(f"{best}\nH:{evh:+.2f}\nS:{evs:+.2f}")
        cell_text.append(sub)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    for _, cell in tbl.get_celld().items():
        cell.set_height(1 / (len(rows) + 1) * 1.5)
        cell.set_width(1 / (len(cols) + 1) * 1.2)
    plt.title("Hit/Stand EVs – best action in first line")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Demo when run standalone
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Learning optimal hit/stand table …")
    ev_tbl = simulate_learning(200_000, seed=42)
    print("Learning complete. Evaluating fixed strategy on 100 hands:\n")
    table_strategy = TableStrategy(ev_tbl)
    analyze_strategy(table_strategy, num_games=1000, verbose=True, seed=123)
    plot_ev_table(ev_tbl)
    plt.show()
