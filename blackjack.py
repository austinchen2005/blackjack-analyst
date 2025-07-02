"""blackjack_sim.py – Monte‑Carlo & Deterministic Grid Learning
================================================================
This script now contains **two** independent learners:

1. **simulate_learning(**`rounds=…`**)** – the previous ε‑greedy Monte‑Carlo
   learner that draws fully random shoes (kept **unchanged**).
2. **simulate_grid_learning(**`rounds=…`**)** – **NEW**.  Walks the hard‑total
   grid deterministically: starting at 20 vs 2 → hit, 20 vs 2 → stand, 20 vs 3,
   … all the way to 4 vs A, then repeats until the requested number of hands is
   reached (default 200 000). Each state is evaluated twice per cycle (hit then
   stand), so the 340‑hand cycle repeats ⌈rounds/340⌉ times.

*No previous functions were removed.*  All evaluation, plotting, and
`TableStrategy` remain intact.
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


def _value_to_rank(val: int) -> str:
    # Map values to valid card ranks
    rank_map = {
        1: "A",
        2: "2", 
        3: "3",
        4: "4", 
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "T"
    }
    return rank_map.get(val, str(val))


class Deck:
    """n‑deck shoe reshuffling at 25 % penetration."""

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


# ----------------------------------------------------------------------------
# Hand + Dealer
# ----------------------------------------------------------------------------
class Hand:
    def __init__(self, total: int = 0):
        if total == 0:
            self.cards: List[Card] = []
            self._total = 0
        else:
            # Create two cards that sum to the total
            if total <= 10:
                # For totals 4-10, use 2 and (total-2)
                v1, v2 = 2, total - 2
            else:
                # For totals 11-20, use 10 and (total-10)
                v1, v2 = 10, total - 10
            
            self.cards: List[Card] = [Card(_value_to_rank(v1), "♠"), Card(_value_to_rank(v2), "♥")]
            self._total = total


    def add(self, card: Card):
        self.cards.append(card)
        
        # Recalculate total from scratch to handle aces properly
        low_total = sum(card_value(c) for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == "A")
        
        # Try to use aces as 11 when possible
        total = low_total
        while aces > 0 and total + 10 <= 21:
            total += 10
            aces -= 1
        
        self._total = total


    def total(self) -> int:
        return self._total

    def is_soft(self):
        low = sum(card_value(c) for c in self.cards)
        return any(c.rank == "A" for c in self.cards) and self.total() != low

    def is_bust(self):
        return self.total() > 21

    def is_blackjack(self):
        return len(self.cards) == 2 and self.total() == 21

    def ranks(self):
        return "".join(c.rank for c in self.cards)


class Dealer:
    """Stands on soft‑17 (typical shoe rule)."""

    def __init__(self):
        self.hand = Hand()

    def play(self, deck: Deck):
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 20:  # Safety check to prevent infinite loops
                print(f"WARNING: Too many loops in dealer.play()")
                break
                
            val, soft = self.hand.total(), self.hand.is_soft()
            if val < 17 or (val == 17 and soft):
                self.hand.add(deck.deal())
            else:
                break


# ----------------------------------------------------------------------------
# EV‑tracking table (unchanged)
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

    def _stats(self, state: State, action: Action):
        return self.data.setdefault(state, {"hit": EVStats(), "stand": EVStats()})[action]

    def update(self, state: State, action: Action, pnl: float):
        self._stats(state, action).update(pnl)

    def ev(self, state: State, action: Action):
        return self._stats(state, action).ev()

    def best_action(self, state: State):
        return "hit" if self.ev(state, "hit") >= self.ev(state, "stand") else "stand"


# ----------------------------------------------------------------------------
# 1) Previous Monte‑Carlo learner (kept intact)
# ----------------------------------------------------------------------------
class LearningStrategy:
    def __init__(self, ev_table: EVTable, episodes: int, eps_start=1.0, eps_end=0.05):
        self.ev_table = ev_table
        self.episodes = episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.episode = 0

    def epsilon(self):
        frac = self.episode / self.episodes
        return self.eps_start * (1 - frac) + self.eps_end * frac

    def decide(self, hand: Hand, up: Card):
        state = (hand.total(), up.rank)
        if random.random() < self.epsilon() or state not in self.ev_table.data:
            return random.choice(["hit", "stand"])
        return self.ev_table.best_action(state)

    def advance(self):
        self.episode += 1


def _play_learning_round(deck: Deck, strat: LearningStrategy, bet: float = 1.0):
    deck.maybe_reshuffle()
    dealer, player = Dealer(), Hand()
    player.add(deck.deal()); dealer.hand.add(deck.deal())
    player.add(deck.deal()); dealer.hand.add(deck.deal())
    up = dealer.hand.cards[0]

    if player.is_blackjack() or dealer.hand.is_blackjack():
        pnl = 0 if player.is_blackjack() and dealer.hand.is_blackjack() else (1.5 if player.is_blackjack() else -1)
        strat.advance(); return [], pnl

    traj: List[Tuple[State, Action]] = []
    while True:
        if player.is_bust():
            pnl = -1; strat.advance(); break
        state: State = (player.total(), up.rank)
        act = strat.decide(player, up)
        traj.append((state, act))
        if act == "stand":
            dealer.play(deck)
            p, d = player.total(), dealer.hand.total()
            pnl = 1 if dealer.hand.is_bust() or p > d else (-1 if p < d else 0)
            strat.advance(); break
        player.add(deck.deal())
    return traj, pnl


def simulate_learning(rounds: int = 200_000, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    deck, table = Deck(), EVTable()
    strat = LearningStrategy(table, episodes=rounds)
    for _ in range(rounds):
        traj, pnl = _play_learning_round(deck, strat)
        for st, act in traj:
            table.update(st, act, pnl)
    return table

# ----------------------------------------------------------------------------
# 2) NEW deterministic grid learner
# ----------------------------------------------------------------------------
_totals_grid = list(range(20, 3, -1))  # 20 down to 4
_up_grid = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]
_actions_cycle = ["hit", "stand"]
_cycle_len = len(_totals_grid) * len(_up_grid) * 2  # 340
_soft_totals_grid = list(range(20, 11, -1))  # 20 down to 12


def _construct_player_hand(total: int) -> List[Card]:
    """Return two non‑Ace cards that sum to *total* (hard)."""
    for v1 in range(2, 11):
        v2 = total - v1
        if 2 <= v2 <= 10:
            return [Card(_value_to_rank(v1), "♠"), Card(_value_to_rank(v2), "♥")]
    # Fallback (shouldn't happen for totals 4‑20)
    return [Card("T", "♠"), Card(_value_to_rank(total - 10), "♥")]


def _play_fixed_state(total: int, up_rank: str, action: Action, ev_table: EVTable, bet: float = 1.0):
    deck = Deck()
    player = Hand(total=total)
    dealer = Dealer()
    dealer.hand.add(Card(up_rank, "♦"))
    dealer.hand.add(deck.deal())  # random hole card

    # Execute chosen first action
    if action == "hit":
        player.add(deck.deal())
        if player.is_bust():
            return -bet
        
        # Continue playing based on EV table
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 10:  # Safety check to prevent infinite loops
                print(f"WARNING: Too many loops in _play_fixed_state for total={total}, up_rank={up_rank}, action={action}")
                break
                
            # Look up best action from EV table for current state
            current_state = (player.total(), up_rank)
            if current_state in ev_table.data:
                best_action = ev_table.best_action(current_state)
            else:
                # If state not in table yet, default to stand
                best_action = "stand"
            
            if best_action == "stand":
                break
            elif best_action == "hit":
                player.add(deck.deal())
                if player.is_bust():
                    return -bet
            else:
                # If any other action, default to stand
                break
    elif action == "stand":
        # For stand action, do nothing - player stands immediately
        pass

    dealer.play(deck)
    p, d = player.total(), dealer.hand.total()
    if dealer.hand.is_bust() or p > d:
        return bet
    if p < d:
        return -bet
    return 0.0


def _play_fixed_state_soft(total: int, up_rank: str, action: Action, ev_table: EVTable, bet: float = 1.0):
    deck = Deck()
    player = Hand()
    
    # Create a soft hand with the given total
    if total == 12:
        # A,A = soft 12 (1+1, but Aces can be 1 or 11)
        player.add(Card("A", "♠"))
        player.add(Card("A", "♥"))
    else:
        # A + (total-11) = soft total
        # For example: A+2=soft 13, A+3=soft 14, etc.
        second_card_value = total - 11
        player.add(Card("A", "♠"))
        player.add(Card(_value_to_rank(second_card_value), "♥"))
    
    dealer = Dealer()
    dealer.hand.add(Card(up_rank, "♦"))
    dealer.hand.add(deck.deal())  # random hole card

    # Execute chosen first action
    if action == "hit":
        player.add(deck.deal())
        if player.is_bust():
            return -bet
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 10:  # Safety check to prevent infinite loops
                print(f"WARNING: Too many loops in _play_fixed_state_soft for total={total}, up_rank={up_rank}, action={action}")
                break
                
            current_state = (player.total(), up_rank)
            if current_state in ev_table.data:
                best_action = ev_table.best_action(current_state)
            else:
                best_action = "stand"
            if best_action == "stand":
                break
            elif best_action == "hit":
                player.add(deck.deal())
                if player.is_bust():
                    return -bet
            else:
                break
    elif action == "stand":
        pass

    dealer.play(deck)
    p, d = player.total(), dealer.hand.total()
    if dealer.hand.is_bust() or p > d:
        return bet
    if p < d:
        return -bet
    return 0.0


def simulate_grid_learning(rounds: int = 200_000, seed: int | None = None):
    """Sweeps hard and soft total grids, distributing rounds evenly across all cells."""
    if seed is not None:
        random.seed(seed)
    print(f"Starting grid learning with {rounds} rounds...")
    
    hard_table = EVTable()
    soft_table = EVTable()
    num_hard_cells = len(_totals_grid) * len(_up_grid) * 2
    num_soft_cells = len(_soft_totals_grid) * len(_up_grid) * 2
    total_cells = num_hard_cells + num_soft_cells
    
    print(f"Hard cells: {num_hard_cells}, Soft cells: {num_soft_cells}, Total cells: {total_cells}")

    # Distribute rounds as evenly as possible
    rounds_per_cell = rounds // total_cells
    extra = rounds % total_cells
    print(f"Rounds per cell: {rounds_per_cell}, Extra rounds: {extra}")

    # Interweave hard and soft totals
    print("Processing hard and soft totals (interleaved)...")
    
    # Create lists of all cells to process
    hard_cells = []
    for n in range(num_hard_cells):
        total_idx = n // (len(_up_grid) * 2)
        rem = n % (len(_up_grid) * 2)
        up_idx = rem // 2
        act_idx = rem % 2
        total = _totals_grid[total_idx]
        up = _up_grid[up_idx]
        act: Action = _actions_cycle[act_idx]
        hard_cells.append((total, up, act, "hard"))
    
    soft_cells = []
    for n in range(num_soft_cells):
        total_idx = n // (len(_up_grid) * 2)
        rem = n % (len(_up_grid) * 2)
        up_idx = rem // 2
        act_idx = rem % 2
        total = _soft_totals_grid[total_idx]
        up = _up_grid[up_idx]
        act: Action = _actions_cycle[act_idx]
        soft_cells.append((total, up, act, "soft"))
    
    # Interleave the cells
    all_cells = []
    max_len = max(len(hard_cells), len(soft_cells))
    for i in range(max_len):
        if i < len(hard_cells):
            all_cells.append(hard_cells[i])
        if i < len(soft_cells):
            all_cells.append(soft_cells[i])
    
    # Process all cells in interleaved order
    for n, (total, up, act, cell_type) in enumerate(all_cells):
        if n % 100 == 0:  # Print progress every 100 cells
            print(f"Cell {n}/{len(all_cells)} ({cell_type})")
        
        reps = rounds_per_cell + (1 if n < extra else 0)
        for _ in range(reps):
            if cell_type == "hard":
                pnl = _play_fixed_state(total, up, act, hard_table)
                hard_table.update((total, up), act, pnl)
            else:  # soft
                pnl = _play_fixed_state_soft(total, up, act, soft_table)
                soft_table.update((total, up), act, pnl)

    print("Grid learning complete!")
    return hard_table, soft_table

# ----------------------------------------------------------------------------
# Strategy & evaluation (UNCHANGED from earlier)
# ----------------------------------------------------------------------------
def _normalize_rank(rank: str) -> str:
    """Convert J, Q, K to T for consistency with EV table."""
    if rank in {"J", "Q", "K"}:
        return "T"
    return rank


class TableStrategy:
    def __init__(self, ev_table: EVTable):
        self.best = {st: ev_table.best_action(st) for st in ev_table.data}

    def decide(self, hand: Hand, up: Card):
        # Safety check: never hit on 21 or higher
        if hand.total() >= 21:
            return "stand"
        
        normalized_rank = _normalize_rank(up.rank)
        return self.best.get((hand.total(), normalized_rank), "stand")


def analyze_strategy(strategy, *, num_games: int = 100, bet: float = 1.0, seed: int | None = None, verbose: bool = True):
    if seed is not None:
        random.seed(seed)
    deck = Deck()
    wins = losses = pushes = 0
    net = 0.0
    for i in range(1, num_games + 1):
        deck.maybe_reshuffle()
        player = Hand(); dealer = Dealer()
        player.add(deck.deal()); dealer.hand.add(deck.deal())
        player.add(deck.deal()); dealer.hand.add(deck.deal())
        up = dealer.hand.cards[0]
        if verbose:
            print(f"--- Hand {i} ---\nDealer upcard: {up.rank}\nPlayer cards: {player.ranks()}")
        # naturals quick‑exit
        if player.is_blackjack() or dealer.hand.is_blackjack():
            if player.is_blackjack() and dealer.hand.is_blackjack():
                outcome, pnl = 0, 0.0
                msg = "Both blackjack – Push"
            elif player.is_blackjack():
                outcome, pnl = 1, 1.5 * bet; msg = "Player blackjack – Win 3:2"
            else:
                outcome, pnl = -1, -bet; msg = "Dealer blackjack – Player loses"
            if verbose: print(msg, "\n")
        else:
            # player decisions
            while True:
                if player.is_bust():
                    outcome, pnl = -1, -bet
                    if verbose: print("Player busts\n")
                    break
                act = strategy.decide(player, up)
                if act == "hit":
                    if verbose: print("Hit")
                    player.add(deck.deal())
                    if verbose: print(f"Player cards: {player.ranks()}")
                else:
                    if verbose: print("Stand")
                    break
            if not player.is_bust():
                dealer.play(deck)
                p, d = player.total(), dealer.hand.total()
                if verbose:
                    print(f"Dealer cards: {dealer.hand.ranks()}")
                    print(f"{d} vs {p}")
                if dealer.hand.is_bust() or p > d:
                    outcome, pnl = 1, bet; msg = "Player wins"
                elif p < d:
                    outcome, pnl = -1, -bet; msg = "Player loses"
                else:
                    outcome, pnl = 0, 0.0; msg = "Push"
                if verbose: print(msg, "\n")
        net += pnl
        if outcome == 1: wins += 1
        elif outcome == -1: losses += 1
        else: pushes += 1
    if verbose:
        print("==== Final Stats ====")
        print(f"Games: {num_games}; Wins: {wins} ({wins/num_games:.1%}); Losses: {losses} ({losses/num_games:.1%}); Pushes: {pushes} ({pushes/num_games:.1%}); Net: {net:+.2f}\n")
    return {"wins": wins, "losses": losses, "pushes": pushes, "net_pnl": net}

# ----------------------------------------------------------------------------
# Plot helper (unchanged)
# ----------------------------------------------------------------------------

def plot_ev_table(ev_table: EVTable, *, title="Hard Totals", figsize: Tuple[int, int] = (14, 8), font_size: int = 10):
    rows = list(range(20, 3, -1)); cols = ["2","3","4","5","6","7","8","9","T","A"]
    data = []
    for t in rows:
        row = []
        for u in cols:
            evh = ev_table.ev((t,u), "hit")
            evs = ev_table.ev((t,u), "stand")
            ch = ev_table.data.get((t,u), {}).get("hit", EVStats()).count
            cs = ev_table.data.get((t,u), {}).get("stand", EVStats()).count
            best = "H" if evh>=evs else "S"
            row.append(f"{best}\nH:{evh:+.2f} ({ch})\nS:{evs:+.2f} ({cs})")
        data.append(row)
    fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
    tbl = ax.table(cellText=data, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center") 
    tbl.auto_set_font_size(False); tbl.set_fontsize(font_size)
    for _, cell in tbl.get_celld().items():
        cell.set_height(1 / (len(rows) + 1) * 1)
        cell.set_width(1 / (len(cols) + 1) * 1.2)
    plt.title(title); plt.tight_layout(); return fig

def plot_ev_table_soft(ev_table: EVTable, *, title="Soft Totals", figsize: Tuple[int, int] = (14, 8), font_size: int = 10):
    rows = list(range(20, 11, -1)); cols = ["2","3","4","5","6","7","8","9","T","A"]
    data = []
    for t in rows:
        row = []
        for u in cols:
            evh = ev_table.ev((t,u), "hit")
            evs = ev_table.ev((t,u), "stand")
            ch = ev_table.data.get((t,u), {}).get("hit", EVStats()).count
            cs = ev_table.data.get((t,u), {}).get("stand", EVStats()).count
            best = "H" if evh>=evs else "S"
            row.append(f"{best}\nH:{evh:+.2f} ({ch})\nS:{evs:+.2f} ({cs})")
        data.append(row)
    fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
    tbl = ax.table(cellText=data, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center") 
    tbl.auto_set_font_size(False); tbl.set_fontsize(font_size)
    for _, cell in tbl.get_celld().items():
        cell.set_height(1 / (len(rows) + 1) * 1)
        cell.set_width(1 / (len(cols) + 1) * 1.2)
    plt.title(title); plt.tight_layout(); return fig

# ----------------------------------------------------------------------------
# Demo when run standalone
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Grid learning 200k deterministic hands …")
    hard_table, soft_table = simulate_grid_learning(4000000, seed=42)
    print("Grid learning complete. Evaluating strategy built from grid\n")
    strat = TableStrategy(hard_table)
    analyze_strategy(strat, num_games=10000, verbose=True, seed=123)
    plot_ev_table(hard_table, title="Hard Totals")
    plot_ev_table_soft(soft_table, title="Soft Totals")
    plt.show()
