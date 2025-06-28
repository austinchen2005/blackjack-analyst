import random
from collections import namedtuple
from typing import Callable, List, Protocol

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ------------------------------
# Card / Deck helpers
# ------------------------------
Card = namedtuple("Card", ["rank", "suit"])
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["s", "h", "d", "c"]  # suits kept only for completeness


def card_value(card: Card) -> int:
    """Return blackjack value for *card* (ace counted as 1 here – handled later)."""
    if card.rank in {"T", "J", "Q", "K"}:
        return 10
    if card.rank == "A":
        return 1
    return int(card.rank)


class Deck:
    """Represents a shoe of *num_decks* shuffled 52‑card decks.

    Reshuffles automatically when empty or when the shoe goes below 25 % penetration.
    """

    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self._reshuffle()

    # ------------------------------------------------------------------
    def _reshuffle(self):
        self.cards = [Card(rank, suit) for rank in RANKS for suit in SUITS] * self.num_decks
        random.shuffle(self.cards)

    # ------------------------------------------------------------------
    def deal(self) -> Card:
        if not self.cards:
            self._reshuffle()
        return self.cards.pop()

    # ------------------------------------------------------------------
    def maybe_reshuffle(self):
        """Reshuffle when there is less than 25 % of the shoe left."""
        if len(self.cards) < 52 * self.num_decks * 0.25:
            self._reshuffle()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.cards)


# ------------------------------
# Hand representation
# ------------------------------
class Hand:
    """A collection of cards with blackjack‑specific helpers."""

    def __init__(self):
        self.cards: List[Card] = []

    # ------------------------------------------------------------------
    def add_card(self, card: Card):
        self.cards.append(card)

    # ------------------------------------------------------------------
    def value(self) -> int:
        total = sum(card_value(c) for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == "A")
        while aces and total + 10 <= 21:
            total += 10
            aces -= 1
        return total

    # ------------------------------------------------------------------
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value() == 21

    # ------------------------------------------------------------------
    def is_bust(self) -> bool:
        return self.value() > 21

    # ------------------------------------------------------------------
    def ranks(self) -> str:
        """Return cards as a compact rank string (no suits)."""
        return "".join(c.rank for c in self.cards)

    # ------------------------------------------------------------------
    def __str__(self):
        ranks = " ".join(f"{c.rank}{c.suit}" for c in self.cards)
        return f"{ranks} ({self.value()})"


# ------------------------------
# Strategy interface & trivial bot
# ------------------------------
class Strategy(Protocol):
    def decide(self, hand: Hand, dealer_upcard: Card) -> str:  # noqa: D401
        """Return either 'hit' or 'stand'."""
        ...


class StandStrategy:
    """A strategy that always stands – useful as the default stub."""

    def decide(self, hand: Hand, dealer_upcard: Card) -> str:  # noqa: D401
        return "stand"


# ------------------------------
# Dealer logic (hits until hard 17 / stands on soft‑17)
# ------------------------------
class Dealer:
    def __init__(self, hit_soft_17: bool = False):
        self.hand = Hand()
        self.hit_soft_17 = hit_soft_17

    # ------------------------------------------------------------------
    def play(self, deck: Deck):
        while True:
            val = self.hand.value()
            soft = any(c.rank == "A" for c in self.hand.cards) and val <= 11
            if val < 17 or (self.hit_soft_17 and val == 17 and soft):
                self.hand.add_card(deck.deal())
            else:
                break


# ------------------------------
# Player wrapper to execute strategy with optional verbose trace
# ------------------------------
class Player:
    def __init__(self, strategy: Strategy):
        self.hand = Hand()
        self.strategy = strategy

    # ------------------------------------------------------------------
    def play(self, deck: Deck, dealer_upcard: Card, verbose: bool = True):
        while not self.hand.is_bust():
            decision = self.strategy.decide(self.hand, dealer_upcard)
            if decision == "hit":
                if verbose:
                    print("H")
                self.hand.add_card(deck.deal())
                if verbose:
                    print(f"P: {self.hand.ranks()}")
            else:
                if verbose:
                    print("S")
                break


# ------------------------------
# Single‑round game engine
# ------------------------------
class BlackjackGame:
    """Plays one round and returns (outcome, pnl)."""

    def __init__(self, deck: Deck, strategy: Strategy):
        self.deck = deck
        self.player = Player(strategy)
        self.dealer = Dealer()

    # ------------------------------------------------------------------
    def _deal_initial(self):
        self.player.hand.add_card(self.deck.deal())
        self.dealer.hand.add_card(self.deck.deal())
        self.player.hand.add_card(self.deck.deal())
        self.dealer.hand.add_card(self.deck.deal())

    # ------------------------------------------------------------------
    def play_round(self, bet: float = 1.0, verbose: bool = True) -> tuple[int, float]:
        """Return (outcome, pnl) where outcome ∈ {+1: win, 0: push, -1: loss}."""
        self._deal_initial()

        upcard = self.dealer.hand.cards[0]
        if verbose:
            print(f"D: {upcard.rank}")
            print(f"P: {self.player.hand.ranks()}")

        # Immediate blackjack checks ------------------------------------------------
        player_bj = self.player.hand.is_blackjack()
        dealer_bj = self.dealer.hand.is_blackjack()
        if player_bj or dealer_bj:
            if player_bj and dealer_bj:
                if verbose:
                    print("Both have blackjack – Push\n")
                return 0, 0.0  # push
            if dealer_bj:
                if verbose:
                    print("Dealer has blackjack – Player loses\n")
                return -1, -bet
            # player blackjack only
            if verbose:
                print("Player has blackjack – Player wins 3:2\n")
            return +1, 1.5 * bet

        # Player action ----------------------------------------------------------------
        self.player.play(self.deck, upcard, verbose)
        if self.player.hand.is_bust():
            if verbose:
                print("P bust\n")
            return -1, -bet

        # Dealer action ----------------------------------------------------------------
        self.dealer.play(self.deck)
        if verbose:
            print(f"D: {self.dealer.hand.ranks()}")
            print(f"{self.dealer.hand.value()} vs {self.player.hand.value()}")

        # Determine outcome -----------------------------------------------------------
        p_val, d_val = self.player.hand.value(), self.dealer.hand.value()
        if self.dealer.hand.is_bust() or p_val > d_val:
            if verbose:
                print("W \n") #win
            return +1, bet
        if p_val < d_val:
            if verbose:
                print("L\n") #lose
            return -1, -bet
        if verbose:
            print("P\n") #push
        return 0, 0.0


# ------------------------------
# Bulk simulation helper
# ------------------------------

def simulate_games(
    num_games: int = 1000,
    strategy: Strategy | None = None,
    bet_func: Callable[[int], float] | None = None,
    seed: int | None = None,
    verbose: bool = True,
):
    """Simulate *num_games* rounds. Return a summary dict including *net_pnl*."""
    if seed is not None:
        random.seed(seed)
    strategy = strategy or StandStrategy()
    bet_func = bet_func or (lambda _: 1.0)

    deck = Deck()

    wins = losses = pushes = 0
    net_pnl = 0.0

    for i in range(num_games):
        deck.maybe_reshuffle()
        bet = float(bet_func(i))
        outcome, pnl = BlackjackGame(deck, strategy).play_round(bet=bet, verbose=verbose)
        net_pnl += pnl
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            pushes += 1

        if verbose:
            print(f"pnl: {net_pnl}")

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / num_games,
        "loss_rate": losses / num_games,
        "push_rate": pushes / num_games,
        "net_pnl": net_pnl,
    }


# ------------------------------
# Quick CLI demo when executed directly
# ------------------------------
if __name__ == "__main__":

    pnls = np.array([])

    for i in range(1000):
        HANDS = 100  # keep small so we can view verbose output easily
        print("Running demo with StandStrategy…\n")
        summary = simulate_games(HANDS, strategy=StandStrategy(), verbose=False)
        print("Summary after", HANDS, "hands:")
        for k, v in summary.items():
            print(f"{k}: {v}")
        
        pnls = np.append(pnls, summary["net_pnl"])

    counts = Counter(pnls)
    x_vals, y_vals = zip(*sorted(counts.items()))
    
    plt.bar(x_vals, y_vals)
    plt.title("Example Array Plot")
    plt.xlabel("pnls")
    plt.ylabel("freq")
    plt.grid(True)
    plt.show()
