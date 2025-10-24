from collections import Counter

_RANK_MAP = {r: i for i, r in enumerate("..23456789TJQKA", start=0)}
# values: '2'->2, ..., 'T'->10, 'J'->11, 'Q'->12, 'K'->13, 'A'->14

def _parse(hand):
    # "4S 5S 7H 8D JC" -> (ranks(list[int]), suits(list[str]))
    cards = hand.split()
    ranks = [_RANK_MAP[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    return ranks, suits

def _is_straight(ranks):
    """Return (is_straight, high_rank) with Ace-low handled."""
    uniq = sorted(set(ranks), reverse=True)
    if len(uniq) != 5:
        return False, 0
    # normal straight
    if uniq[0] - uniq[-1] == 4:
        return True, uniq[0]
    # A-2-3-4-5 wheel: treat Ace as 1 -> high = 5
    if uniq == [14, 5, 4, 3, 2]:
        return True, 5
    return False, 0

def _is_flush(suits):
    return len(set(suits)) == 1

def _hand_rank(hand):
    """
    Produce a rank tuple comparable with max():
    Category order (ascending):
      0 High Card, 1 One Pair, 2 Two Pair, 3 Three of a Kind,
      4 Straight, 5 Flush, 6 Full House, 7 Four of a Kind, 8 Straight Flush
    """
    ranks, suits = _parse(hand)
    ranks_sorted = sorted(ranks, reverse=True)
    flush = _is_flush(suits)
    straight, straight_high = _is_straight(ranks)

    if straight and flush:
        return (8, straight_high)

    cnt = Counter(ranks)
    # items sorted by (count desc, rank desc)
    by_count_then_rank = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [c for _, c in by_count_then_rank]
    ordered_ranks = [r for r, _ in by_count_then_rank]  # ranks grouped by multiplicity

    if counts == [4, 1]:
        # Four of a kind: quad rank, kicker
        quad, kicker = ordered_ranks[0], ordered_ranks[1]
        return (7, quad, kicker)

    if counts == [3, 2]:
        # Full house: trips rank, pair rank
        trips, pair = ordered_ranks[0], ordered_ranks[1]
        return (6, trips, pair)

    if flush:
        # Flush: compare all ranks high to low
        return (5, *ranks_sorted)

    if straight:
        return (4, straight_high)

    if counts == [3, 1, 1]:
        trips = ordered_ranks[0]
        kickers = sorted(ordered_ranks[1:], reverse=True)
        return (3, trips, *kickers)

    if counts == [2, 2, 1]:
        high_pair, low_pair = sorted(ordered_ranks[:2], reverse=True)
        kicker = ordered_ranks[2]
        return (2, high_pair, low_pair, kicker)

    if counts == [2, 1, 1, 1]:
        pair = ordered_ranks[0]
        kickers = sorted(ordered_ranks[1:], reverse=True)
        return (1, pair, *kickers)

    # High card
    return (0, *ranks_sorted)

def best_hands(hands):
    """
    Given a list of hand strings, return a list with the best hand(s).
    If multiple tie for best, return all of them (original strings).
    """
    scored = [(h, _hand_rank(h)) for h in hands]
    best_score = max(score for _, score in scored)
    return [h for h, score in scored if score == best_score]
