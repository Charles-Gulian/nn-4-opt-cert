"""Greedy minimum-fill elimination ordering and bag extraction for chordal SDP.

Implements the algorithm from Permutation.m / Fill_In.m / Deg.m in the
OPF Solver (Madani, Ashraphijuo, Lavaei 2014), translated to Python.

The elimination ordering produces a sequence of bags B_0, ..., B_{n-1}
where each bag is the set {v} ∪ N(v) at the moment v is eliminated from the
fill-in graph.  The maximum |B_i| - 1 is an upper bound on the treewidth.
"""

import numpy as np


def greedy_elimination(branch_from, branch_to, n_buses, alpha=0.0):
    """Greedy minimum-fill elimination ordering.

    At each step, picks the node v minimising  alpha * deg(v) + fill_in(v),
    where fill_in(v) is the number of edges that would be added to make
    N(v) a clique.  alpha=0 is pure min-fill (default, matches the MATLAB
    code default of settings.alpha=0).

    Parameters
    ----------
    branch_from, branch_to : array-like of 0-based bus indices
    n_buses : int
    alpha : float

    Returns
    -------
    bags : list of sorted lists of 0-based bus indices (one per eliminated bus)
    treewidth : int   upper bound on treewidth
    """
    # Build adjacency sets from branch list
    neighbours = [set() for _ in range(n_buses)]
    for f, t in zip(branch_from, branch_to):
        f, t = int(f), int(t)
        neighbours[f].add(t)
        neighbours[t].add(f)

    def _fill_in(v):
        """Edges missing from N(v) to make it a clique."""
        nbrs = neighbours[v]
        d = len(nbrs)
        shared = sum(len(nbrs & neighbours[u]) for u in nbrs)
        return (d * (d - 1) - shared) // 2

    def _degree(v):
        return len(neighbours[v])

    # Initialise fill-in and degree arrays
    fil = np.array([_fill_in(v) for v in range(n_buses)], dtype=float)
    deg = np.array([_degree(v)  for v in range(n_buses)], dtype=float)

    remaining = set(range(n_buses))
    bags = []
    treewidth = 1

    for _ in range(n_buses):
        # Pick node with minimum cost (break ties by index for determinism).
        # When alpha=0 and deg[v]=inf (already eliminated), 0*inf=nan — use
        # fil directly in that case to avoid the RuntimeWarning.
        if alpha == 0.0:
            score = fil
        else:
            score = alpha * deg + fil
        score_remaining = {v: score[v] for v in remaining}
        m = min(score_remaining, key=lambda v: (score_remaining[v], v))

        bag = sorted([m] + list(neighbours[m]))
        bags.append(bag)
        treewidth = max(treewidth, len(bag) - 1)

        nbrs_m = list(neighbours[m])

        # Make N(m) a clique by adding fill-in edges
        for u in nbrs_m:
            for w in nbrs_m:
                if w != u:
                    neighbours[u].add(w)

        # Remove m from all neighbour sets and from the active set
        for u in nbrs_m:
            neighbours[u].discard(m)
        neighbours[m] = set()
        remaining.discard(m)

        # Recompute fil/deg for affected nodes (m's old neighbours and their neighbours)
        affected = set(nbrs_m)
        for u in nbrs_m:
            affected |= neighbours[u]
        for v in affected & remaining:
            fil[v] = _fill_in(v)
            deg[v] = _degree(v)

        # Mark eliminated node so it is never picked again
        fil[m] = np.inf
        deg[m] = np.inf

    return bags, treewidth


def unique_pairs_in_bags(bags):
    """Return the set of unique ordered pairs (k, m) with k < m that appear
    together in at least one bag.  These are exactly the entries of the
    upper triangle of W that the chordal SDP needs as explicit variables.
    """
    pairs = set()
    for bag in bags:
        bag_sorted = sorted(bag)
        for i, a in enumerate(bag_sorted):
            for b in bag_sorted[i + 1:]:
                pairs.add((a, b))
    return pairs
