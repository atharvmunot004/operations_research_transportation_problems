from typing import List, Optional, Tuple, Dict
from timed import timed
from utils import balance_problem

@timed
def least_cost_method(
    supply: List[float],
    demand: List[float],
    cost: List[List[float]],
    dummy_cost: float = 0.0,
    tie_break: str = "max_allocation",  # "max_allocation" | "min_index"
    trace: bool = False
) -> Dict:
    """
    Compute a Least Cost Method (LCM) allocation.
    - Balances if needed (adds dummy row/col with dummy_cost).
    - Greedily allocates to the lowest-cost available cell each step.
    - tie_break:
        - "max_allocation": if multiple cells have the same minimum cost,
                           pick the one allowing the largest immediate allocation.
        - "min_index": break ties by (row, col) smallest index (deterministic).
    Returns dict with 'allocation', 'total_cost', 'balanced_*', and 'meta'.
    """
    if cost is None:
        raise ValueError("LCM requires a cost matrix.")
    if len(cost) != len(supply) or any(len(row) != len(demand) for row in cost):
        raise ValueError("Cost matrix shape must match original supply x demand before balancing.")

    # Reuse the balancing logic from earlier
    bs, bd, bc, meta = balance_problem(supply, demand, cost, dummy_cost)
    m, n = len(bs), len(bd)

    alloc = [[0.0 for _ in range(n)] for _ in range(m)]
    rs = bs[:]  # remaining supply
    rd = bd[:]  # remaining demand

    active_rows = set(range(m))
    active_cols = set(range(n))

    step_log = []

    def argmin_cells():
        """Return list of (i,j,c) cells with global minimum cost over active rows/cols."""
        min_c = None
        bucket = []
        for i in active_rows:
            for j in active_cols:
                c = bc[i][j]
                if (min_c is None) or (c < min_c):
                    min_c = c
                    bucket = [(i, j, c)]
                elif c == min_c:
                    bucket.append((i, j, c))
        return bucket  # all share minimal cost

    while active_rows and active_cols:
        candidates = argmin_cells()
        if not candidates:
            break  # should not happen for a consistent transportation table

        # Tie-break among same-cost cells
        if tie_break == "max_allocation":
            # pick cell that allows max immediate allocation
            best = None
            best_amt = -1
            for (i, j, c) in candidates:
                amt = min(rs[i], rd[j])
                if amt > best_amt:
                    best_amt = amt
                    best = (i, j, c)
        elif tie_break == "min_index":
            best = min(candidates, key=lambda x: (x[0], x[1]))
        else:
            raise ValueError("Unknown tie_break; use 'max_allocation' or 'min_index'.")

        i, j, c = best
        x = min(rs[i], rd[j])
        alloc[i][j] = x
        rs[i] -= x
        rd[j] -= x

        if trace:
            step_log.append({
                "cell": (i, j),
                "cost": c,
                "allocated": x,
                "remaining_row": rs[i],
                "remaining_col": rd[j]
            })

        # Remove exhausted row/col from active sets
        row_zero = (rs[i] == 0)
        col_zero = (rd[j] == 0)
        if row_zero:
            active_rows.discard(i)
        if col_zero:
            active_cols.discard(j)
        # If both are zero, both get removed (degenerate basic feasible solution may occur; OK for IBFS)

        # Stop when all demand satisfied (or supply exhausted)
        if sum(rd) == 0 or sum(rs) == 0:
            break

    total_cost = sum(alloc[r][c] * bc[r][c] for r in range(m) for c in range(n))

    out = {
        "allocation": alloc,
        "total_cost": total_cost,
        "balanced_supply": bs,
        "balanced_demand": bd,
        "balanced_cost": bc,
        "meta": meta
    }
    if trace:
        out["trace"] = step_log
    return out
