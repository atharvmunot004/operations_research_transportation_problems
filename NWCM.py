from typing import List, Optional, Tuple, Dict
from timed import timed
from utils import balance_problem

@timed
def nw_corner_method(
    supply: List[float],
    demand: List[float],
    cost: Optional[List[List[float]]] = None,
    dummy_cost: float = 0.0
) -> Dict:
    """
    Compute a North-West Corner allocation.
    Returns dict with 'allocation', 'total_cost', 'balanced_supply', 'balanced_demand', and 'meta'.
    """
    # Balance if needed
    bs, bd, bc, meta = balance_problem(supply, demand, cost, dummy_cost)
    m, n = len(bs), len(bd)

    # Basic checks if cost provided
    if cost is not None:
        if len(cost) != len(supply) or any(len(row) != len(demand) for row in cost):
            raise ValueError("Cost matrix shape must match original supply x demand before balancing.")
        # bc is already expanded in balance_problem

    alloc = [[0.0 for _ in range(n)] for _ in range(m)]
    i = j = 0
    rs = bs[:]  # remaining supply
    rd = bd[:]  # remaining demand

    # NW corner loop
    while i < m and j < n:
        x = min(rs[i], rd[j])
        alloc[i][j] = x
        rs[i] -= x
        rd[j] -= x

        # Move down/right; if both exhausted, move diagonally (classic NWCM tie rule)
        if rs[i] == 0 and rd[j] == 0:
            i += 1
            j += 1
        elif rs[i] == 0:
            i += 1
        else:  # rd[j] == 0
            j += 1

    # Cost
    total_cost = None
    if bc is not None:
        total_cost = sum(alloc[r][c] * bc[r][c] for r in range(m) for c in range(n))

    return {
        "allocation": alloc,
        "total_cost": total_cost,
        "balanced_supply": bs,
        "balanced_demand": bd,
        "balanced_cost": bc,
        "meta": meta
    }
