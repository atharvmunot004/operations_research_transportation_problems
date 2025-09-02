import time
from typing import List, Tuple, Dict, Optional, Set
from timed import timed

EPS = 1e-9

def _is_basic(x: float) -> bool:
    return x > EPS

def _count_basics(alloc: List[List[float]]) -> int:
    return sum(1 for r in alloc for v in r if _is_basic(v))

def _ensure_basis_size(alloc: List[List[float]], supply: List[float], demand: List[float]) -> None:
    """
    Ensure exactly m + n - 1 basic variables by adding tiny eps in zero cells if needed (degeneracy fix).
    Modifies 'alloc' in place. We only add eps where it won’t violate row/col sums (it never does,
    ε is virtual for basis bookkeeping).
    """
    m, n = len(alloc), len(alloc[0])
    needed = m + n - 1
    have = _count_basics(alloc)
    if have >= needed:
        return

    # Simple heuristic: scan row by row, add eps to cells that keep independence (no cycle duplication).
    # For practical use, adding eps to any distinct row/col pattern is acceptable to break degeneracy.
    for i in range(m):
        for j in range(n):
            if have >= needed:
                return
            if alloc[i][j] <= EPS:
                alloc[i][j] = EPS
                have += 1

def _compute_uv(alloc: List[List[float]], cost: List[List[float]]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Solve u_i + v_j = c_ij for all basic cells (alloc>0) with u_0=0.
    Returns (u, v) lists, some entries may remain None if table disconnected (rare with ε-fix).
    """
    m, n = len(alloc), len(alloc[0])
    u = [None] * m
    v = [None] * n
    u[0] = 0.0  # anchor

    # BFS-style propagation over basic cells
    changed = True
    while changed:
        changed = False
        for i in range(m):
            for j in range(n):
                if _is_basic(alloc[i][j]):  # equation exists
                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i][j] - u[i]
                        changed = True
                    elif u[i] is None and v[j] is not None:
                        u[i] = cost[i][j] - v[j]
                        changed = True
    return u, v

def _reduced_costs(alloc: List[List[float]], cost: List[List[float]], u: List[Optional[float]], v: List[Optional[float]]) -> List[List[float]]:
    """
    Compute opportunity (reduced) costs for NON-basic cells: delta_ij = c_ij - (u_i + v_j)
    For basic cells we set delta to 0.
    """
    m, n = len(alloc), len(alloc[0])
    delta = [[0.0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if not _is_basic(alloc[i][j]):
                if u[i] is None or v[j] is None:
                    # Disconnected component (degenerate). Treat as very large opportunity cost so it won't enter now.
                    delta[i][j] = 0.0
                else:
                    delta[i][j] = cost[i][j] - (u[i] + v[j])
            else:
                delta[i][j] = 0.0
    return delta

def _find_entering_cell(delta: List[List[float]]) -> Optional[Tuple[int, int, float]]:
    """
    Choose the most negative delta (best improvement). Return (i,j,delta_ij) or None if all >= 0.
    """
    entering = None
    min_val = 0.0
    m, n = len(delta), len(delta[0])
    for i in range(m):
        for j in range(n):
            if delta[i][j] < min_val - 1e-15:
                min_val = delta[i][j]
                entering = (i, j, min_val)
    return entering

def _basic_positions(alloc: List[List[float]]) -> Set[Tuple[int, int]]:
    return {(i, j) for i in range(len(alloc)) for j in range(len(alloc[0])) if _is_basic(alloc[i][j])}

def _build_cycle(alloc: List[List[float]], start: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Build a closed +/− cycle starting and ending at 'start' by alternating moves along
    rows and columns through existing BASIC positions. Returns an ordered list of positions
    (even length >= 4) where the first is the entering cell (start). Pattern: + - + - ...
    """
    m, n = len(alloc), len(alloc[0])
    basics = _basic_positions(alloc)  # set of (i,j) with alloc>0
    i0, j0 = start

    # We temporarily treat 'start' as basic to find a cycle
    basics_with_start = set(basics)
    basics_with_start.add(start)

    # Precompute row->cols and col->rows adjacency among basics
    row_cols = {i: [] for i in range(m)}
    col_rows = {j: [] for j in range(n)}
    for (i, j) in basics_with_start:
        row_cols[i].append(j)
        col_rows[j].append(i)

    # DFS alternating between columns and rows
    # State: (i,j, phase, path) phase 0 means move along row (change j), phase 1 means move along col (change i)
    # Start by moving along row (phase 0)
    stack = [ (i0, j0, 0, [(i0, j0)]) ]
    visited = set()

    while stack:
        i, j, phase, path = stack.pop()
        key = (i, j, phase)
        if key in visited:
            continue
        visited.add(key)

        if phase == 0:
            # move along row i: choose any basic in same row with different col
            for jj in row_cols[i]:
                if jj == j: 
                    continue
                nxt = (i, jj)
                # if we reached start and path length >= 4 and alternates (+-+-), we found a cycle
                if nxt == start and len(path) >= 4 and len(path) % 2 == 0:
                    return path
                if nxt in basics_with_start:
                    stack.append((i, jj, 1, path + [nxt]))
        else:
            # move along column j: choose any basic in same column with different row
            for ii in col_rows[j]:
                if ii == i: 
                    continue
                nxt = (ii, j)
                if nxt == start and len(path) >= 4 and len(path) % 2 == 0:
                    return path
                if nxt in basics_with_start:
                    stack.append((ii, j, 0, path + [nxt]))
    return None

def _theta_and_positions(alloc: List[List[float]], cycle: List[Tuple[int, int]]) -> Tuple[float, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Given the alternating cycle (start is +), compute theta = min of allocations on 'minus' positions.
    Return (theta, plus_positions, minus_positions).
    """
    plus = cycle[0::2]
    minus = cycle[1::2]
    theta = min(alloc[i][j] for (i, j) in minus)
    return theta, plus, minus

def _apply_pivot(alloc: List[List[float]], plus: List[Tuple[int, int]], minus: List[Tuple[int, int]], theta: float) -> None:
    """Apply +theta to plus positions and -theta to minus positions (in place)."""
    for (i, j) in plus:
        alloc[i][j] += theta
    for (i, j) in minus:
        alloc[i][j] -= theta
        if alloc[i][j] < EPS:
            alloc[i][j] = 0.0  # clean tiny negatives to zero

@timed
def modi_optimize_from_ibfs(
    ibfs: Dict,
    max_iter: int = 10_000,
    add_eps_to_fix_degeneracy: bool = True,
    trace: bool = False,
    timed: bool = True
) -> Dict:
    """
    Optimize an initial feasible transportation solution using MODI.
    Args:
        ibfs: dict returned by your NWCM/LCM/VAM with keys:
              'allocation', 'balanced_cost', 'balanced_supply', 'balanced_demand'
        max_iter: safety cap on iterations
        add_eps_to_fix_degeneracy: ensure exactly m+n-1 basics by ε if needed
        trace: include per-iteration details
        timed: include 'time_taken_sec' in result
    Returns:
        dict with:
         - 'allocation' (optimized),
         - 'total_cost',
         - 'iterations',
         - 'improved' (bool),
         - 'trace' (optional)
    """
    start_t = time.perf_counter() if timed else None

    alloc = [row[:] for row in ibfs["allocation"]]
    cost = ibfs["balanced_cost"]
    supply = ibfs["balanced_supply"]
    demand = ibfs["balanced_demand"]

    m, n = len(alloc), len(alloc[0])
    assert len(cost) == m and len(cost[0]) == n, "Cost shape mismatch with allocation."

    # Degeneracy handling
    if add_eps_to_fix_degeneracy:
        _ensure_basis_size(alloc, supply, demand)

    iters = 0
    improved = False
    steps = []

    while iters < max_iter:
        iters += 1

        # 1) Compute potentials u, v
        u, v = _compute_uv(alloc, cost)

        # 2) Compute reduced costs for non-basics
        delta = _reduced_costs(alloc, cost, u, v)

        # 3) Choose entering cell (most negative delta)
        enter = _find_entering_cell(delta)
        if enter is None:
            # Could be all zeros; if any negative exists, we’d have found it.
            break
        ei, ej, dval = enter

        if dval >= -1e-15:
            # All deltas >= 0 => optimal
            break

        improved = True

        # 4) Build cycle including entering cell
        cycle = _build_cycle(alloc, (ei, ej))
        if cycle is None:
            # Very rare; try a gentle ε fix and continue
            if add_eps_to_fix_degeneracy:
                _ensure_basis_size(alloc, supply, demand)
                continue
            else:
                # Cannot proceed
                break

        # 5) Theta and pivot
        theta, plus, minus = _theta_and_positions(alloc, cycle)
        _apply_pivot(alloc, plus, minus, theta)

        if trace:
            steps.append({
                "iteration": iters,
                "entering_cell": (ei, ej),
                "entering_delta": dval,
                "theta": theta,
                "plus_positions": plus,
                "minus_positions": minus,
            })

    # Final cost
    total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))

    result = {
        "allocation": alloc,
        "total_cost": total_cost,
        "iterations": iters,
        "improved": improved
    }
    if trace:
        result["trace"] = steps
    if timed:
        result["time_taken_sec"] = time.perf_counter() - start_t
    return result


# # ----------------- Tiny demo -----------------
# if __name__ == "__main__":
#     # Example IBFS from VAM / NWCM / LCM:
#     supply = [20, 30, 25]
#     demand = [10, 25, 20, 20]
#     cost = [
#         [8, 6, 10, 9],
#         [9, 7,  4, 2],
#         [3, 4,  2, 5],
#     ]

#     # Suppose you already computed IBFS using one of your functions:
#     # res = nw_corner_method(supply, demand, cost)
#     # or:
#     # res = least_cost_method(supply, demand, cost)
#     # or:
#     # res = vogels_approximation_method(supply, demand, cost)

#     # For a quick check, here’s a trivial NW corner (inline) to make an IBFS:
#     def quick_nwcm(s, d):
#         ss, dd = s[:], d[:]
#         m, n = len(ss), len(dd)
#         A = [[0]*n for _ in range(m)]
#         i = j = 0
#         while i < m and j < n:
#             x = min(ss[i], dd[j])
#             A[i][j] = x
#             ss[i] -= x; dd[j] -= x
#             if ss[i] == 0 and dd[j] == 0:
#                 i += 1; j += 1
#             elif ss[i] == 0:
#                 i += 1
#             else:
#                 j += 1
#         return A

#     allocation0 = quick_nwcm(supply, demand)
#     ibfs = {
#         "allocation": allocation0,
#         "balanced_cost": cost,
#         "balanced_supply": supply,
#         "balanced_demand": demand,
#     }

#     opt = modi_optimize_from_ibfs(ibfs, trace=True)
#     print("Optimized total cost:", opt["total_cost"], "iterations:", opt["iterations"], "improved:", opt["improved"])
#     # Optional: inspect steps
#     # from pprint import pprint; pprint(opt["trace"])
