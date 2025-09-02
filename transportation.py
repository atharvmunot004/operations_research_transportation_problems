from typing import List, Optional, Tuple, Dict
import time

def timed(func):
    """Decorator to measure execution time of allocation methods."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        result["time_taken_sec"] = end - start
        return result
    return wrapper

def balance_problem(
    supply: List[float],
    demand: List[float],
    cost: Optional[List[List[float]]] = None,
    dummy_cost: float = 0.0
) -> Tuple[List[float], List[float], Optional[List[List[float]]], Dict]:
    """Balance supply and demand by adding a dummy row/column if needed."""
    S, D = sum(supply), sum(demand)
    meta = {"added_dummy": None, "diff": 0.0}

    bs, bd = supply[:], demand[:]
    bc = [row[:] for row in cost] if cost is not None else None

    if S > D:  # add dummy destination (column)
        diff = S - D
        bd.append(diff)
        if bc is not None:
            for row in bc:
                row.append(dummy_cost)
        meta.update({"added_dummy": "destination", "diff": diff})
    elif D > S:  # add dummy source (row)
        diff = D - S
        bs.append(diff)
        if bc is not None:
            cols = len(bc[0]) if bc else len(demand)
            bc.append([dummy_cost] * cols)
        meta.update({"added_dummy": "source", "diff": diff})

    return bs, bd, bc, meta

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


def print_transport_table(allocation: List[List[float]],
                          cost: Optional[List[List[float]]] = None) -> None:
    """Pretty print allocations (and per-cell cost if provided)."""
    m, n = len(allocation), len(allocation[0])
    def cell(r, c):
        a = allocation[r][c]
        if cost:
            return f"{int(a) if a.is_integer() else a} @ {cost[r][c]}"
        return f"{int(a) if isinstance(a, float) and a.is_integer() else a}"

    widths = [0]*n
    for c in range(n):
        widths[c] = max(len(cell(r,c)) for r in range(m))
    line = "+".join("-"*(w+2) for w in widths)

    for r in range(m):
        row = " | ".join(cell(r, c).rjust(widths[c]) for c in range(n))
        print(row)
        if r < m-1:
            print(line)

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

@timed
def vogels_approximation_method(
    supply: List[float],
    demand: List[float],
    cost: List[List[float]],
    dummy_cost: float = 0.0,
    tie_break: str = "max_allocation",  # "max_allocation" | "min_index"
    trace: bool = False
) -> Dict:
    """
    Vogel's Approximation Method (VAM) to build an initial basic feasible solution (IBFS).
    - Balances with a dummy row/col if needed (dummy_cost).
    - Each step: compute row/col penalties (difference between two smallest active costs),
      pick the line with the largest penalty, then allocate to its cheapest cell.
    - Tie-breaks:
        * If multiple lines share the same penalty, pick the line whose cheapest cell cost is smallest.
        * If still tied, use 'tie_break':
            - 'max_allocation' -> choose the cell allowing the largest immediate allocation.
            - 'min_index'      -> choose smallest (row, col) index.
    Returns dict with 'allocation', 'total_cost', 'balanced_supply', 'balanced_demand', 'balanced_cost', 'meta'
    and optional 'trace' list if trace=True.
    """
    if cost is None:
        raise ValueError("VAM requires a cost matrix.")
    if len(cost) != len(supply) or any(len(row) != len(demand) for row in cost):
        raise ValueError("Cost matrix shape must match original supply x demand before balancing.")

    # Reuse earlier helper to balance
    bs, bd, bc, meta = balance_problem(supply, demand, cost, dummy_cost)
    m, n = len(bs), len(bd)

    alloc = [[0.0 for _ in range(n)] for _ in range(m)]
    rs = bs[:]  # remaining supply
    rd = bd[:]  # remaining demand

    active_rows = set(range(m))
    active_cols = set(range(n))
    step_log = []

    def two_smallest(lst: List[Tuple[int, float]]) -> Tuple[float, float]:
        """Return (min1, min2) values from list of (index, value). If only one exists, min2 = min1."""
        if not lst:
            return (float("inf"), float("inf"))
        vals = sorted(v for _, v in lst)
        if len(vals) == 1:
            return (vals[0], vals[0])
        return (vals[0], vals[1])

    def row_penalty(i: int) -> Tuple[float, int, float]:
        """Return (penalty, argmin_col, cheapest_cost) for row i over active cols."""
        candidates = [(j, bc[i][j]) for j in active_cols]
        if not candidates:
            return (-1, -1, float("inf"))
        c1, c2 = two_smallest(candidates)
        penalty = c2 - c1  # classic VAM penalty
        # cheapest cell in this row:
        jmin, cmin = min(candidates, key=lambda x: x[1])
        return (penalty, jmin, cmin)

    def col_penalty(j: int) -> Tuple[float, int, float]:
        """Return (penalty, argmin_row, cheapest_cost) for column j over active rows."""
        candidates = [(i, bc[i][j]) for i in active_rows]
        if not candidates:
            return (-1, -1, float("inf"))
        c1, c2 = two_smallest(candidates)
        penalty = c2 - c1
        imin, cmin = min(candidates, key=lambda x: x[1])
        return (penalty, imin, cmin)

    while active_rows and active_cols:
        # 1) Compute penalties for all active rows/cols
        row_info = {i: row_penalty(i) for i in active_rows}
        col_info = {j: col_penalty(j) for j in active_cols}

        # 2) Find max penalty across rows and cols
        max_row_pen = max((row_info[i][0] for i in active_rows), default=-1)
        max_col_pen = max((col_info[j][0] for j in active_cols), default=-1)

        # Decide whether to pick a row or a column
        line_type = None
        chosen_line = None
        chosen_cell = None  # (i, j)
        cheapest_cost_on_line = None

        if max_row_pen > max_col_pen:
            # Pick best row among those with max penalty; break ties by smallest cheapest cell cost
            candidate_rows = [i for i in active_rows if row_info[i][0] == max_row_pen]
            chosen_line = min(candidate_rows, key=lambda i: row_info[i][2])
            line_type = "row"
            i = chosen_line
            j = row_info[i][1]
            cheapest_cost_on_line = row_info[i][2]
            chosen_cell = (i, j)
        elif max_col_pen > max_row_pen:
            candidate_cols = [j for j in active_cols if col_info[j][0] == max_col_pen]
            chosen_line = min(candidate_cols, key=lambda j: col_info[j][2])
            line_type = "col"
            j = chosen_line
            i = col_info[j][1]
            cheapest_cost_on_line = col_info[j][2]
            chosen_cell = (i, j)
        else:
            # Equal penalties: compare best row's cheapest vs best col's cheapest
            candidate_rows = [i for i in active_rows if row_info[i][0] == max_row_pen]
            best_row = min(candidate_rows, key=lambda i: row_info[i][2]) if candidate_rows else None
            candidate_cols = [j for j in active_cols if col_info[j][0] == max_col_pen]
            best_col = min(candidate_cols, key=lambda j: col_info[j][2]) if candidate_cols else None

            # If both exist, pick the line with smaller cheapest cell cost
            if best_row is not None and best_col is not None:
                if row_info[best_row][2] < col_info[best_col][2]:
                    line_type = "row"; chosen_line = best_row
                    i = chosen_line; j = row_info[i][1]
                    cheapest_cost_on_line = row_info[i][2]
                    chosen_cell = (i, j)
                elif col_info[best_col][2] < row_info[best_row][2]:
                    line_type = "col"; chosen_line = best_col
                    j = chosen_line; i = col_info[j][1]
                    cheapest_cost_on_line = col_info[j][2]
                    chosen_cell = (i, j)
                else:
                    # Still tied: decide by tie_break rule at cell level
                    # Build all min-cost cells across the tied lines
                    row_cells = [ (r, row_info[r][1]) for r in candidate_rows if row_info[r][2] == row_info[best_row][2] ]
                    col_cells = [ (col_info[c][1], c) for c in candidate_cols if col_info[c][2] == col_info[best_col][2] ]
                    all_cells = row_cells + col_cells
                    if tie_break == "max_allocation":
                        chosen_cell = max(all_cells, key=lambda rc: min(rs[rc[0]], rd[rc[1]]))
                    else:  # "min_index"
                        chosen_cell = min(all_cells, key=lambda rc: (rc[0], rc[1]))
                    i, j = chosen_cell
                    cheapest_cost_on_line = bc[i][j]
                    line_type = "row" if i in active_rows else "col"
            elif best_row is not None:
                line_type = "row"; chosen_line = best_row
                i = chosen_line; j = row_info[i][1]
                cheapest_cost_on_line = row_info[i][2]
                chosen_cell = (i, j)
            elif best_col is not None:
                line_type = "col"; chosen_line = best_col
                j = chosen_line; i = col_info[j][1]
                cheapest_cost_on_line = col_info[j][2]
                chosen_cell = (i, j)
            else:
                break  # nothing left

        # 3) Allocate to the chosen cheapest cell on the chosen line
        i, j = chosen_cell
        x = min(rs[i], rd[j])
        alloc[i][j] += x  # (+= to be safe if logic revisits; typically it's 0 before)
        rs[i] -= x
        rd[j] -= x

        if trace:
            step_log.append({
                "picked_line_type": line_type,
                "picked_line": chosen_line,
                "penalty": (row_info[i][0] if line_type == "row" else col_info[j][0]),
                "cell": (i, j),
                "cell_cost": bc[i][j],
                "allocated": x,
                "remaining_row_supply": rs[i],
                "remaining_col_demand": rd[j]
            })

        # 4) Deactivate exhausted row/col
        if rs[i] == 0:
            active_rows.discard(i)
        if rd[j] == 0:
            active_cols.discard(j)

        # optional early stop
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


import random

def generate_transportation_problem(
    num_sources: int,
    num_destinations: int,
    supply_range=(10, 50),
    demand_range=(10, 50),
    cost_range=(1, 20),
    balance: bool = True
):
    """
    Generate random supply list, demand list, and cost matrix for a transportation problem.

    Args:
        num_sources (int): number of supply nodes
        num_destinations (int): number of demand nodes
        supply_range (tuple): (min, max) range for supply values
        demand_range (tuple): (min, max) range for demand values
        cost_range (tuple): (min, max) range for costs
        balance (bool): whether to balance total supply and demand

    Returns:
        supply (list[int])
        demand (list[int])
        cost (list[list[int]])
    """
    # Random supply and demand
    supply = [random.randint(*supply_range) for _ in range(num_sources)]
    demand = [random.randint(*demand_range) for _ in range(num_destinations)]

    total_supply, total_demand = sum(supply), sum(demand)

    if balance:
        # Adjust last element of demand to balance totals
        diff = total_supply - total_demand
        demand[-1] += diff
        if demand[-1] < 0:  # avoid negatives, just force balance differently
            demand[-1] = abs(demand[-1])
            supply[-1] += demand[-1]
        total_supply, total_demand = sum(supply), sum(demand)

    # Random cost matrix
    cost = [
        [random.randint(*cost_range) for _ in range(num_destinations)]
        for _ in range(num_sources)
    ]

    return supply, demand, cost


# Example usage
if __name__ == "__main__":
    supply, demand, cost = generate_transportation_problem(3, 4, balance=True)

    print("Supply:", supply)
    print("Demand:", demand)
    print("Cost Matrix:")
    for row in cost:
        print(row)




# --------- Example usage ----------
if __name__ == "__main__":
    # Example (balanced): 3 sources, 4 destinations
    supply = [20, 30, 25]
    demand = [10, 25, 20, 20]
    cost = [
        [8, 6, 10, 9],
        [9, 7, 4, 2],
        [3, 4, 2, 5],
    ]

    res = nw_corner_method(supply, demand, cost)
    print("NWCM Allocation (amount @ cost):")
    print_transport_table(res["allocation"], res["balanced_cost"])
    print("\nTotal Cost:", res["total_cost"])
    print("Meta:", res["meta"])
    print ("-------------------------------------------------")
    supply = [20, 30, 25]
    demand = [10, 25, 20, 20]
    cost = [
        [8, 6, 10, 9],
        [9, 7,  4, 2],
        [3, 4,  2, 5],
    ]

    lcm_res = least_cost_method(supply, demand, cost, tie_break="max_allocation", trace=True)
    print("LCM Allocation (amount @ cost):")
    print_transport_table(lcm_res["allocation"], lcm_res["balanced_cost"])
    print("\nTotal Cost:", lcm_res["total_cost"])
    print("Meta:", lcm_res["meta"])
    # Optional: print trace
    # from pprint import pprint; pprint(lcm_res["trace"])
    print ("-------------------------------------------------")
    supply = [20, 30, 25]
    demand = [10, 25, 20, 20]
    cost = [
        [8, 6, 10, 9],
        [9, 7,  4, 2],
        [3, 4,  2, 5],
    ]
    vam_res = vogels_approximation_method(supply, demand, cost, tie_break="max_allocation", trace=True)
    print("VAM Allocation (amount @ cost):")
    print_transport_table(vam_res["allocation"], vam_res["balanced_cost"])
    print("\nTotal Cost:", vam_res["total_cost"])
    print("Meta:", vam_res["meta"])
    # from pprint import pprint; pprint(vam_res["trace"])
    print ("-------------------------------------------------")
    res_nwcm = nw_corner_method(supply, demand, cost)
    print("NWCM cost:", res_nwcm["total_cost"], "time:", res_nwcm["time_taken_sec"], "sec")

    res_lcm = least_cost_method(supply, demand, cost)
    print("LCM cost:", res_lcm["total_cost"], "time:", res_lcm["time_taken_sec"], "sec")

    res_vam = vogels_approximation_method(supply, demand, cost)
    print("VAM cost:", res_vam["total_cost"], "time:", res_vam["time_taken_sec"], "sec")