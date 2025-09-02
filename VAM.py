from typing import List, Optional, Tuple, Dict
from timed import timed
from utils import balance_problem

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
