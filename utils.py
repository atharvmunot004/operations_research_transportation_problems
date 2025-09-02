from typing import List, Optional, Tuple, Dict

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
