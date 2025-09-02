import random

def generate_transportation_problem(
    supply_range=(10, 50),
    demand_range=(10, 50),
    cost_range=(1, 20),
    balance: bool = True
):
    """
    Generate random supply list, demand list, and cost matrix for a transportation problem.

    Args:
        supply_range (tuple): (min, max) range for supply values
        demand_range (tuple): (min, max) range for demand values
        cost_range (tuple): (min, max) range for costs
        balance (bool): whether to balance total supply and demand

    Returns:
        supply (list[int])
        demand (list[int])
        cost (list[list[int]])
    """
    # Ensure at least 1 source and 1 destination and that the numbers are reasonable
    num_sources = random.randint(1, 10)
    num_destinations = random.randint(1, 10)

    # Random supply and demand
    supply = [random.randint(*supply_range) for _ in range(num_sources)]
    demand = [random.randint(*demand_range) for _ in range(num_destinations)]

    total_supply, total_demand = sum(supply), sum(demand)

    if balance:
        if (total_supply == total_demand):
            pass

        if (total_supply > total_demand):
            # Add to demand
            diff = total_supply - total_demand
            avg_add = diff // num_destinations
            reaminder = diff % num_destinations
            for i in range(num_destinations):
                demand[i] += avg_add
                if reaminder > 0:
                    demand[i] += 1
                    reaminder -= 1

        if (total_demand > total_supply):
            # Add to supply
            diff = total_demand - total_supply
            avg_add = diff // num_sources
            reaminder = diff % num_sources
            for i in range(num_sources):
                supply[i] += avg_add
                if reaminder > 0:
                    supply[i] += 1
                    reaminder -= 1

        total_supply, total_demand = sum(supply), sum(demand)

    # Random cost matrix
    cost = [
        [random.randint(*cost_range) for _ in range(num_destinations)]
        for _ in range(num_sources)
    ]

    return supply, demand, cost


# # Example usage
# if __name__ == "__main__":
#     supply, demand, cost = generate_transportation_problem(balance=True)

#     print("Supply:", supply)
#     print("Demand:", demand)
#     print("Cost Matrix:")
#     for row in cost:
#         print(row)
