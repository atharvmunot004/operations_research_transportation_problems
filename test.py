from generate_problem import generate_transportation_problem
from NWCM import nw_corner_method
from LCM import least_cost_method
from VAM import vogels_approximation_method
from MODI import modi_optimize_from_ibfs
from typing import List, Optional, Tuple, Dict
from utils import print_transport_table

def run_large_test():
    # Generate a large transportation problem
    supply, demand, cost = generate_transportation_problem(balance=True)

    print("Generated Transportation Problem:")
    print("Supply:", supply)
    print("Demand:", demand)
    print("Cost Matrix:")
    for row in cost:
        print(row)

    # Solve using North-West Corner Method
    nwcm_result = nw_corner_method(supply, demand, cost)
    print("\nNorth-West Corner Method Allocation:")
    print_transport_table(nwcm_result['allocation'], nwcm_result['balanced_cost'])
    print("Total Cost:", nwcm_result['total_cost'])

    # Solve using Least Cost Method
    lcm_result = least_cost_method(supply, demand, cost)
    print("\nLeast Cost Method Allocation:")
    print_transport_table(lcm_result['allocation'], lcm_result['balanced_cost'])
    print("Total Cost:", lcm_result['total_cost'])

    # Solve using Vogel's Approximation Method
    vam_result = vogels_approximation_method(supply, demand, cost)
    print("\nVogel's Approximation Method Allocation:")
    print_transport_table(vam_result['allocation'], vam_result['balanced_cost'])
    print("Total Cost:", vam_result['total_cost'])


    # Optimize using MODI Method starting from VAM result
    modi_result = modi_result = modi_optimize_from_ibfs({
    "allocation":       vam_result["allocation"],
    "balanced_cost":    vam_result["balanced_cost"],
    "balanced_supply":  vam_result["balanced_supply"],
    "balanced_demand":  vam_result["balanced_demand"],
}, trace=True)
    print("\nMODI Method Optimized Allocation (from VAM):")
    print_transport_table(modi_result['allocation'], cost)
    print("Total Cost:", modi_result['total_cost'])
    print("Time taken for MODI optimization:", modi_result['time_taken_sec'], "seconds")

if __name__ == "__main__":
    run_large_test()
