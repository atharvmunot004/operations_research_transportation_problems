import uuid
from generate_problem import generate_transportation_problem
from NWCM import nw_corner_method
from LCM import least_cost_method
from VAM import vogels_approximation_method
from MODI import modi_optimize_from_ibfs
from typing import List, Optional, Tuple, Dict
from utils import print_transport_table, balance_problem
import csv
from plot_graphs import plot_transport_times_with_diff_dual_axis

def create_csv():
    with open("dataset/transportation_results.csv", "w", newline='') as csvfile:
        fieldnames = ["problem_id", "method", "rows", "comlumns", "total_cost", "time_taken_sec", "optimized_cost", "time_to_optimize_sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def solve_nwcm(supply, demand, cost, problem_id):
    # Solve using North-West Corner Method
    nwcm_result = nw_corner_method(supply, demand, cost)
    optimized_result = modi_optimize_from_ibfs({
        "allocation":       nwcm_result["allocation"],
        "balanced_cost":    nwcm_result["balanced_cost"],
        "balanced_supply":  nwcm_result["balanced_supply"],
        "balanced_demand":  nwcm_result["balanced_demand"],
    }, trace=False)
    with open(f"dataset/transportation_results.csv", "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["problem_id", "method", "rows", "comlumns", "total_cost", "time_taken_sec", "optimized_cost", "time_to_optimize_sec"])
        writer.writerow({
            "problem_id": problem_id,
            "method": "NWCM",
            "rows": len(nwcm_result['balanced_supply']),
            "comlumns": len(nwcm_result['balanced_demand']),
            "total_cost": nwcm_result['total_cost'],
            "time_taken_sec": nwcm_result['time_taken_sec'],
            "optimized_cost": optimized_result['total_cost'],
            "time_to_optimize_sec": optimized_result['time_taken_sec']
        })


    with open(f"dataset/nwcm_solution_{problem_id}.txt", "a") as f:
        f.write("\nNorth-West Corner Method Allocation:\n")
        for row in nwcm_result['allocation']:
            f.write(" ".join(f"{x:.2f}" for x in row) + "\n")
        f.write(f"Total Cost: {nwcm_result['total_cost']}\n")
        f.write(f"Time taken: {nwcm_result['time_taken_sec']} seconds\n")
        f.write("\nMODI Method Optimized Allocation (from NWCM):\n")
        for row in optimized_result['allocation']:
            f.write(" ".join(f"{x:.2f}" for x in row) + "\n")
        f.write(f"Total Cost after MODI Optimization: {optimized_result['total_cost']}\n")
        f.write(f"Time taken for MODI optimization: {optimized_result['time_taken_sec']} seconds\n")  

def solve_lcm(supply, demand, cost, problem_id):
    # Solve using Least Cost Method
    lcm_result = least_cost_method(supply, demand, cost)
    optimized_result = modi_optimize_from_ibfs({
        "allocation":       lcm_result["allocation"],
        "balanced_cost":    lcm_result["balanced_cost"],
        "balanced_supply":  lcm_result["balanced_supply"],
        "balanced_demand":  lcm_result["balanced_demand"],
    }, trace=False)
    with open(f"dataset/transportation_results.csv", "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["problem_id", "method", "rows", "comlumns", "total_cost", "time_taken_sec", "optimized_cost", "time_to_optimize_sec"])
        writer.writerow({
            "problem_id": problem_id,
            "method": "LCM",
            "rows": len(lcm_result['balanced_supply']),
            "comlumns": len(lcm_result['balanced_demand']),
            "total_cost": lcm_result['total_cost'],
            "time_taken_sec": lcm_result['time_taken_sec'],
            "optimized_cost": optimized_result['total_cost'],
            "time_to_optimize_sec": optimized_result['time_taken_sec']
        })

    with open(f"dataset/lcm_solution_{problem_id}.txt", "a") as f:
        f.write("\nLeast Cost Method Allocation:\n")
        for row in lcm_result['allocation']:
            f.write(" ".join(f"{x:.2f}" for x in row) + "\n")
        f.write(f"Total Cost: {lcm_result['total_cost']}\n")
        f.write(f"Time taken: {lcm_result['time_taken_sec']} seconds\n")
        f.write("\nMODI Method Optimized Allocation (from LCM):\n")
        for row in optimized_result['allocation']:
            f.write(" ".join(f"{x:.2f}" for x in row) + "\n")
        f.write(f"Total Cost after MODI Optimization: {optimized_result['total_cost']}\n")
        f.write(f"Time taken for MODI optimization: {optimized_result['time_taken_sec']} seconds\n")


def solve_vam(supply, demand, cost, problem_id):
    # Solve using Vogel's Approximation Method
    vam_result = vogels_approximation_method(supply, demand, cost)
    optimized_result = modi_optimize_from_ibfs({
        "allocation":       vam_result["allocation"],
        "balanced_cost":    vam_result["balanced_cost"],
        "balanced_supply":  vam_result["balanced_supply"],
        "balanced_demand":  vam_result["balanced_demand"],
    }, trace=False)
    with open(f"dataset/transportation_results.csv", "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["problem_id", "method", "rows", "comlumns", "total_cost", "time_taken_sec", "optimized_cost", "time_to_optimize_sec"])
        writer.writerow({
            "problem_id": problem_id,
            "method": "VAM",
            "rows": len(vam_result['balanced_supply']),
            "comlumns": len(vam_result['balanced_demand']),
            "total_cost": vam_result['total_cost'],
            "time_taken_sec": vam_result['time_taken_sec'],
            "optimized_cost": optimized_result['total_cost'],
            "time_to_optimize_sec": optimized_result['time_taken_sec']
        })

    with open(f"dataset/vam_solution_{problem_id}.txt", "a") as f:
        f.write("\nVogel's Approximation Method Allocation:\n")
        for row in vam_result['allocation']:
            f.write(" ".join(f"{x:.2f}" for x in row) + "\n")
        f.write(f"Total Cost: {vam_result['total_cost']}\n")
        f.write(f"Time taken: {vam_result['time_taken_sec']} seconds\n")
        f.write("\nMODI Method Optimized Allocation (from VAM):\n")
        for row in optimized_result['allocation']:
            f.write(" ".join(f"{x:.2f}" for x in row) + "\n")
        f.write(f"Total Cost after MODI Optimization: {optimized_result['total_cost']}\n")
        f.write(f"Time taken for MODI optimization: {optimized_result['time_taken_sec']} seconds\n")

def itterate():
    # Generate a large transportation problem
    supply, demand, cost = generate_transportation_problem(balance=True)

    problem_id = uuid.uuid4().hex[:6]
    print(f"Generated Transportation Problem ID: {problem_id}")

    # Save problem to a text file
    with open(f"dataset/transport_problem_{problem_id}.txt", "w") as f:
        f.write("Supply:\n")
        f.write(" ".join(map(str, supply)) + "\n")
        f.write("Demand:\n")
        f.write(" ".join(map(str, demand)) + "\n")
        f.write("Cost Matrix:\n")
        for row in cost:
            f.write(" ".join(map(str, row)) + "\n")
    
    # Solve using North-West Corner Method
    solve_nwcm(supply, demand, cost, problem_id)

    # Solve using Least Cost Method
    solve_lcm(supply, demand, cost, problem_id)

    # Solve using Vogel's Approximation Method
    solve_vam(supply, demand, cost, problem_id)


if __name__ == "__main__":
    create_csv()
    for _ in range(100):  # Generate and solve 5 problems
        itterate()
    plot_transport_times_with_diff_dual_axis()
    
