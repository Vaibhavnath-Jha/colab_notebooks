###
# Solving Multi-Comodity Flow Problem with Benders Decomposition
###


import pandas as pd
import sys
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
    Domain,
    ModelStatus,
)


def benders_cflp(niter, filename=""):
    m = Container()

    data = pd.read_json(f"instances\\{filename}")

    # Sets
    i = Set(
        m,
        name="i",
        records=[f"i{i}" for i in range(data.params.I)],
        description="potential facility locations",
    )
    j = Set(
        m,
        name="j",
        records=[f"j{j}" for j in range(data.params.J)],
        description="customers",
    )

    # Parameters
    transportation_cost = Parameter(
        m,
        "c_ij",
        domain=[i, j],
        records=[
            [f"i{i}", f"j{j}", value]
            for i, row in enumerate(data.params.c_ij)
            for j, value in enumerate(row)
        ],
        description="unit transportation cost from facility i to customer j",
        is_miro_input=True,
    )

    demand = Parameter(
        m,
        "d_j",
        domain=j,
        records=[[f"j{j}", value] for j, value in enumerate(data.params.D_j)],
        description="demand of customer j",
    )

    opening_cost = Parameter(
        m,
        "f_i",
        domain=i,
        records=[[f"i{i}", value] for i, value in enumerate(data.params.F_i)],
        description="fixed cost for opening facility i",
    )

    capacity = Parameter(
        m,
        "q_i",
        domain=i,
        records=[[f"i{i}", value] for i, value in enumerate(data.params.Q_i)],
        description="capacity of facility i",
    )

    # Variables
    y = Variable(
        m,
        "y",
        type="binary",
        domain=i,
        description="is facility i open",
        is_miro_output=True,
    )

    theta = Variable(
        m,
        name="theta",
        # need to set some bound and the transportation cost never becomes negative
        type="positive",
        description="objective variable of sub problem",
    )

    #### Equations
    total_demand_met = Equation(
        m,
        name="total_demand_met",
        description="the total demand should be fulfilled by all facilities",
    )

    total_demand_met[...] = Sum(i, capacity[i] * y[i]) >= Sum(j, demand[j])

    obj_mp = Sum(i, opening_cost[i] * y[i]) + theta

    # Benders
    iter = Set(
        m,
        "iter",
        records=[f"iter{idx}" for idx in range(1, niter + 1)],
        description="max Benders iterations",
    )

    active_cut = Set(
        m,
        "active_cut",
        domain=iter,
        description="active benders cuts",
    )

    cut_const = Parameter(
        m,
        "cut_const",
        domain=iter,
        description="constants in active Benders cuts",
    )

    cut_coefficients = Parameter(
        m,
        "cut_coefficients",
        domain=[iter, i],
        description="coefficients in active Benders cuts",
    )

    optimality_cut = Equation(
        m,
        domain=iter,
        name="optimality_cut",
        description="optimality cut",
    )

    optimality_cut[active_cut] = theta >= cut_const[active_cut] + Sum(
        i, cut_coefficients[active_cut, i] * y[i]
    )

    # Masterproblem
    master_problem = Model(
        m,
        name="master_problem",
        equations=[optimality_cut, total_demand_met],
        problem="MIP",
        sense=Sense.MIN,
        objective=obj_mp,
    )

    # Subproblem
    x = Variable(
        m,
        "x",
        type="positive",
        domain=[i, j],
        description="volume demand of customer j assigned to facility i",
        is_miro_output=True,
    )

    demand_fulfilled = Equation(
        m,
        name="demand_fulfilled",
        domain=j,
        description="every customers demand should be fulfilled",
    )

    capacity_constraints = Equation(
        m,
        name="capacity_constraints",
        domain=i,
        description="facility can't deliver more than its capacity",
    )

    logical_linking = Equation(
        m,
        name="logical_linking",
        domain=[i, j],
        description="only opened facilities can deliver",
    )

    y_bar = Parameter(m, "y_bar", domain=i, description="level of y from MP solution")

    demand_fulfilled[j] = Sum(i, x[i, j]) == demand[j]

    capacity_constraints[i] = Sum(j, x[i, j]) <= capacity[i] * y_bar[i]

    logical_linking[i, j] = x[i, j] <= demand[j] * y_bar[i]

    obj_sp = Sum(Domain(i, j), transportation_cost[i, j] * x[i, j])

    sub_problem = Model(
        m,
        name="sub_problem",
        equations=[demand_fulfilled, capacity_constraints, logical_linking],
        problem="MIP",
        sense=Sense.MIN,
        objective=obj_sp,
    )

    for cut, _ in iter.records.itertuples(index=False):
        y.lo[i] = 0
        y.l[i] = 0
        y.up[i] = 1
        master_problem.solve(
            solver="CPLEX",
            # options=Options(equation_listing_limit=3)
        )
        # print(f"Cut Equation:\n{optimality_cut.getEquationListing()}")
        y_bar[i] = y.l[i]
        sub_problem.solve(solver="CPLEX")
        lb = master_problem.objective_value
        ub = lb + sub_problem.objective_value - theta.toValue("level")
        gap = abs(ub - lb) / abs(ub)
        print(f"LB: {lb:.3f} | UB: {ub:.3f} | GAP: {gap*100:.3f}%")
        abs_opt_gap = (
            1e-3  ### should be set to 1e-6, but this benders is slow!!! to converge
        )
        if sub_problem.status == ModelStatus(
            19
        ):  # 19 = ModelStatus.InfeasibleNoSolution
            raise Exception("Subproblem is infeasible!")
            ### feasibility cuts if infeasible, our MP should always provide feasible start points for SP
        elif gap < abs_opt_gap:
            print("Optimal Solution found!!!")
            sol_gdx = f"optimal_benders_sol_i{niter}_{filename}.gdx"
            m.write(sol_gdx)
            return ub, gap, sol_gdx
        else:
            active_cut[cut] = True
            cut_const[cut] = Sum(j, demand[j] * demand_fulfilled.m[j])
            cut_coefficients[cut, i] = capacity[i] * capacity_constraints.m[i] + Sum(
                j, demand[j] * logical_linking.m[i, j]
            )
            print(f"Adding cut >{cut}< to next iteration")

    sol_gdx = f"feasible_benders_sol_i{niter}_{filename}.gdx"
    m.write(sol_gdx)

    return ub, gap, sol_gdx


if __name__ == "__main__":
    niter = 10
    filename = "cap121.json"

    args = sys.argv[1:]

    for arg in args:
        if arg.startswith("--niter="):
            try:
                niter = int(arg.split("=")[1])
            except (ValueError, IndexError):
                print("Error: Invalid niter value. Using default.")
        elif arg.startswith("--filename="):
            filename = arg.split("=")[1]
        else:
            print(f"Warning: Unknown argument '{arg}'.")

    obj_val, gap, sol_gdx = benders_cflp(niter=niter, filename=filename)
    print(f"Objective value: {obj_val:.4f}\nGAP: {gap * 100:.3f}%")
