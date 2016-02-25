using MCTS
using POMDPModels
using Base.Test

n_iter = 100
depth = 15
ec = 1.0

solver = MCTSSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec)
mdp = GridWorld()

policy = solve(solver, mdp)

state = GridWorldState(1,1)

@time a = action(policy, state)

include("dpw_test.jl")
include("aggregation_test.jl")
