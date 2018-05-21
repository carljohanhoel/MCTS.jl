using Revise

using MCTS
using POMDPs
using POMDPModels
using Base.Test
using NBInclude
using POMDPToolbox
using D3Trees

##

n_iter = 5000
depth = 15
ec = 100.0

rng=MersenneTwister(53)
rng_dpw = deepcopy(rng)

estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
estimator = NNEstimator(rng, estimator_path)

solver = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec,
                  k_state=3.,
                  tree_in_info=true,
                  alpha_state=0.2,
                  enable_action_pw=false,
                  check_repeat_state=false,
                  rng=rng,
                  estimate_value=estimator,
                  init_P=estimator
                  )
mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )

policy = solve(solver, mdp)

state = GridWorldState(1,1)

a, ai = action_info(policy, state)
inchromium(D3Tree(ai[:tree],init_expand=1))

##
#DPW reference

solver_dpw = DPWSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec,
                  tree_in_info=true,
                  k_state=3.,
                  alpha_state=0.2,
                  enable_action_pw=false,
                  check_repeat_state=false,
                  rng=rng_dpw
                  )

policy_dpw = solve(solver_dpw, mdp)

a_dpw, ai_dpw = @inferred action_info(policy_dpw, state)

inchromium(D3Tree(ai_dpw[:tree],init_expand=1))
