using Revise

using MCTS
using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test
using D3Trees

##

n_iter = 10000
depth = 15
ec = 10.0

rng=MersenneTwister(54)
rng_dpw = deepcopy(rng)

mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )
state = GridWorldState(3,4)


n_s = length(MCTS.convert_state(state))
n_a = n_actions(mdp)

estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
estimator = NNEstimator(rng, estimator_path, n_s, n_a)

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/SavedNetwork_180528_203048_10000")


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


policy = solve(solver, mdp)

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

a_dpw, ai_dpw = action_info(policy_dpw, state)

estimator.py_class[:debug_print_n_calls]()

inchromium(D3Tree(ai_dpw[:tree],init_expand=1))
