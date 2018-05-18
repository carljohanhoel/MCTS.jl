using Revise

using MCTS
using POMDPs
using POMDPModels
using Base.Test
using NBInclude
using POMDPToolbox
using D3Trees

n_iter = 5000
depth = 15
ec = 100.0

solver = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec,
                  tree_in_info=true,
                  k_state=3.,
                  alpha_state=0.2,
                  enable_action_pw=false,
                  check_repeat_state=false
                  )
mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )

policy = solve(solver, mdp)

state = GridWorldState(1,1)

a = @inferred action(policy, state)

a, ai = @inferred action_info(policy, state)
inchromium(D3Tree(ai[:tree],init_expand=1))


#DPW reference

solver_dpw = DPWSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec,
                  tree_in_info=true,
                  k_state=3.,
                  alpha_state=0.2,
                  enable_action_pw=false,
                  check_repeat_state=false
                  )

policy_dpw = solve(solver_dpw, mdp)

a_dpw, ai_dpw = @inferred action_info(policy_dpw, state)

inchromium(D3Tree(ai_dpw[:tree],init_expand=1))
