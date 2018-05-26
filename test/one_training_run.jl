using Revise

using MCTS
using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test
using D3Trees

##

n_iter = 1000
depth = 15
c_puct = 10.0

rng=MersenneTwister(53)


mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )
s_initial = GridWorldState(1,1)

n_s = length(MCTS.convert_state(s_initial))
n_a = n_actions(mdp)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
estimator = NNEstimator(rng, estimator_path, n_s, n_a)

solver = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=c_puct,
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


sim = HistoryRecorder(rng=rng, max_steps=100, show_progress=false)

# trainer = Trainer(rng=rng, training_steps=2, save_freq=1, show_progress=true)
trainer = Trainer(rng=rng, training_steps=10000, save_freq=1000, show_progress=true)
train(trainer, sim, mdp, policy)
