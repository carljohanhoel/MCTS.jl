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

replay_memory_max_size = 55
training_start = 40
training_steps = 100
save_freq = 20
eval_freq = 20
eval_eps = 3
# replay_memory_max_size = 100000
# training_start = 5000
# training_steps = 100000
# save_freq = 5000
# eval_freq = 5000
# eval_eps = 100


sim_max_steps = 25



rng=MersenneTwister(53)

mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
s_initial = GridWorldState(1,1)

n_s = length(MCTS.convert_state(s_initial))
n_a = n_actions(mdp)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start)

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


sim = HistoryRecorder(rng=rng, max_steps=sim_max_steps, show_progress=false)

##
trainer = Trainer(rng=rng, training_steps=training_steps, save_freq=save_freq, eval_freq=eval_freq, eval_eps=eval_eps, show_progress=true, log_dir=log_path)
train(trainer, sim, mdp, policy)
