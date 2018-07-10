using Revise

using MCTS
using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test
using D3Trees

##

n_iter = 2000
depth = 15
ec =  1/20*5#100.#2.#5.#10.0
ec_dpw = ec*20

rng=MersenneTwister(65)
rng_dpw = deepcopy(rng)

mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
state = GridWorldState(3,3)


n_s = length(MCTS.convert_state(state, mdp))
n_a = n_actions(mdp)
v_max = 1*1.05
v_min = -v_max
replay_memory_max_size = 55
training_start = 40
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180530_022108/50001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180530_200610/10001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180530_232149/5014")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180531_025035/45001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180601_010824_dirichlet_noise_added/70012")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180609_022938_smaller_replay_mem/100008")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005257_100_updates_per_episode/100001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180618_195141_2000_mcts_iterations_100_updates/75001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180703_190208_not_parallel_weights_1_100/40011")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180621_181229_2000_mcts_iterations_100_updates_2_puct/100001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180707_011126_serial_weights_1_1_5puct_new_gridworld/40006")

solver = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec,
                k_state=3.,
                tree_in_info=true,
                alpha_state=0.2,
                enable_action_pw=false,
                check_repeat_state=false,
                rng=rng,
                estimate_value=estimator,
                init_P=estimator,
                noise_dirichlet = 4,
                noise_eps = 0.25
                )


policy = solve(solver, mdp)
policy.training_phase = false   #if false, evaluate trained agent, no randomness in action choices

a, ai = action_info(policy, state)
inchromium(D3Tree(ai[:tree],init_expand=1))

##
#DPW reference, just uncomment to run comparison

solver_dpw = DPWSolver(n_iterations=n_iter, depth=depth, exploration_constant=ec_dpw,
                  tree_in_info=true,
                  k_state=3.,
                  alpha_state=0.2,
                  enable_action_pw=false,
                  check_repeat_state=false,
                  rng=rng_dpw
                  )

policy_dpw = solve(solver_dpw, mdp)

a_dpw, ai_dpw = action_info(policy_dpw, state)
inchromium(D3Tree(ai_dpw[:tree],init_expand=1))
