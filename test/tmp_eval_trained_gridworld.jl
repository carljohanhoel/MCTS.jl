using PyCall
using MCTS
using POMDPModels

# mdp = GridWorld(5,5,
#                 penalty=-1.,
#                 rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
#                 )
mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
state = GridWorldState(1,1)

vec_state = MCTS.convert_state(state,mdp)

n_s = length(vec_state)
n_a = n_actions(mdp)
v_min = -10.
v_max = 10.
replay_memory_max_size = 55
training_start = 40
rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")

# load_network = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180605_020108_same_as_previous_but_queue_in_py/100000"
# load_network = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180605_224505_loss_value_100/45000"
# load_network = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180608_191845_rolled_back_from_parallel/30007"

estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180530_022108/50001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180531_021250/65003")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180531_025035/55004")

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180531_230831_gridworld_std/75002")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180531_232726_sgd/75001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180601_010824_dirichlet_noise_added/70012")

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180608_191845_rolled_back_from_parallel/30007")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180609_022938_smaller_replay_mem/100008")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005148_std/100016")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005257_100_updates_per_episode/100001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180618_195010_10000_mcts_searches_100_updates/20020")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180618_195141_2000_mcts_iterations_100_updates/75001")
load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180620_210355_2000_mcts_searches_100_updates_5_puct/25000")



allowed_actions = [1.0, 1.0, 1.0, 1.0]
##
for y = 5:-1:1
   for x = 1:5
      state = GridWorldState(x,y)
      vec_state = MCTS.convert_state(state,mdp)
      v = estimate_value(estimator,state,mdp)
      @printf("%.2f",v[1])
      print(" ")
   end
   println()
end
println()

for y = 5:-1:1
   for x = 1:5
      state = GridWorldState(x,y)
      vec_state = MCTS.convert_state(state,mdp)
      p,v = estimator.py_class[:forward_pass](vec_state)
      print(indmax(p))
      print(" ")
   end
   println()
end
println(actions(mdp))
