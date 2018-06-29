parallel_version = true   #Test code in parallel mode
# parallel_version = false

if parallel_version
   addprocs(2)
   @everywhere using MCTS
else
   using MCTS
end
using POMDPModels

mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
state = GridWorldState(4,1)

vec_state = MCTS.convert_state(state, mdp)

n_s = length(vec_state)
n_a = n_actions(mdp)
v_min = -10.
v_max = 10.
replay_memory_max_size = 55
training_start = 40
rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
if parallel_version
   @spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, true),cmd_queue,res_queue)
   estimator = NNEstimatorParallel(v_min, v_max)
else
   estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005257_100_updates_per_episode/100001")
load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180629_024700_2000_mcts_searches_100_updates_2_puct_8_queues/5001")

allowed_actions = ones(1,4)

if !parallel_version
   p,v = estimator.py_class[:forward_pass](vec_state)
   println(v)
   println(p)
   estimator.py_class[:save_network](dirname(dirname(estimator_path))*"/Logs/ttt")
   estimator.py_class[:load_network](dirname(dirname(estimator_path))*"/Logs/ttt")
end

estimate_value(estimator,state,mdp)
estimate_distribution(estimator,state, allowed_actions,mdp)
add_samples_to_memory(estimator,[state,state],rand(2,4),rand(2,1),mdp)
update_network(estimator)
save_network(estimator,dirname(dirname(estimator_path))*"/Logs/ttt")
load_network(estimator,dirname(dirname(estimator_path))*"/Logs/ttt")

sleep(2)
println("Tests passed")
