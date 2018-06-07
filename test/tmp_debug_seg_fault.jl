using PyCall
using MCTS
using POMDPModels

mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
state = GridWorldState(1,1)

vec_state = MCTS.convert_state(state)

n_s = length(vec_state)
n_a = n_actions(mdp)
replay_memory_max_size = 55
training_start = 40
rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start)

allowed_actions = [1.0, 1.0, 1.0, 1.0]

v = estimator.py_class[:estimate_value](vec_state)
p = estimator.py_class[:estimate_distribution](vec_state, allowed_actions)
println(v)
println(p)

estimator.py_class[:debug_save_input](vec_state, allowed_actions)
estimator.py_class[:debug_print_n_calls]()

estimator.py_class[:save_network](dirname(dirname(estimator_path))*"/Logs/ttt")

new_states = Vector{GridWorldState}(5)
for i in 1:length(new_states)
   new_states[i] = GridWorldState(5,1)
end
new_values = ones(5)
new_distributions = ones(5,4)

##
#Update
println("Update network")

converted_states = convert_state(new_states)

for i in 1:1000
   # tic()
   # print("Call no ")
   # println(i)
   estimator.py_class[:update_network](converted_states, new_distributions, new_values)
   # update_network(estimator, new_states, new_distributions, new_values)
   # toc()
end
update_network(estimator, new_states, new_distributions, new_values)
#
estimator.py_class[:estimate_value](vec_state)
# sleep(5)
# terminate_estimator(estimator)

estimator.py_class[:debug_print_n_calls]()
