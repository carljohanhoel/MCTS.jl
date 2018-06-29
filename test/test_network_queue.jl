addprocs(3)   #Seems to be necessary to add this first, before @everywhere using MCTS

@everywhere using MCTS

# @everywhere using POMDPs
# @everywhere using PyCall
# include("../src/network_queue.jl")
using POMDPModels

##
n_s = 3
n_a = 4
v_min = -10.
v_max = 10.
replay_memory_max_size = 100
training_start = 20
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_name = length(ARGS)>0 ? ARGS[1] : ""
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS_")*log_name
##
@spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, true),cmd_queue,res_queue)

##

put!(cmd_queue,("predict_distribution",myid(),rand(1,n_s),nothing,nothing,nothing,nothing,nothing))
take!(res_queue[myid()])
put!(cmd_queue,("predict_value",myid(),rand(1,n_s),nothing,nothing,nothing,nothing,nothing))
take!(res_queue[myid()])

@spawnat 3 put!(cmd_queue,("predict_distribution",myid(),rand(1,n_s),nothing,nothing,nothing,nothing,nothing))
@spawnat 3 put!(cmd_queue,("predict_value",myid(),rand(1,n_s),nothing,nothing,nothing,nothing,nothing))
out1 = @spawnat 3 take!(res_queue[myid()])
fetch(out1)
out2 = @spawnat 3 take!(res_queue[myid()])
fetch(out2)

##

@spawnat 4 put!(cmd_queue,("stash_size",myid(),nothing,nothing,nothing,nothing,1,nothing))

@spawnat 4 put!(cmd_queue,("add_samples_to_memory",myid(),nothing,rand(3,n_s),rand(3,n_s),rand(3,1),nothing,nothing))
out3 = @spawnat 4 take!(res_queue[myid()])
fetch(out3)

@spawnat 3 put!(cmd_queue,("update_network",myid(),nothing,nothing,nothing,nothing,nothing,nothing))
out3 = @spawnat 3 take!(res_queue[myid()])
fetch(out3)

out4 = @spawnat 3 put!(cmd_queue,("save",myid(),nothing,nothing,nothing,nothing,nothing,"../Logs/testSave3"))

out4 = @spawnat 4 put!(cmd_queue,("load",myid(),nothing,nothing,nothing,nothing,nothing,"../Logs/testSave3"))

##
#Higher level calls (queue initialized above)

mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
state = GridWorldState(5,4)

estimator = NNEstimatorParallel(v_min, v_max)

load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180620_210355_2000_mcts_searches_100_updates_5_puct/25000")

allowed_actions = ones(1,4)

estimate_distribution(estimator, state, allowed_actions, mdp)
estimate_value(estimator, state, mdp)
add_samples_to_memory(estimator, state, rand(1,4), rand(1,1), mdp)
update_network(estimator)
save_network(estimator, "../Logs/testSave4")
load_network(estimator, "../Logs/testSave4")
set_stash_size(estimator,1)

sleep(2)
print("Tests passed")
