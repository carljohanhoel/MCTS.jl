parallel_version = true   #Test code in parallel mode
# parallel_version = false

if parallel_version
   addprocs(3)
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
v_max = 1*1.05
v_min = -v_max
replay_memory_max_size = 55
training_start = 40
rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
if parallel_version
   @spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, false),cmd_queue,res_queue)
   estimator = NNEstimatorParallel(v_min, v_max)
   sleep(10) #Wait for queue to be set up before continuing
   clear_queue() #Something strange makes this necessary...
else
   estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005257_100_updates_per_episode/100001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180629_024700_2000_mcts_searches_100_updates_2_puct_8_queues/5001")

allowed_actions = ones(1,4)
##
N_samples = 25
idx_x = [1,2,3,4,5]
idx_y = [1,2,3,4,5]
idx_x = shuffle(rng,idx_x)
idx_y = shuffle(rng,idx_y)
state = Array{GridWorldState}(N_samples)
for i in 1:5
   for j in 1:5
      state[i+5*(j-1)] = GridWorldState(idx_x[i],idx_y[j])
   end
end

dist = rand(rng,N_samples,n_a)
dist = dist./sum(dist,2)
val = 2*rand(rng,N_samples)-1

##
v = estimate_value(estimator,state[1],mdp)[1]
p = estimate_distribution(estimator,state[1], allowed_actions,mdp)
println(v)
println(p)
println(val[1])
println(dist[1,:])
add_samples_to_memory(estimator,state,dist,val,mdp)
add_samples_to_memory(estimator,state,dist,val,mdp)
##
if parallel_version
   @spawnat 3 update_network(estimator,100)
else
   update_network(estimator,10000)
end
##
v_vec = Array{Float64}(N_samples)
p_vec = Array{Float64}(N_samples,4)
for i in 1:N_samples
   v_vec[i] = estimate_value(estimator,state[i],mdp)[1]
   p_vec[i,:] = estimate_distribution(estimator,state[i], allowed_actions,mdp)
end
v_err = val-v_vec
p_err = dist-p_vec
sv_err = sum(abs.(v_err))
sp_err = sum(abs.(p_err))

stash_size = 2
set_stash_size(estimator,stash_size)

out1 = @spawnat 3 estimate_distribution(estimator,state[1], allowed_actions,mdp)
out2 = @spawnat 4 estimate_distribution(estimator,state[2], allowed_actions,mdp)
fetch(out1)
fetch(out2)
#
# out3 = @spawnat 3 estimate_value(estimator,state[1],mdp)
# fetch(out3)
