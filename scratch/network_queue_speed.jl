parallel_version = true   #Test code in parallel mode
# parallel_version = false

if parallel_version
   n_workers = 20
   # n_workers = 4
   # n_workers = 2
   addprocs(n_workers+1)
   @everywhere using MCTS
else
   using MCTS
end
@everywhere using POMDPModels

@everywhere mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
state = GridWorldState(4,1)

vec_state = MCTS.convert_state(state, mdp)

n_s = length(vec_state)
n_a = n_actions(mdp)
@everywhere v_max = 1*1.05
@everywhere v_min = -v_max
replay_memory_max_size = 55
training_start = 40
rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
if parallel_version
   @spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, false),cmd_queue,res_queue)
   @everywhere estimator = NNEstimatorParallel(v_min, v_max)
   sleep(10) #Wait for queue to be set up before continuing
   clear_queue() #Something strange makes this necessary...
else
   estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005257_100_updates_per_episode/100001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180629_024700_2000_mcts_searches_100_updates_2_puct_8_queues/5001")

allowed_actions = ones(1,4)
##
@everywhere N_samples = 25
idx_x = [1,2,3,4,5]
idx_y = [1,2,3,4,5]
idx_x = shuffle(rng,idx_x)
idx_y = shuffle(rng,idx_y)
@everywhere state = Array{GridWorldState}(N_samples)
for i in 1:5
   for j in 1:5
      state[i+5*(j-1)] = GridWorldState(idx_x[i],idx_y[j])
   end
end

dist = rand(rng,N_samples,n_a)
dist = dist./sum(dist,2)
val = 2*rand(rng,N_samples)-1



##
add_samples_to_memory(estimator,state,dist,val,mdp)
add_samples_to_memory(estimator,state,dist,val,mdp)
##
stash_size = 1
set_stash_size(estimator,stash_size)

@everywhere function test_estimate_value(n::Int)
   for i in 1:n
      v = estimate_value(estimator,state[1],mdp)[1]
   end
end

function test_update_network(n::Int)
   for i in 1:n
      v = update_network(estimator,10)
   end
end

@time test_estimate_value(1)
@time test_estimate_value(1)
@time test_estimate_value(1000)
@time test_estimate_value(1000)

@time test_update_network(1)
@time test_update_network(1)
@time test_update_network(100)
@time test_update_network(100)

##
stash_size = min(13,n_workers)
set_stash_size(estimator,stash_size)

function test_estimate_value_parallel(n::Int)
   out = @spawnat mod(1,n_workers)+3 estimate_value(estimator,state[1], mdp)
   for i in 2:n-1
      out = @spawnat mod(i,n_workers)+3 estimate_value(estimator,state[1], mdp)
   end
   out = @spawnat mod(n,n_workers)+3 estimate_value(estimator,state[2], mdp)
   fetch(out)
   # println(out)
end

#This is the test that really matters
function test_estimate_value_parallel_2(n::Int)
   out = @spawnat mod(1,n_workers)+3 test_estimate_value(div(n,n_workers))
   for i in 2:n_workers-1
      @spawnat mod(i,n_workers)+3 test_estimate_value(div(n,n_workers))
   end
   out = @spawnat mod(n,n_workers)+3 test_estimate_value(div(n,n_workers))
   # #Below to make sure all workers can finish (does not quite work...)
   # @spawnat mod(1,n_workers)+3 test_estimate_value(1)
   # for i in 2:n_workers-1
   #    @spawnat mod(i,n_workers)+3 test_estimate_value(1)
   # end
   # @spawnat mod(n,n_workers)+3 test_estimate_value(1)
   fetch(out)
end

@time test_estimate_value_parallel(stash_size)
@time test_estimate_value_parallel(stash_size)
@time test_estimate_value_parallel(1000)
@time test_estimate_value_parallel(1000)
# @time test_estimate_value_parallel(25000)

@time test_estimate_value_parallel_2(1)
@time test_estimate_value_parallel_2(1600)
@time test_estimate_value_parallel_2(16000)


# julia> set_stash_size(estimator,1)
# RemoteChannel{Channel{MCTS.QueueCommand}}(1, 1, 1)
#
# julia> @time test_estimate_value_parallel_2(16000)
#  46.835212 seconds (4.17 M allocations: 424.700 MiB, 0.09% gc time)
#
# julia> set_stash_size(estimator,4)
# RemoteChannel{Channel{MCTS.QueueCommand}}(1, 1, 1)
#
# julia> @time test_estimate_value_parallel_2(16000)
#  17.664934 seconds (4.15 M allocations: 421.186 MiB, 0.26% gc time)
#
# julia> set_stash_size(estimator,8)
# RemoteChannel{Channel{MCTS.QueueCommand}}(1, 1, 1)
#
# julia> @time test_estimate_value_parallel_2(16000)
#  12.285509 seconds (4.16 M allocations: 423.925 MiB, 0.36% gc time)
#
# julia> set_stash_size(estimator,16)
# RemoteChannel{Channel{MCTS.QueueCommand}}(1, 1, 1)


#old queue, 4 workers, 4 stash size
# 0.884404 seconds (265.55 k allocations: 23.711 MiB, 0.56% gc time)
#   2.296536 seconds (682.19 k allocations: 56.819 MiB, 0.50% gc time)
#  18.446616 seconds (4.11 M allocations: 401.742 MiB, 0.21% gc time)
#  18.155481 seconds (4.11 M allocations: 402.253 MiB, 0.29% gc time)
#  18.785550 seconds (4.17 M allocations: 404.900 MiB, 0.27% gc time)
#20 workers, 13 stash size
# 1.122411 seconds (408.18 k allocations: 41.992 MiB, 0.73% gc time)
# 10.531167 seconds (4.11 M allocations: 422.885 MiB, 0.42% gc time)
# 10.818520 seconds (4.22 M allocations: 427.842 MiB, 0.50% gc time)



#new queue with pycall, 4 workers, 4 stash size
# 3.101693 seconds (946.26 k allocations: 70.939 MiB, 0.71% gc time)
# 19.948838 seconds (5.19 M allocations: 438.569 MiB, 0.34% gc time)
# 20.162086 seconds (5.20 M allocations: 439.717 MiB, 0.39% gc time)
#20 workers, 13 stash size
# 1.444715 seconds (526.30 k allocations: 47.317 MiB, 0.66% gc time)
# 13.791815 seconds (5.18 M allocations: 465.862 MiB, 0.52% gc time)
# 14.247861 seconds (5.19 M allocations: 466.926 MiB, 0.60% gc time)
