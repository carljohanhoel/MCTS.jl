using Revise

# parallel_version = true   #Test code in parallel mode
parallel_version = false

if parallel_version
   addprocs(2)
   @everywhere using MCTS
else
   using MCTS
end

using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test
using D3Trees

##

n_iter = 2000
depth = 15
c_puct = 1/20*5 #5.#2.#5. #10.0
tau = 1.1

rng=MersenneTwister(54)

mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
initial_state = GridWorldState(5,1)

n_s = length(MCTS.convert_state(initial_state, mdp))
n_a = n_actions(mdp)
v_max = 1*1.05
v_min = -v_max
replay_memory_max_size = 55
training_start = 40
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
if parallel_version
   @spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, false),cmd_queue,res_queue)
   estimator = NNEstimatorParallel(v_min, v_max)
else
   estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180616_005257_100_updates_per_episode/100001")
load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180718_180800_20_workers_weights_1_10_puct_0p25_10_updates_per_sample_tau_1p1_stash_1_1p5/3507")

solver = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=c_puct,
               k_state=3.,
               tree_in_info=true,
               alpha_state=0.2,
               tau=tau,
               enable_action_pw=false,
               check_repeat_state=false,
               rng=rng,
               estimate_value=estimator,
               init_P=estimator,
               noise_dirichlet = 4,
               noise_eps = 0.25
               )
policy = solve(solver, mdp)


##
policy.training_phase = false   #if false, evaluate trained agent, no randomness in action choices
sim = HistoryRecorder(rng=rng, max_steps=25, show_progress=true)
hist = simulate(sim, mdp, policy, initial_state)

println("sim done")


step = 5
print(hist.action_hist[step])
inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))

##Extract training samples
p = mdp #to make code the same as in trainer
nn_estimator = estimator #to make code the same as in trainer
n_a = length(actions(p))
all_actions = actions(policy.mdp)
n_new_samples = length(hist.state_hist)
new_states = deepcopy(hist.state_hist)
new_z_values = Vector{Float64}(length(new_states))
new_q_values = Vector{Float64}(length(new_states))
new_distributions = Array{Float64}(length(new_states),n_a)
##
end_state = new_states[end]
end_value = isterminal(p,end_state) ? 0 : estimate_value(nn_estimator, end_state, p)[1]
new_z_values[end] = end_value
value = end_value
new_q_values[end] = end_value
## for (i,state) in enumerate(new_states[end-1:-1:1])
for i in 1:length(new_states)-1
   value = hist.reward_hist[end+1-i] + p.discount*value
   new_z_values[end-i] = value
   new_distributions[end-i,:] = hist.ainfo_hist[end+1-i][:action_distribution]
   a_idx = findfirst(all_actions,hist.action_hist[end+1-i])
   new_q_values[end-i] = hist.ainfo_hist[end+1-i][:q_values][a_idx]
end

if isterminal(p,end_state) #If terminal state, keep value 0 and add dummy distribution, otherwise remove last sample (the simulation gives no information about it)
   new_distributions[end,:] = ones(1,n_a)/n_a
else
   pop!(new_states)
   pop!(new_z_values)
   pop!(new_q_values)
   new_distributions = new_distributions[1:end-1,:]
   n_new_samples-=1
end
new_values = new_z_values
new_values_combined = (new_z_values+new_q_values)/2


######################
##Extract training samples OLD
n_a = length(actions(mdp))
new_states2 = deepcopy(hist.state_hist)
new_values2 = Vector{Float64}(length(new_states))
new_distributions2 = Array{Float64}(length(new_states),n_a)

end_state = new_states2[end]
end_value = isterminal(mdp,end_state) ? 0 : estimate_value(solver.estimate_value, end_state, mdp)[1]
new_values2[end] = end_value
value = end_value
for (i,state) in enumerate(new_states2[end-1:-1:1])
   # print(state)
   value = hist.reward_hist[end+1-i] + mdp.discount*value
   new_values2[end-i] = value
   new_distributions2[i,:] = hist.ainfo_hist[i][:action_distribution]
end

if isterminal(mdp,end_state) #If terminal state, keep value 0 and add dummy distribution, otherwise remove last sample
   new_distributions2[end,:] = ones(1,n_a)/n_a
else
   pop!(new_states2)
   pop!(new_values2)
   new_distributions2 = new_distributions2[1:end-1,:]
end

####################





##
# #Update
# println("Update network")
# update_network(solver.estimate_value, new_states, new_distributions, new_values)

##
if parallel_version
   allowed_actions = ones(1,4)
   p = estimate_distribution(estimator, initial_state, allowed_actions, mdp)
   v = estimate_value(estimator,initial_state, mdp)[1]
   println(p)
   println(v)
else
   vec_state =  MCTS.convert_state(initial_state, mdp)
   allowed_actions = ones(1,4)
   p,v = estimator.py_class[:forward_pass](vec_state)
   println(p)
   println(v)
end
