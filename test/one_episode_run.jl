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
rng_dpw = deepcopy(rng)


mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )
initial_state = GridWorldState(1,1)

n_s = length(MCTS.convert_state(initial_state))
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




sim = HistoryRecorder(rng=rng, max_steps=100, show_progress=true)
hist = simulate(sim, mdp, policy, initial_state)

println("sim done")
estimator.py_class[:debug_print_n_calls]()


step = 1
print(hist.action_hist[step])
inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))


#Extract training samples
new_states = hist.state_hist
new_values = Vector{Float64}(length(new_states))
new_distributions = Array{Float64}(length(new_states)-1,length(actions(mdp)))

end_state = new_states[end]
end_value = end_state.done ? 0 : estimate_value(solver.estimate_value, end_state)
new_values[end] = end_value
value = end_value
for (i,state) in enumerate(new_states[end-1:-1:1])
   # print(state)
   value = hist.reward_hist[end+1-i] + mdp.discount_factor*value
   new_values[end-i] = value
   new_distributions[i,:] = hist.ainfo_hist[i][:action_distribution]
end

##
#Update
println("Update network")
update_network(solver.estimate_value, new_states[1:end-1], new_distributions, new_values[1:end-1])

##
vec_state =  MCTS.convert_state(initial_state)
allowed_actions = [1.0, 1.0, 1.0, 1.0]
v = estimator.py_class[:estimate_value](vec_state)
p = estimator.py_class[:estimate_distribution](vec_state, allowed_actions)
println(v)
print(p)
