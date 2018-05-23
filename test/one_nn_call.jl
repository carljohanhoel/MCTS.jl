using PyCall
using MCTS
using POMDPModels

#Simple example for GridWorld
function convert_state(state::GridWorldState)
    converted_state = Array{Float64}(1,2)
    converted_state[1] = state.x
    converted_state[2] = state.y
    return converted_state
end

rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
estimator = NNEstimator(rng, estimator_path)

mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )
state = GridWorldState(1,1)

vec_state = convert_state(state)

possible_actions = actions(mdp, vec_state)

v = estimator.py_class[:estimate_value](vec_state)
p = estimator.py_class[:estimate_probabilities](vec_state, possible_actions)
print(v)
print(p)

estimator.py_class[:debug_save_input](vec_state, possible_actions)
estimator.py_class[:debug_print_n_calls]()
