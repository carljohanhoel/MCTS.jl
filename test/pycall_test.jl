using PyCall
using MCTS
using POMDPModels

rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
estimator = NNEstimator(rng, estimator_path)

state = GridWorldState(1,1)
possible_actions = [1.,2,3,4]

v = estimator.py_class[:estimate_value](state)
p = estimator.py_class[:estimate_probabilities](state,possible_actions)
