### Neural network estimator ###
"""
Call to python for a neural network estimator of state value and action probabilities
"""
mutable struct NNEstimator
    rng::AbstractRNG
    py_class::PyCall.PyObject
end

function NNEstimator(rng::AbstractRNG, estimator_path::String) #
    py_class = initialize_estimator(estimator_path)
    return NNEstimator(rng, py_class)
end

function initialize_estimator(estimator_path::String)
    unshift!(PyVector(pyimport("sys")["path"]), dirname(estimator_path))
    eval(parse(string("@pyimport ", basename(estimator_path), " as python_module")))
    py_class = python_module.NNEstimator()
    return py_class
end


estimate_value(estimator::NNEstimator, mdp::MDP, state, depth::Int) = estimate_value(estimator, state)

function estimate_value(estimator::NNEstimator, state)
    converted_state = convert_state(state)
    value = estimator.py_class[:estimate_value](converted_state)
    return value
end

function estimate_distribution(estimator::NNEstimator, state, allowed_actions)
    converted_state = convert_state(state)
    dist = estimator.py_class[:estimate_distribution](converted_state,allowed_actions)
    return dist
end

#Needs to be defined for each problem to fit the input of the nerual network
function convert_state(state::Type)
    converted_state = state
    return converted_state
end

#Simple example for GridWorld, here for tests. Remove later.
using POMDPModels
function convert_state(state::GridWorldState)
    converted_state = Array{Float64}(1,2)
    converted_state[1] = state.x
    converted_state[2] = state.y
    return converted_state
end

### Neural network policy ###
"""
Policy that picks the action with highest probability from the neural network.
"""
#ZZZ Not implemented, just a placeholder. Now outputs random action! ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
mutable struct NNPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
NNPolicy(problem::Union{POMDP,MDP};
             rng=Base.GLOBAL_RNG,
             updater=VoidUpdater()) = NNPolicy(rng, problem, updater)

## policy execution ##
function POMDPs.action(policy::NNPolicy, s)
    return rand(policy.rng, actions(policy.problem, s))
end

function POMDPs.action(policy::NNPolicy, b::Void)
    return rand(policy.rng, actions(policy.problem))
end

## convenience functions ##
updater(policy::NNPolicy) = policy.updater


"""
solver that produces a neural network policy
"""
mutable struct NNSolver <: Solver
    rng::AbstractRNG
end
NNSolver(;rng=Base.GLOBAL_RNG) = NNSolver(rng)
POMDPs.solve(solver::NNSolver, problem::Union{POMDP,MDP}) = NNPolicy(solver.rng, problem, VoidUpdater())
