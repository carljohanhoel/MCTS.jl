### Neural network estimator ###
"""
Call to python for a neural network estimator of state value and action probabilities
"""
mutable struct NNEstimator
    rng::AbstractRNG
    py_class::PyCall.PyObject
    estimator_path::String
end

function NNEstimator(rng::AbstractRNG, estimator_path::String, log_path::String, n_states::Int, n_actions::Int, replay_memory_max_size::Int, training_start::Int) #
    py_class = initialize_estimator(estimator_path, log_path, n_states, n_actions, replay_memory_max_size, training_start)
    return NNEstimator(rng, py_class, estimator_path)
end

function initialize_estimator(estimator_path::String, log_path::String, n_states::Int, n_actions::Int, replay_memory_max_size::Int, training_start::Int)
    unshift!(PyVector(pyimport("sys")["path"]), dirname(estimator_path))
    eval(parse(string("@pyimport ", basename(estimator_path), " as python_module")))
    py_class = python_module.NNEstimator(n_states, n_actions, replay_memory_max_size, training_start, log_path)
    return py_class
end


estimate_value(estimator::NNEstimator, p::Union{POMDP,MDP}, state, depth::Int) = estimate_value(estimator, state, p)

function estimate_value(estimator::NNEstimator, state, p::Union{POMDP,MDP})
    converted_state = convert_state(state, p)
    value = estimator.py_class[:estimate_value](converted_state)
    return value
end

function estimate_distribution(estimator::NNEstimator, state, allowed_actions, p::Union{POMDP,MDP})
    converted_state = convert_state(state, p)
    dist = estimator.py_class[:estimate_distribution](converted_state,allowed_actions)
    return dist
end

function add_samples_to_memory(estimator::NNEstimator, states, dists, vals, p)
    converted_states = convert_state(states, p)
    estimator.py_class[:add_samples_to_memory](converted_states, dists, vals)
end

function update_network(estimator::NNEstimator)
    estimator.py_class[:update_network]()
end

function save_network(estimator::NNEstimator, name::String)
    estimator.py_class[:save_network](name)
end

function load_network(estimator::NNEstimator, name::String)
    estimator.py_class[:load_network](name)
end

#Needs to be defined for each problem to fit the input of the nerual network
function convert_state(state::Type, p::Union{POMDP,MDP})
    converted_state = state
    return converted_state
end

function state_dist() #Dummy function, to be defined for each problem
end

#Simple example for GridWorld, here for tests. Remove later.
using POMDPModels
function convert_state(state::Vector{GridWorldState}, mdp::GridWorld)
    n = length(state)
    converted_state = Array{Float64}(n,3)
    for i in 1:n
        converted_state[i,:] = convert_state(state[i], mdp)
    end
    return converted_state
end
function convert_state(state::GridWorldState, mdp::GridWorld)
    converted_state = Array{Float64}(1,3)
    converted_state[1] = state.x
    converted_state[2] = state.y
    converted_state[3] = state.done ? 1 : 0
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
