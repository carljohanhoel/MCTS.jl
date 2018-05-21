### Neural network estimator ###
"""
Call to python for a neural network estimator of state value and action probabilities
"""
mutable struct NNEstimator
    rng::AbstractRNG
    python_module::Module
end

function NNEstimator(rng::AbstractRNG, estimator_path::String) #
    python_module = initialize_estimator(estimator_path)
    return NNEstimator(rng, python_module)
end

function initialize_estimator(estimator_path::String)
    unshift!(PyVector(pyimport("sys")["path"]), dirname(estimator_path))
    eval(parse(string("@pyimport ", basename(estimator_path), " as python_module")))
    return python_module
end


estimate_value(estimator::NNEstimator, mdp::MDP, state, depth::Int) = estimate_value(estimator, state)

function estimate_value(estimator::NNEstimator, state)
    value = estimator.python_module.estimate_value(state)
    return value   #ZZZ Fiz, get from NN
end

function estimate_probabilities(estimator::NNEstimator, state, possible_actions)
    probabilities = estimator.python_module.estimate_probabilities(state,possible_actions)
    return probabilities
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
