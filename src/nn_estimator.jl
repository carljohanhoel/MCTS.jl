### Neural network estimator ###
"""
Neural network estimator of state value and action probabilities
"""
mutable struct NNEstimator
    rng::AbstractRNG
end

estimate_value(estimator::NNEstimator, mdp::MDP, state, depth::Int) = estimate_value(estimator, state)

function estimate_value(estimator::NNEstimator, state)
    return 0.0   #ZZZ Fiz, get from NN
end

function estimate_probabilities(estimator::NNEstimator, mdp, state, possible_actions)
    A = action_type(mdp)
    n_actions = length(possible_actions)
    probabilities = Vector{Float64}(n_actions)
    for i in 1:n_actions
        probabilities[i] = 1/n_actions   #ZZZ Fiz, get from NN
    end
    return probabilities
end


### Neural network policy ###
"""
Policy that picks the action with highest probability from the neural network.
"""
#ZZZ Not implemented, just a placeholder. Now outputs random action!
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
