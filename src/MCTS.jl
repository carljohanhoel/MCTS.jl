__precompile__()
module MCTS

using POMDPs
using POMDPToolbox

using PyCall
using StatsBase

using Compat
using Blink
using CPUTime

export
    MCTSSolver,
    MCTSPlanner,
    DPWSolver,
    DPWPlanner,
    AZSolver,
    AZPlanner,
    NNSolver,
    NNPolicy,
    NNEstimator,
    BeliefMCTSSolver,
    AbstractMCTSPlanner,
    AbstractMCTSSolver,
    solve,
    action,
    action_info,
    rollout,
    StateNode,
    RandomActionGenerator,
    RolloutEstimator,
    next_action,
    clear_tree!,
    estimate_value,
    init_N,
    init_Q,
    children,
    n_children,
    isroot,
    default_action

export
    AbstractStateNode,
    StateActionStateNode,
    DPWStateActionNode,
    DPWStateNode,
    AZStateActionNode,
    AZStateNode,

    ExceptionRethrow,
    ReportWhenUsed

abstract type AbstractMCTSPlanner{P<:Union{MDP,POMDP}} <: Policy end
abstract type AbstractMCTSSolver <: Solver end
abstract type AbstractStateNode end

include("requirements_info.jl")
include("nn_estimator.jl")
include("domain_knowledge.jl")
include("vanilla.jl")
include("dpw_types.jl")
include("dpw.jl")
include("az_types.jl")
include("az.jl")
include("action_gen.jl")
include("util.jl")
include("default_action.jl")
include("belief_mcts.jl")

include("visualization.jl")

end # module
