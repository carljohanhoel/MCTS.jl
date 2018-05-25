#Neural network trainer

"""
Trains the neural network by:
1. Running through one episode, using APV-MCTS and the previous network
2. Calculating the actual value and estimated action distribution for every state
3. Adding new samples to the replay memory
4. Updating the network
"""

mutable struct Trainer
    rng::AbstractRNG
    n_steps::Int


    # options
    show_progress::Bool

    # optional: if these are null, they will be ignored
    initial_state_generator::Nullable{Any}

end

function Trainer(;rng=MersenneTwister(rand(UInt32)),
                  n_steps::Int=1,
                  show_progress=false,
                  initial_state_generator=Nullable{Any}(),
                 )
    return Trainer(rng, n_steps, show_progress, initial_state_generator)
end


function train{S,A}(trainer::Trainer,
                    sim::HistoryRecorder,
                    mdp::MDP{S,A}, policy::Policy
                   )
    n_steps = trainer.n_steps
    if trainer.show_progress
        prog = POMDPToolbox.Progress(n_steps, "Training..." )
    end

    step = 1
    while step <= n_steps
        #Generate initial state
        #ZZZ
        initial_state = GridWorldState(1,1)
        #ZZZ

        #Simulate one episode
        hist = POMDPs.simulate(sim, mdp, policy, initial_state)

        #Extract training samples
        new_states = hist.state_hist
        new_values = Vector{Float64}(length(new_states))
        new_distributions = Array{Float64}(length(new_states)-1,length(actions(mdp)))

        end_state = new_states[end]
        end_value = end_state.done ? 0 : estimate_value(policy.solver.estimate_value, end_state)
        new_values[end] = end_value
        value = end_value
        for (i,state) in enumerate(new_states[end-1:-1:1])
           value = hist.reward_hist[end+1-i] + mdp.discount_factor*value
           new_values[end-i] = value
           new_distributions[i,:] = hist.ainfo_hist[i][:action_distribution]
        end

        #Update network
        update_network(policy.solver.estimate_value, new_states[1:end-1], new_distributions, new_values[1:end-1])

        step += 1

        if trainer.show_progress
            POMDPToolbox.ProgressMeter.next!(prog)
        end
    end

    if sim.show_progress
        POMDPToolbox.ProgressMeter.finish!(prog)
    end
end
