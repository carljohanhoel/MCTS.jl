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
    training_steps::Int
    save_freq::Int
    show_progress::Bool
end

function Trainer(;rng=MersenneTwister(rand(UInt32)),
                  training_steps::Int=1,
                  save_freq::Int=Inf,
                  show_progress=false,
                 )
    return Trainer(rng, training_steps, save_freq, show_progress)
end


function train{S,A}(trainer::Trainer,
                    sim::HistoryRecorder,
                    mdp::MDP{S,A}, policy::Policy
                   )
    training_steps = trainer.training_steps
    if trainer.show_progress
        prog = POMDPToolbox.Progress(training_steps, "Training..." )
    end

    step = 1
    while step <= training_steps
        #Generate initial state
        s_initial = initial_state(mdp,trainer.rng)

        #Simulate one episode
        hist = POMDPs.simulate(sim, mdp, policy, s_initial)

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

        if step%trainer.save_freq==0
            save_network(policy.solver.estimate_value, dirname(dirname(policy.solver.estimate_value.estimator_path))*"/Logs/ttt")
        end

        step += 1

        if trainer.show_progress
            POMDPToolbox.ProgressMeter.next!(prog)
        end
    end

    if sim.show_progress
        POMDPToolbox.ProgressMeter.finish!(prog)
    end
end
