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
    rng_eval::AbstractRNG
    training_steps::Int
    n_network_updates_per_episode::Int
    save_freq::Int
    eval_freq::Int
    eval_eps::Int
    fix_eval_eps::Bool
    show_progress::Bool
    log_dir::String
end

function Trainer(;rng=MersenneTwister(rand(UInt32)),
                  rng_eval=MersenneTwister(rand(UInt32)),
                  training_steps::Int=1,
                  n_network_updates_per_episode::Int=1,
                  save_freq::Int=Inf,
                  eval_freq::Int=Inf,
                  eval_eps::Int=1,
                  fix_eval_eps::Bool=true,
                  show_progress=false,
                  log_dir::String="./"
                 )
    return Trainer(rng, rng_eval, training_steps, n_network_updates_per_episode, save_freq, eval_freq, eval_eps, fix_eval_eps, show_progress, log_dir)
end

struct VoidUpdater <: Updater
end


function train(trainer::Trainer,
                    sim::HistoryRecorder,
                    p::Union{POMDP,MDP},
                    policy::Policy,
                    belief_updater::Updater=VoidUpdater()
                   )
    training_steps = trainer.training_steps
    if trainer.show_progress
        prog = POMDPToolbox.Progress(training_steps, "Training..." )
    end

    nn_estimator = policy isa AZPlanner ? policy.solver.estimate_value : policy.planner.solver.estimate_value

    n_saves = 0
    n_evals = 0
    step = 1
    while step <= training_steps
        #Generate initial state
        s_initial = initial_state(p,trainer.rng)

        #Simulate one episode
        if p isa POMDP
            initial_state_dist = state_dist(s_initial)
            hist = POMDPs.simulate(sim, p, policy, belief_updater, initial_state_dist, s_initial)
        else
            hist = POMDPs.simulate(sim, p, policy, s_initial)
        end

        #Extract training samples
        n_a = length(actions(p))
        n_new_samples = length(hist.state_hist)
        new_states = deepcopy(hist.state_hist)
        new_values = Vector{Float64}(length(new_states))
        new_distributions = Array{Float64}(length(new_states),n_a)

        end_state = new_states[end]
        end_value = isterminal(p,end_state) ? 0 : estimate_value(nn_estimator, end_state, p)
        new_values[end] = end_value
        value = end_value
        for (i,state) in enumerate(new_states[end-1:-1:1])
           value = hist.reward_hist[end+1-i] + p.discount*value
           new_values[end-i] = value
           new_distributions[i,:] = hist.ainfo_hist[i][:action_distribution]
        end

        if isterminal(p,end_state) #If terminal state, keep value 0 and add dummy distribution, otherwise remove last sample (the simulation gives no information about it)
           new_distributions[end,:] = ones(1,n_a)/n_a
        else
           pop!(new_states)
           pop!(new_values)
           new_distributions = new_distributions[1:end-1,:]
           n_new_samples-=1
        end

        #Update network
        add_samples_to_memory(nn_estimator, new_states, new_distributions, new_values, p)
        for i in 1:trainer.n_network_updates_per_episode
            update_network(nn_estimator)
        end

        step += n_new_samples


        if div(step,trainer.save_freq) > n_saves
        # if step%trainer.save_freq == 0
            filename = trainer.log_dir*"/"*string(step)
            save_network(nn_estimator, filename)
            n_saves+=1
        end

        if div(step,trainer.eval_freq) > n_evals
            eval_eps = 1
            if policy isa AZPlanner
                policy.training_phase=false
            else
                policy.planner.training_phase=false
            end
            rng = trainer.fix_eval_eps ? copy(trainer.rng_eval) : trainer.rng_eval   #if fix_eval, keep rng constant to always evaluate the same set of episodes
            episode_reward = []
            while eval_eps <= trainer.eval_eps
                s_initial = initial_eval_state(p, rng)
                if p isa POMDP
                    initial_state_dist = state_dist(s_initial)
                    hist = POMDPs.simulate(sim, p, policy, belief_updater, initial_state_dist, s_initial)
                else
                    hist = POMDPs.simulate(sim, p, policy, s_initial)
                end
                push!(episode_reward, sum(hist.reward_hist))
                eval_eps+=1
            end
            open(trainer.log_dir*"/"*"evalResults.txt","a") do f
                writedlm(f, [[step, mean(episode_reward), episode_reward]], ", ")
            end
            if policy isa AZPlanner
                policy.training_phase=true
            else
                policy.planner.training_phase=true
            end
            n_evals+=1
        end

        if trainer.show_progress
            POMDPToolbox.ProgressMeter.update!(prog, step)
        end
    end

    if trainer.show_progress
        POMDPToolbox.ProgressMeter.update!(prog, training_steps)
    end

    if trainer.show_progress
        POMDPToolbox.ProgressMeter.finish!(prog)
    end
end
