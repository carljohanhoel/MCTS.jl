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
    n_network_updates_per_sample::Int
    save_freq::Int
    eval_freq::Int
    eval_eps::Int
    fix_eval_eps::Bool
    remove_end_samples::Int
    stash_factor::Float64
    save_evaluation_history::Bool
    show_progress::Bool
    log_dir::String
end

function Trainer(;rng=MersenneTwister(rand(UInt32)),
                  rng_eval=MersenneTwister(rand(UInt32)),
                  training_steps::Int=1,
                  n_network_updates_per_sample::Int=1,
                  save_freq::Int=Inf,
                  eval_freq::Int=Inf,
                  eval_eps::Int=1,
                  fix_eval_eps::Bool=true,
                  remove_end_samples::Int=0,
                  stash_factor::Float64=3.0,
                  save_evaluation_history::Bool=false,
                  show_progress=false,
                  log_dir::String="./"
                 )
    return Trainer(rng, rng_eval, training_steps, n_network_updates_per_sample, save_freq, eval_freq, eval_eps, fix_eval_eps, remove_end_samples, stash_factor, save_evaluation_history, show_progress, log_dir)
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
    # if trainer.show_progress
    #     prog = POMDPToolbox.Progress(training_steps, "Training..." )
    # end
    process_id = myid()
    out = @spawnat 1 println("Training started on process "*string(process_id))
    fetch(out)

    nn_estimator = policy isa AZPlanner ? policy.solver.estimate_value : policy.planner.solver.estimate_value

    n_saves = 0
    n_evals = -1 #Forces an evaluation run before the network is trained.
    step = 1

    while step <= training_steps
        #Generate initial state
        s_initial = initial_state(p,trainer.rng)

        #Simulate one episode
        if p isa POMDP
            initial_state_dist = state_dist(p, s_initial)
            hist = POMDPs.simulate(sim, p, policy, belief_updater, initial_state_dist, s_initial)
        else
            hist = POMDPs.simulate(sim, p, policy, s_initial)
        end

        #Extract training samples
        n_a = length(actions(p))
        all_actions = actions(policy.mdp)
        n_new_samples = length(hist.state_hist)
        new_states = deepcopy(hist.state_hist)
        new_z_values = Vector{Float64}(length(new_states))
        new_q_values = Vector{Float64}(length(new_states))
        new_distributions = Array{Float64}(length(new_states),n_a)
        ##
        end_state = new_states[end]
        end_value = isterminal(p,end_state) ? 0 : estimate_value(nn_estimator, end_state, p)[1]
        new_z_values[end] = end_value
        value = end_value
        new_q_values[end] = end_value
        ## for (i,state) in enumerate(new_states[end-1:-1:1])
        for i in 1:length(new_states)-1
           value = hist.reward_hist[end+1-i] + p.discount*value
           new_z_values[end-i] = value
           new_distributions[end-i,:] = hist.ainfo_hist[end+1-i][:action_distribution]
           a_idx = findfirst(all_actions,hist.action_hist[end+1-i])
           new_q_values[end-i] = hist.ainfo_hist[end+1-i][:q_values][a_idx]
        end

        if isterminal(p,end_state) #If terminal state, keep value 0 and add dummy distribution, otherwise remove last sample (the simulation gives no information about it)
           new_distributions[end,:] = ones(1,n_a)/n_a
        else
           pop!(new_states)
           pop!(new_z_values)
           pop!(new_q_values)
           new_distributions = new_distributions[1:end-1,:]
           n_new_samples-=1

           if trainer.remove_end_samples > 0    #Remove some more samples, to reduce effect of badly estimated final value
               for i in 1:trainer.remove_end_samples
                   pop!(new_states)
                   pop!(new_z_values)
                   pop!(new_q_values)
                   new_distributions = new_distributions[1:end-1,:]
                   n_new_samples-=1
               end
           end
        end



        new_values = new_z_values
        # new_values = (new_z_values+new_q_values)/2

        #Update network
        add_samples_to_memory(nn_estimator, new_states, new_distributions, new_values, p)
        update_network(nn_estimator,trainer.n_network_updates_per_sample*n_new_samples) #Runs update n times

        step += n_new_samples


        if div(step,trainer.save_freq) > n_saves
        # if step%trainer.save_freq == 0
            filename = trainer.log_dir*"/"*string(step)
            save_network(nn_estimator, filename)
            n_saves+=1
            out = @spawnat 1 println("Network saved on process "*string(process_id)*" at "*Dates.format(Dates.now(), "yymmdd_HHMMSS"))
            fetch(out)
        end

        if div(step,trainer.eval_freq) > n_evals
            out = @spawnat 1 println("Evaluation started on process "*string(process_id)*" after "*string(step)*" steps")
            fetch(out)
            eval_eps = 1
            if policy isa AZPlanner
                policy.training_phase=false
            else
                policy.planner.training_phase=false
            end
            if trainer.fix_eval_eps   #if fix_eval, keep rng constant to always evaluate the same set of episodes
                rng = copy(trainer.rng_eval)
                rng_sim = copy(sim.rng)    #Save sim.rng for resetting it after evaluatio is done
                sim.rng = MersenneTwister(Int(rng.seed[1])+1)
                rng_policy = copy(policy.rng)
                rng_solver = copy(policy.solver.rng)
                policy.rng = MersenneTwister(Int(rng.seed[1])+2)
                policy.solver.rng = MersenneTwister(Int(rng.seed[1])+3)
            end
            # rng = trainer.fix_eval_eps ? copy(trainer.rng_eval) : trainer.rng_eval
            # if n_evals == 0
                open(trainer.log_dir*"/"*"rngs.txt","a") do f
                    writedlm(f, [[process_id, Int(rng.seed[1]), Int(sim.rng.seed[1]), Int(policy.rng.seed[1]), Int(policy.solver.rng.seed[1])]], " ")
                end
            # end
            episode_reward = []
            episode_discounted_reward = []
            log = []
            while eval_eps <= trainer.eval_eps
                s_initial = initial_eval_state(p, rng)
                if p isa POMDP
                    initial_state_dist = state_dist(p, s_initial)
                    hist = POMDPs.simulate(sim, p, policy, belief_updater, initial_state_dist, s_initial)
                else
                    hist = POMDPs.simulate(sim, p, policy, s_initial)
                end
                push!(episode_reward, sum(hist.reward_hist))
                push!(episode_discounted_reward, sum(hist.reward_hist)*p.discount^(length(hist.reward_hist)-1))
                push!(log, create_eval_log(p,hist, process_id, step))
                eval_eps+=1
            end
            open(trainer.log_dir*"/"*"evalResults2.txt","a") do f
                for (i,row) in enumerate(log)
                    writedlm(f, row, " ")
                end
            end
            open(trainer.log_dir*"/"*"evalResults.txt","a") do f   #This is deprecated and should be removed in the future
                writedlm(f, [[process_id, step, mean(episode_reward), mean(episode_discounted_reward), episode_reward, episode_discounted_reward]], " ")
            end
            if policy isa AZPlanner
                policy.training_phase=true
            else
                policy.planner.training_phase=true
            end
            if trainer.fix_eval_eps   #Reset simulator rng if temproarily fixed during evaluation
                sim.rng = rng_sim
                policy.rng = rng_policy
                policy.solver.rng = rng_solver

            end
            if trainer.save_evaluation_history && process_id <= 7   #Hard coded just save history for 5 processes (first worker is process 3)
                JLD.save(trainer.log_dir*"/"*"eval_hist_process_"*string(process_id)*"_step_"*string(step)*".jld", "hist", hist)
            end
            n_evals+=1
            out = @spawnat 1 println("Evaluation finished on process "*string(process_id))
            fetch(out)
        end

        # if trainer.show_progress
        #     POMDPToolbox.ProgressMeter.update!(prog, step)
        # end
        # @spawnat 1 println("Worker "*string(process_id)*", step "*string(step))
    end

    # if trainer.show_progress
    #     POMDPToolbox.ProgressMeter.update!(prog, training_steps)
    # end

    # if trainer.show_progress
    #     POMDPToolbox.ProgressMeter.finish!(prog)
    # end
    out = @spawnat 1 println("Worker "*string(process_id)*" finished")
    fetch(out)
end



##############

#Parallelization of training
function train_parallel(trainer::Trainer,
                        sim::HistoryRecorder,
                        p::Union{POMDP,MDP},
                        policy::Policy,
                        belief_updater::Updater=VoidUpdater()
                        )

    n_procs = nprocs()
    assert(n_procs>2) #First process is main and second is queue. 3 and higher are training processes.

    trainer_vec = []
    sim_vec = []
    problem_vec = []
    policy_vec = []
    belief_vec = []
    for i in 1:n_procs-2
        push!(trainer_vec,deepcopy(trainer))
        push!(sim_vec,deepcopy(sim))
        push!(problem_vec,deepcopy(p))
        push!(policy_vec,deepcopy(policy))
        push!(belief_vec,deepcopy(belief_updater))
    end
    for i in 2:n_procs-2
        #Set different RNGs
        # rng_seed = 342*i+630 #rand(trainer.rng,1:1000000) #Not used
        rng_evaluator=MersenneTwister(Int(trainer.rng_eval.seed[1])+100*(i-1))
        rng_solver=MersenneTwister(Int(policy.rng.seed[1])+100*(i-1))
        rng_history=MersenneTwister(Int(sim.rng.seed[1])+100*(i-1))
        rng_trainer=MersenneTwister(Int(trainer.rng.seed[1])+100*(i-1))

        policy_vec[i].solver.rng = rng_solver
        policy_vec[i].rng = rng_solver
        sim_vec[i].rng = rng_history
        trainer_vec[i].rng = rng_trainer
        trainer_vec[i].rng_eval = rng_evaluator

        if isdefined(belief_vec[i],:rng)
            rng_belief=MersenneTwister(Int(belief_updater.rng.seed[1])+100*(i-1))
            belief_vec[i].rng = rng_belief
        end

        #Remove saving on all except first process (number 3)
        trainer_vec[i].save_freq = typemax(Int)
    end

    # stash_size = min(Sys.CPU_CORES,round(Int,(n_procs-2)/2))  #n_procs-2 = #workers. Divide by 2 to always have some active workers
    # stash_size = round(Int,(n_procs-2)/2)  #n_procs-2 = #workers. Divide by 2 to always have some active workers
    # stash_size = max(round(Int,(n_procs-2)/6),1)  #n_procs-2 = #workers. Ratio 1:6 as in nochi code
    # stash_size = max(round(Int,(n_procs-2)/3),1)  #n_procs-2 = #workers. Ratio 1:3
    # stash_size = max(round(Int,(n_procs-2)/1.5),1)  #n_procs-2 = #workers. Ratio 1:1.5
    stash_size = max(round(Int,(n_procs-2)/trainer.stash_factor),1)  #n_procs-2 = #workers. Ratio 1:1.5
    set_stash_size(policy.solved_estimate, stash_size)

    processes = []
    for i in 1:n_procs-2
        out = @spawnat i+2 train(trainer_vec[i], sim_vec[i], problem_vec[i], policy_vec[i], belief_vec[i])
        sleep(3)
        push!(processes, out)
    end

    return processes
end



function create_eval_log(p::Union{MDP,POMDP},hist::Union{POMDPToolbox.MDPHistory,POMDPToolbox.POMDPHistory}, process_id::Int, step::Int)
    log = []
    push!(log,process_id)
    push!(log,step)
    push!(log, sum(hist.reward_hist))
    push!(log, sum(hist.reward_hist)*p.discount^(length(hist.reward_hist)-1)) #This is only valid for GridWorld, where only reward is the last one
    return log'
end
