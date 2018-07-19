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
    stash_factor::Float64
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
                  stash_factor::Float64=3.0,
                  show_progress=false,
                  log_dir::String="./"
                 )
    return Trainer(rng, rng_eval, training_steps, n_network_updates_per_sample, save_freq, eval_freq, eval_eps, fix_eval_eps, stash_factor, show_progress, log_dir)
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
    n_evals = 0
    step = 1

    proc = myid()-2 ### dbg
    vv_tmp = (proc-10)/20 ### dbg
    # dd_tmp = rand(1,4) ### dbg
    # dd_tmp = [0.25 0.2 0.35 0.2] ### dbg
    dd_tmp = [proc 25-proc proc/2 (25-proc)/2] ### dbg
    dd_tmp = dd_tmp/sum(dd_tmp) ### dbg
    # out = @spawnat 1 println(vv_tmp) ### dbg
    # fetch(out) ### dbg
    # out = @spawnat 1 println(dd_tmp/sum(dd_tmp)) ### dbg
    # fetch(out) ### dbg
    while step <= training_steps
        if false #This part is just for debug. Remove later
            n_new_samples = rand(3:12) ### dbg
            new_states = Array{GridWorldState}(n_new_samples) ### dbg
            for i in 1:n_new_samples ### dbg
                new_states[i] = GridWorldState(div(proc-1,5)+1,(proc-1)%5+1) ### dbg
            end ### dbg
            new_z_values = ones(n_new_samples,1)*vv_tmp+0.5*(2*rand(n_new_samples,1)-1) ### dbg
            new_distributions = ones(n_new_samples)*dd_tmp+0.5*rand(n_new_samples,4) ### dbg
            new_distributions = new_distributions./sum(new_distributions,2) ### dbg
        else
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
            end
            # new_values = new_z_values
            new_values = (new_z_values+new_q_values)/2
        end

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
            rng = trainer.fix_eval_eps ? copy(trainer.rng_eval) : trainer.rng_eval   #if fix_eval, keep rng constant to always evaluate the same set of episodes
            episode_reward = []
            episode_discounted_reward = []
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
                eval_eps+=1
            end
            open(trainer.log_dir*"/"*"evalResults.txt","a") do f
                writedlm(f, [[process_id, step, mean(episode_reward), mean(episode_discounted_reward), episode_reward, episode_discounted_reward]], ", ")
            end
            if policy isa AZPlanner
                policy.training_phase=true
            else
                policy.planner.training_phase=true
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
        rng_seed = i+rand(trainer.rng,1:1000000)
        rng_estimator=MersenneTwister(rng_seed+1)
        rng_evaluator=MersenneTwister(rng_seed+2)
        rng_solver=MersenneTwister(rng_seed+3)
        rng_history=MersenneTwister(rng_seed+4)
        rng_trainer=MersenneTwister(rng_seed+5)
        rng_belief=MersenneTwister(rng_seed+6)

        policy_vec[i].solver.rng = rng_solver
        policy_vec[i].rng = rng_solver
        sim_vec[i].rng = rng_history
        trainer_vec[i].rng = rng_evaluator
        trainer_vec[i].rng_eval = rng_trainer

        if isdefined(belief_vec[i],:rng)
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
