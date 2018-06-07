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
    eval_freq::Int
    eval_eps::Int
    show_progress::Bool
    log_dir::String
    process_id::Int
end

function Trainer(;rng=MersenneTwister(rand(UInt32)),
                  training_steps::Int=1,
                  save_freq::Int=Inf,
                  eval_freq::Int=Inf,
                  eval_eps::Int=1,
                  show_progress=false,
                  log_dir::String="./",
                  process_id::Int=1
                 )
    return Trainer(rng, training_steps, save_freq, eval_freq, eval_eps, show_progress, log_dir, process_id)
end


function train{S,A}(trainer::Trainer,
                    sim::HistoryRecorder,
                    mdp::MDP{S,A}, policy::Policy
                   )
    @printf("Start training with process %d \n", trainer.process_id)
    policy.solver.estimate_value.py_class[:net][:ri] = trainer.process_id

    training_steps = trainer.training_steps
    if trainer.show_progress
        prog = POMDPToolbox.Progress(training_steps, "Training..." )
    end

    n_saves = 0
    n_evals = 0
    step = 1
    while step <= training_steps
        #Generate initial state
        s_initial = initial_state(mdp,trainer.rng)

        #Simulate one episode
        hist = POMDPs.simulate(sim, mdp, policy, s_initial)

        #Extract training samples
        n_a = length(actions(mdp))
        n_new_samples = length(hist.state_hist)
        new_states = deepcopy(hist.state_hist)
        new_values = Vector{Float64}(length(new_states))
        new_distributions = Array{Float64}(length(new_states),n_a)

        end_state = new_states[end]
        end_value = isterminal(mdp,end_state) ? 0 : estimate_value(policy.solver.estimate_value, end_state)
        new_values[end] = end_value
        value = end_value
        for (i,state) in enumerate(new_states[end-1:-1:1])
           value = hist.reward_hist[end+1-i] + mdp.discount_factor*value
           new_values[end-i] = value
           new_distributions[i,:] = hist.ainfo_hist[i][:action_distribution]
        end

        if isterminal(mdp,end_state) #If terminal state, keep value 0 and add dummy distribution, otherwise remove last sample (the simulation gives no information about it)
           new_distributions[end,:] = ones(1,n_a)/n_a
        else
           pop!(new_states)
           pop!(new_values)
           new_distributions = new_distributions[1:end-1,:]
           n_new_samples-=1
        end

        #Update network   - ZZZZZZZZZZZZZZZ Add option to run several times after every episode
        # println("Update network")
        for i in 1:10
            update_network(policy.solver.estimate_value, new_states, new_distributions, new_values)
        end

        step += n_new_samples


        if div(step,trainer.save_freq) > n_saves
        # if step%trainer.save_freq == 0
            filename = trainer.log_dir*"/"*string(step)
            # println("Saving")
            # println(filename)
            save_network(policy.solver.estimate_value, filename)
            n_saves+=1
        end

        if div(step,trainer.eval_freq) > n_evals
            eval_eps = 1
            policy.training_phase=false
            s_initial = GridWorldState(5,1)   #ZZZZZZZZZZZZZ Fix, generalize
            episode_reward = []
            while eval_eps <= trainer.eval_eps
                hist = POMDPs.simulate(sim, mdp, policy, s_initial)
                push!(episode_reward, sum(hist.reward_hist))
                eval_eps+=1
            end
            open(trainer.log_dir*"/"*"evalResults.txt","a") do f
                writedlm(f, [[step, mean(episode_reward), episode_reward]], ", ")
            end
            policy.training_phase=true
            n_evals+=1
        end

        if trainer.show_progress
            POMDPToolbox.ProgressMeter.update!(prog, step)
        end
    end

    if sim.show_progress
        POMDPToolbox.ProgressMeter.finish!(prog)
    end
end




#Parallelization of training
function train_parallel(trainer::Trainer,
                        sim::HistoryRecorder,
                        mdp::MDP,
                        policy::Policy,
                        n_processes::Int,
                        policy2::Policy,
                        mdp2::MDP
                        )
    #=
    frame_lines = pmap(progress, queue) do sim
        result = simulate(sim)
        return process(sim, result)
    end
    =#

    np = nprocs()
    addprocs(max(n_processes-np,0))
    if n_processes == 1
        warn("""
             run_parallel(...) was started with only 1 process.
             """)
    end


    #ZZZ Create copies of trainer
    trainer_vec = []
    sim_vec = []
    mdp_vec = []
    policy_vec = []
    # for i=1:n_processes
    #     push!(trainer_vec,deepcopy(trainer))
    #     push!(sim_vec,deepcopy(sim))
    #     push!(mdp_vec,deepcopy(mdp))
    #     push!(policy_vec,policy)
    # end
    # for i=2:n_processes
    #     #Set same NN estimator
    #     # policy_vec[i].solver.estimate_value = policy_vec[1].solver.estimate_value
    #
    #     #Change RNGs
    #     rng = MersenneTwister(i+rand(trainer.rng,1:1000000))
    #     policy_vec[i].solver.rng = rng
    #     sim_vec[i].rng = rng
    #     trainer_vec[i].rng = rng
    #
    #     #Set process id
    #     trainer_vec[i].process_id = i
    #
    #     #Remove saving
    #     trainer_vec[i].save_freq = typemax(Int)
    # end

    push!(trainer_vec, trainer)
    push!(trainer_vec, deepcopy(trainer))
    push!(sim_vec, sim)
    push!(sim_vec, deepcopy(sim))
    push!(mdp_vec, mdp)
    push!(mdp_vec, mdp2)
    push!(policy_vec, policy)
    push!(policy_vec, policy2)


    # policy.solver.estimate_value.py_class[:net][:stash_size](n_processes) #ZZZZZZZZZZZ This should be included

    i = 1
    prog = 0
    # based on the simple implementation of pmap here: https://docs.julialang.org/en/latest/manual/parallel-computing
    nextidx() = (idx=i; i+=1; idx)
    prog_lock = ReentrantLock()
    @sync begin
        for p in 1:n_processes
            if np == 1 || p != myid()
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n_processes
                            break
                        end
                        #ZZZ Set up process id to python code
                        remotecall_fetch(train, p, trainer_vec[idx], sim_vec[idx], mdp_vec[idx], policy_vec[idx])

                    end
                end
            end
        end
    end
    # print(frame_lines)
end
