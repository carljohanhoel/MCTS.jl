using Revise #21 MB ###### Comment for real runs to free RAM

# debug = true
debug = false

parallel_version = true   #Test code in parallel mode
# parallel_version = false

simple_run = true
# simple_run = false

if parallel_version
   # n_workers = 50
   n_workers = 8
   # n_workers = 4
   # n_workers = 1
   if simple_run
      n_workers = 4
   end
   addprocs(n_workers+1)
   @everywhere using MCTS
else
   n_workers = 1
   using MCTS
end

using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test
using D3Trees

##

n_iter = 2000
depth = 15
c_puct = 1/20*5   # (1/20 because actual normalized max for gridworld is 1/20)
tau = 1.1
stash_factor = 1.5 #stash_size = n_workers/stash_factor
noise_dirichlet = 1.0

if simple_run
   n_iter = 20

   replay_memory_max_size = 200
   training_start = 100
   training_steps = Int(ceil(1000/n_workers))*1000 ###
   n_network_updates_per_sample = 1
   save_freq = Int(ceil(100/n_workers))*10 ###
   eval_freq = Int(ceil(100/n_workers)) ###
   eval_eps = Int(ceil(8/n_workers))
   save_evaluation_history = true
else
   # replay_memory_max_size = 100000
   # training_start = 5000
   # training_steps = 100000
   # n_network_updates_per_episode = 10
   # save_freq = 5000
   # eval_freq = 5000
   # eval_eps = 100
   # replay_memory_max_size = 100000
   # training_start = 10000
   # training_steps = 1000000
   # n_network_updates_per_episode = 10
   # save_freq = 10000
   # eval_freq = 10000
   # eval_eps = 100
   replay_memory_max_size = 10000
   training_start = 5000 #This is used in py, so includes all workers
   training_steps = Int(ceil(1000000/n_workers))
   n_network_updates_per_sample = 1
   save_freq = Int(ceil(5000/n_workers))
   eval_freq = Int(ceil(5000/n_workers))
   eval_eps = Int(ceil(100/n_workers))
   save_evaluation_history = true
end


sim_max_steps = 25

rng_seed = 14
rng_estimator=MersenneTwister(rng_seed+1)
rng_evaluator=MersenneTwister(rng_seed+2)
rng_solver=MersenneTwister(rng_seed+3)
rng_history=MersenneTwister(rng_seed+4)
rng_trainer=MersenneTwister(rng_seed+5)

mdp = GridWorld(5,5,
                penalty=0.,
                rs=[GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                tp = 0.8,
                terminals = [GridWorldState(3,3),GridWorldState(5,3),GridWorldState(5,5),GridWorldState(1,1)],
                )
s_initial = GridWorldState(1,1)

n_s = length(MCTS.convert_state(s_initial, mdp))
n_a = n_actions(mdp)
v_max = 1*1.05 #Reward should be in range [-1,1]. Factor 1.05 helps with avoiding extreme values needed to create 1.000 in sigmoid
v_min = -v_max
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_name = length(ARGS)>0 ? ARGS[1] : ""
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS_")*log_name
if parallel_version
   #Start queue on process 2
   @spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, debug),cmd_queue,res_queue)
   estimator = NNEstimatorParallel(v_min, v_max)
   sleep(3) #Wait for queue to be set up before continuing
   clear_queue()
else
   estimator = NNEstimator(rng_estimator, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

#Load trained network to continue training
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180629_024700_2000_mcts_searches_100_updates_2_puct_8_queues/5001")

solver = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=c_puct,
               k_state=3.,
               tree_in_info=false,
               alpha_state=0.2,
               tau=tau,
               enable_action_pw=false,
               check_repeat_state=false,
               rng=rng_solver,
               estimate_value=estimator,
               init_P=estimator,
               noise_dirichlet = noise_dirichlet,
               noise_eps = 0.25
               )
policy = solve(solver, mdp)

sim = HistoryRecorder(rng=rng_history, max_steps=sim_max_steps, show_progress=false)

## Save files to log to be able to check parameters
if !ispath(log_path)
   mkdir(log_path)
end
mkdir(log_path*"/code")
cp(pwd()*"/test/",log_path*"/code/test/")
cp(pwd()*"/src/",log_path*"/code/src/")
# cp(pwd()*"/test/one_training_run.jl",log_path*"/one_training_run.jl")
cp(estimator_path*".py",log_path*"/neural_net.py")

##
trainer = Trainer(rng=rng_trainer, rng_eval=rng_evaluator, training_steps=training_steps, n_network_updates_per_sample=n_network_updates_per_sample, save_freq=save_freq, eval_freq=eval_freq, eval_eps=eval_eps, fix_eval_eps=true, stash_factor=stash_factor, save_evaluation_history=save_evaluation_history, show_progress=true, log_dir=log_path)
if parallel_version
   processes = train_parallel(trainer, sim, mdp, policy)

   for proc in processes #This make Julia wait with terminating until all processes are done. However, all processes will never finish when stash size is bigger than 1. Fine for now...
      fetch(proc)
   end
else
   train(trainer, sim, mdp, policy)
end
