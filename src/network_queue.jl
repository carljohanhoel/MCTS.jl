const cmd_queue = RemoteChannel(()->Channel{Tuple}(128))
const res_queue = fill(RemoteChannel(()->Channel{Array{Float64}}(4)),128) #128 max number of processes

mutable struct NetworkQueue
   py_class::PyCall.PyObject
   stash::Array{Tuple}
   trigger::Int
   update_counter::Int
   debug::Bool
end

function NetworkQueue(estimator_path::String, log_path::String, n_states::Int, n_actions::Int, replay_memory_max_size::Int, training_start::Int, debug::Bool=false) #
    py_class = initialize_queue(estimator_path, log_path, n_states, n_actions, replay_memory_max_size, training_start)
    return NetworkQueue(py_class, Array{Tuple}(0), 1, 0, debug)
end

function initialize_queue(estimator_path::String, log_path::String, n_states::Int, n_actions::Int, replay_memory_max_size::Int, training_start::Int)
    unshift!(PyVector(pyimport("sys")["path"]), dirname(estimator_path))
    eval(parse(string("@pyimport ", basename(estimator_path), " as python_module")))
    py_class = python_module.NeuralNetwork(n_states, n_actions, replay_memory_max_size, training_start, log_path)
    return py_class
end

#This function is started in some process and then continuously running there. Requests for e.g. forward passes or update the network are placed in the cmd_queue and results are reported back in the res_queue corresponding to the requesting process.
function run_queue(q::NetworkQueue, cmd_queue, res_queue)
   remotecall_fetch(println,1,"Starting network queue")

   function process_stash(q::NetworkQueue)
      if length(q.stash) == 0
         return
      end
      states = vcat([o[2] for o in q.stash]...)
      dist, val = q.py_class[:forward_pass](states)
      for (i,obj) in enumerate(q.stash)
         kind, state, proc = obj
         if kind == 0
            put!(res_queue[proc],reshape(dist[i,:], (size(dist[i,:])...,1))') #Reshape creates 1×n Array{Float32,2}
         else
            put!(res_queue[proc],[val[i]])
         end
      end
      q.stash = Array{Tuple}(0)
   end
   function add(q::NetworkQueue, kind, state, proc)
      push!(q.stash,(kind,state,proc))
      if length(q.stash) >= q.trigger
         process_stash(q)
      end
   end

   while true
      if q.debug remotecall_fetch(println,1,"in loop") end
      cmd, proc, state, states, dists, vals, trigger, name = take!(cmd_queue)
      if q.debug remotecall_fetch(println,1,string(cmd)*" "*string(state)*" "*string(proc)) end
      if cmd == "stash_size"
         process_stash(q)
         if q.debug remotecall_fetch(println,1,trigger) end
         q.trigger = trigger
      elseif cmd == "add_samples_to_memory"
         process_stash(q)
         q.py_class[:add_samples_to_memory](states, dists, vals)
         if q.debug remotecall_fetch(println,1,"memory updated") end
         put!(res_queue[proc],[12])
      elseif cmd == "update_network"
         process_stash(q)
         q.update_counter += 1
         q.py_class[:update_network]()
         if q.debug remotecall_fetch(println,1,"network updated") end
         put!(res_queue[proc],[13])
      elseif cmd == "predict_distribution"
         add(q,0,state,proc)
         if q.debug remotecall_fetch(println,1,"pred dist added") end
      elseif cmd == "predict_value"
         add(q,1,state,proc)
         if q.debug remotecall_fetch(println,1,"pred val added") end
      elseif cmd == "save"
         process_stash(q)
         q.py_class[:save_network](name)
         if q.debug remotecall_fetch(println,1,"save net as "*name) end
      elseif cmd == "load"
         process_stash(q)
         q.py_class[:load_network](name)
         remotecall_fetch(println,1,"load net "*name)
      else
         remotecall(println,1,"Error in cmd queue")
      end
   end
end


struct NNEstimatorParallel
   v_min::Float64
   v_max::Float64
end

estimate_value(estimator::NNEstimatorParallel, p::Union{POMDP,MDP}, state, depth::Int) = estimate_value(estimator, state, p)

function estimate_value(estimator::NNEstimatorParallel, state, p::Union{POMDP,MDP})
    converted_state = convert_state(state, p)
    put!(cmd_queue,("predict_value",myid(),converted_state,nothing,nothing,nothing,nothing,nothing))
    value = take!(res_queue[myid()])
    value = value*(estimator.v_max-estimator.v_min)+estimator.v_min #Scale [0,1]->[v_min,v_max]
    return value[1] #Convert to scalar
end

function estimate_distribution(estimator::NNEstimatorParallel, state, allowed_actions, p::Union{POMDP,MDP})
    converted_state = convert_state(state, p)
    put!(cmd_queue,("predict_distribution",myid(),converted_state,nothing,nothing,nothing,nothing,nothing))
    dist = take!(res_queue[myid()])
    dist = dist.*allowed_actions
    sum_dist = sum(dist,2)
    if any(sum_dist.==0)   #Before the network is trained, the only allowed actions could get prob 0. In that case, set equal prior prob.
         println("error, sum allowed dist = 0")
         println(state)
         println(dist)
         println(allowed_actions)
         add_dist = ((dist*0+1) .* (sum_dist .== 0.)).*allowed_actions
         dist += add_dist
         sum_dist += sum(add_dist,2)
    end
    # dist = [dist[i,:]/sum_dist[i] for i in range(0,len(sum_dist))]
    dist = dist./sum_dist

    return dist
end

function add_samples_to_memory(estimator::NNEstimatorParallel, states, dists, vals, p)
    converted_states = convert_state(states, p)
    vals = (vals-estimator.v_min)/(estimator.v_max-estimator.v_min)
    put!(cmd_queue,("add_samples_to_memory",myid(),nothing,converted_states,dists,vals,nothing,nothing))
    out = take!(res_queue[myid()])
end

function update_network(estimator::NNEstimatorParallel)
    put!(cmd_queue,("update_network",myid(),nothing,nothing,nothing,nothing,nothing,nothing))
    out = take!(res_queue[myid()])
end

function save_network(estimator::NNEstimatorParallel, name::String)
    put!(cmd_queue,("save",myid(),nothing,nothing,nothing,nothing,nothing,name))
end

function load_network(estimator::NNEstimatorParallel, name::String)
    put!(cmd_queue,("load",myid(),nothing,nothing,nothing,nothing,nothing,name))
end


# #Needs to be defined for each problem to fit the input of the nerual network
# function convert_state(state::Type, p::Union{POMDP,MDP})
#     converted_state = state
#     return converted_state
# end
#
# function state_dist() #Dummy function, to be defined for each problem
# end
#
# #Simple example for GridWorld, here for tests. Remove later.
# using POMDPModels
# function convert_state(state::Vector{GridWorldState}, mdp::GridWorld)
#     n = length(state)
#     converted_state = Array{Float64}(n,3)
#     for i in 1:n
#         converted_state[i,:] = convert_state(state[i], mdp)
#     end
#     return converted_state
# end
# function convert_state(state::GridWorldState, mdp::GridWorld)
#     converted_state = Array{Float64}(1,3)
#     converted_state[1] = state.x
#     converted_state[2] = state.y
#     converted_state[3] = state.done ? 1 : 0
#     return converted_state
# end
