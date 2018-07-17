struct QueueCommand
### Fill out
end

struct QueueResult
   process_id::Int
   distribution::Array{Float64}
   value::Float64
end

const cmd_queue = RemoteChannel(()->Channel{Tuple}(128))
# const res_queue = fill(RemoteChannel(()->Channel{Array{Float64}}(1)),128) #128 max number of processes. (1 is number of elements each result queue can hold. Since it is sequential, 1 should be enough.)
# const res_queue = fill(RemoteChannel(()->Channel{QueueResult}(1)),128) #128 max number of processes. (1 is number of elements each result queue can hold. Since it is sequential, 1 should be enough.)
tmp_queue = []
for i in 1:128
   push!(tmp_queue,RemoteChannel(()->Channel{QueueResult}(1)))
end
const res_queue = tmp_queue

function clear_queue()   #Something makes queue sometimes not empty at start. This function clears it.
   for i in 1:128
      if isready(res_queue[i])
         out = take!(res_queue[i])
         println(out)
      end
   end
end

mutable struct NetworkQueue
   py_class::PyCall.PyObject
   stash::Array{Tuple}
   trigger::Int
   update_counter::Int
   debug::Bool
end

function NetworkQueue(estimator_path::String, log_path::String, n_states::Int, n_actions::Int, replay_memory_max_size::Int, training_start::Int, debug::Bool=false) #
    py_class = initialize_queue(estimator_path, log_path, n_states, n_actions, replay_memory_max_size, training_start, debug)
    return NetworkQueue(py_class, Array{Tuple}(0), 1, 0, debug)
end

function initialize_queue(estimator_path::String, log_path::String, n_states::Int, n_actions::Int, replay_memory_max_size::Int, training_start::Int, debug::Bool=false)
    unshift!(PyVector(pyimport("sys")["path"]), dirname(estimator_path))
    eval(parse(string("@pyimport ", basename(estimator_path), " as python_module")))
    py_class = python_module.NeuralNetwork(n_states, n_actions, replay_memory_max_size, training_start, log_path, debug=debug)
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
      if q.debug remotecall_fetch(println,1,"stash size: "*string(length(q.stash))) end
      # remotecall_fetch(println,1,states)
      dist, val = q.py_class[:forward_pass](states)
      for (i,obj) in enumerate(q.stash)
         kind, state, proc = obj
         # if kind == 0
         #    put!(res_queue[proc],reshape(dist[i,:], (size(dist[i,:])...,1))') #Reshape creates 1×n Array{Float32,2}
         # else
         #    put!(res_queue[proc],[val[i]])
         # end
         if kind == 0
            r = QueueResult(proc,reshape(dist[i,:], (size(dist[i,:])...,1))',11.)  #Reshape creates 1×n Array{Float32,2}
            put!(res_queue[proc],r)
         else
            r = QueueResult(proc,ones(1,1)*11,val[i])  #Reshape creates 1×n Array{Float32,2}
            put!(res_queue[proc],r)
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
      cmd, proc, state, states, dists, vals, trigger, n_updates, name = take!(cmd_queue)
      if q.debug remotecall_fetch(println,1,string(cmd)*" "*string(proc)*" "*string(state)*" "*string(trigger)*" "*string(n_updates)*" "*string(name)) end
      if cmd == "stash_size"
         process_stash(q)
         if q.debug remotecall_fetch(println,1,trigger) end
         q.trigger = trigger
      elseif cmd == "add_samples_to_memory"
         # process_stash(q)
         q.py_class[:add_samples_to_memory](states, dists, vals)
         if q.debug remotecall_fetch(println,1,"memory updated") end
         # put!(res_queue[proc],[12])
         r = QueueResult(proc,zeros(1,1),12.)
         put!(res_queue[proc],r)
      elseif cmd == "update_network"
         process_stash(q)
         q.update_counter += n_updates
         for i in 1:n_updates
            q.py_class[:update_network]()
         end
         if q.debug remotecall_fetch(println,1,string(n_updates)*" updates") end
         if q.debug remotecall_fetch(println,1,"network updated") end
         # put!(res_queue[proc],[13])
         r = QueueResult(proc,zeros(1,1),13.)
         put!(res_queue[proc],r)
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
    put!(cmd_queue,("predict_value",myid(),converted_state,nothing,nothing,nothing,nothing,nothing,nothing))
    r = take!(res_queue[myid()])
    value = r.value
    if value == 12. || value == 13.
      remotecall_fetch(println,1,"in estimate_value got: "*string(r)*", process: "*string(myid()))
    end
    if r.process_id != myid() || r.distribution != ones(1,1)*11
      remotecall_fetch(println,1,"in estimate_value got: "*string(r)*", process: "*string(myid()))
   end
    value = value*(estimator.v_max-estimator.v_min)+estimator.v_min #Scale [0,1]->[v_min,v_max]
    return value
end

function estimate_distribution(estimator::NNEstimatorParallel, state, allowed_actions, p::Union{POMDP,MDP})
    converted_state = convert_state(state, p)
    put!(cmd_queue,("predict_distribution",myid(),converted_state,nothing,nothing,nothing,nothing,nothing,nothing))
    r = take!(res_queue[myid()])
    dist = r.distribution
    if r.value == 12 || r.value == 13.
      remotecall_fetch(println,1,"in predict_distribution got: "*string(r)*", process: "*string(myid()))
    end
    if r.process_id != myid() || r.value != 11.
      remotecall_fetch(println,1,"in predict_distribution got: "*string(r)*", process: "*string(myid()))
   end
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
    put!(cmd_queue,("add_samples_to_memory",myid(),nothing,converted_states,dists,vals,nothing,nothing,nothing))
    out = take!(res_queue[myid()])
    if out.value != 12.
      remotecall_fetch(println,1,"in add_samples_to_memory got: "*string(out)*", process: "*string(myid()))
    end
    # print("add samples to memory out: "*string(out)*", id: "*string(myid()))
end

function update_network(estimator::NNEstimatorParallel,n_updates::Int=1)
    put!(cmd_queue,("update_network",myid(),nothing,nothing,nothing,nothing,nothing,n_updates,nothing))
    out = take!(res_queue[myid()])
    if out.value != 13.
      remotecall_fetch(println,1,"in update_network got: "*string(out)*", process: "*string(myid()))
    end
    # print("update network out: "*string(out)*", id: "*string(myid()))
end

function save_network(estimator::NNEstimatorParallel, name::String)
    put!(cmd_queue,("save",myid(),nothing,nothing,nothing,nothing,nothing,nothing,name))
end

function load_network(estimator::NNEstimatorParallel, name::String)
    put!(cmd_queue,("load",myid(),nothing,nothing,nothing,nothing,nothing,nothing,name))
end

function set_stash_size(estimator::NNEstimatorParallel, stash_size::Int)
   put!(cmd_queue,("stash_size",myid(),nothing,nothing,nothing,nothing,stash_size,nothing,nothing))
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
