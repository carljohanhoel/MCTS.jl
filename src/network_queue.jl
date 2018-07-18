struct QueueCommand
   activity::Int
   proc_id::Int
   state::Array{Float64,2}
   states::Array{Float64,2}
   dists::Array{Float64,2}
   vals::Array{Float64,1}
   trigger::Int
   n_updates::Int
   name::String
end

function QueueCommand(
   activity::Int, #stash_size, add_samples_to_memory, update_network, predict_distribution, predict_value, save, load
   proc_id::Int;
   state::Array{Float64,2}=zeros(1,1),
   states::Array{Float64,2}=zeros(1,1),
   dists::Array{Float64,2}=zeros(1,1),
   vals::Array{Float64,1}=zeros(1),
   trigger::Int=0,
   n_updates::Int=0,
   name::String=""
   )
   return QueueCommand(activity, proc_id, state, states, dists, vals, trigger, n_updates, name)
end

# struct QueueResult
#    process_id::Int
#    distribution::Array{Float64,2}
#    value::Float64
# end
struct QueueResult
   process_id::Int
   distribution::PyCall.PyVector{PyCall.PyObject}
   value::PyCall.PyVector{PyCall.PyObject}
end

function QueueResult(
   process_id::Int;
   distribution::PyCall.PyVector{PyCall.PyObject}=PyVector([PyObject([0])]),
   value::PyCall.PyVector{PyCall.PyObject}=PyVector([PyObject([0])])
   )
   return QueueResult(process_id, distribution, value)
end

# const cmd_queue = RemoteChannel(()->Channel{Tuple}(128))
const cmd_queue = RemoteChannel(()->Channel{QueueCommand}(128))
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
         println("clearing queue: "*string(out))
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
      py_object = pycall(q.py_class[:forward_pass],PyObject,states)
      if q.debug remotecall_fetch(println,1,"forward pass done") end
      if q.debug remotecall_fetch(println,1,py_object) end
      for (i,obj) in enumerate(q.stash)
         kind, state, proc_id = obj
         # if kind == 0
         #    put!(res_queue[proc_id],QueueResult(proc_id,distribution=reshape(dist[i,:], (size(dist[i,:])...,1))'))  #Reshape creates 1×n Array{Float32,2}
         # else
         #    put!(res_queue[proc_id],QueueResult(proc_id,value=val[i]))
         # end
         if q.debug remotecall_fetch(println,1,"before putting in queue") end
         if kind == 0
            put!(res_queue[proc_id],QueueResult(proc_id,distribution=get(py_object,PyVector{PyObject},0)))
         else
            put!(res_queue[proc_id],QueueResult(proc_id,value=get(py_object,PyVector{PyObject},1)))
         end
         if q.debug remotecall_fetch(println,1,"after putting in queue") end
      end
      q.stash = Array{Tuple}(0)
   end
   function add(q::NetworkQueue, kind::Int, state::Array{Float64,2}, proc_id::Int)
      push!(q.stash,(kind,state,proc_id))
      if length(q.stash) >= q.trigger
         process_stash(q)
      end
   end

   while true
      if q.debug remotecall_fetch(println,1,"in loop") end
      # cmd, proc, state, states, dists, vals, trigger, n_updates, name = take!(cmd_queue)
      cmd = take!(cmd_queue)
      # if q.debug remotecall_fetch(println,1,string(cmd)*" "*string(proc)*" "*string(state)*" "*string(trigger)*" "*string(n_updates)*" "*string(name)) end
      if q.debug remotecall_fetch(println,1,string(cmd.activity)*" "*string(cmd.proc_id)*" "*string(cmd.state)*" "*string(cmd.trigger)*" "*string(cmd.n_updates)*" "*string(cmd.name)) end
      if cmd.activity == 1
         process_stash(q)
         if q.debug remotecall_fetch(println,1,cmd.trigger) end
         q.trigger = cmd.trigger
      elseif cmd.activity == 2
         # process_stash(q)
         q.py_class[:add_samples_to_memory](cmd.states, cmd.dists, cmd.vals)
         if q.debug remotecall_fetch(println,1,"memory updated") end
         # put!(res_queue[proc],[12])
         put!(res_queue[cmd.proc_id],QueueResult(cmd.proc_id,value=PyVector([PyObject([12.])])))
      elseif cmd.activity == 3
         process_stash(q)
         q.update_counter += cmd.n_updates
         for i in 1:cmd.n_updates
            q.py_class[:update_network]()
         end
         if q.debug remotecall_fetch(println,1,string(cmd.n_updates)*" updates") end
         if q.debug remotecall_fetch(println,1,"network updated") end
         # put!(res_queue[proc],[13])
         put!(res_queue[cmd.proc_id],QueueResult(cmd.proc_id,value=PyVector([PyObject([13.])])))
      elseif cmd.activity == 4
         add(q,0,cmd.state,cmd.proc_id)
         if q.debug remotecall_fetch(println,1,"pred dist added") end
      elseif cmd.activity == 5
         add(q,1,cmd.state,cmd.proc_id)
         if q.debug remotecall_fetch(println,1,"pred val added") end
      elseif cmd.activity == 6
         process_stash(q)
         q.py_class[:save_network](cmd.name)
         if q.debug remotecall_fetch(println,1,"save net as "*cmd.name) end
      elseif cmd.activity == 7
         process_stash(q)
         q.py_class[:load_network](cmd.name)
         remotecall_fetch(println,1,"load net "*cmd.name)
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
    put!(cmd_queue,QueueCommand(5,myid(),state=converted_state))
    r = take!(res_queue[myid()])
    value = convert(Float64,r.value[1])
    distribution = convert(Array{Float64},r.distribution[1])
    if value == 12. || value == 13.
      remotecall_fetch(println,1,"in estimate_value got: "*string(r)*", process: "*string(myid()))
    end
    if r.process_id != myid() || distribution != [0.0]
       remotecall_fetch(println,1,"in estimate_value got: "*string(r)*", process: "*string(myid()))
    end
    value = value*(estimator.v_max-estimator.v_min)+estimator.v_min #Scale [0,1]->[v_min,v_max]
    return value
end

function estimate_distribution(estimator::NNEstimatorParallel, state, allowed_actions, p::Union{POMDP,MDP})
    converted_state = convert_state(state, p)
    put!(cmd_queue,QueueCommand(4,myid(),state=converted_state))
    r = take!(res_queue[myid()])
    tmp = convert(Array{Float64},r.distribution[1])
    dist = reshape(tmp, (size(tmp)...,1))'
    value = convert(Float64,r.value[1])
    if value == 12 || value == 13.
      remotecall_fetch(println,1,"in predict_distribution got: "*string(r)*", process: "*string(myid()))
    end
   if r.process_id != myid() || value != 0
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
    put!(cmd_queue,QueueCommand(2,myid(),states=converted_states,dists=dists,vals=vals))
    out = take!(res_queue[myid()])
    value = convert(Float64,out.value[1])
    if value != 12.
      remotecall_fetch(println,1,"in add_samples_to_memory got: "*string(out)*", process: "*string(myid()))
    end
    # print("add samples to memory out: "*string(out)*", id: "*string(myid()))
end

function update_network(estimator::NNEstimatorParallel,n_updates::Int=1)
    put!(cmd_queue,QueueCommand(3,myid(),n_updates=n_updates))
    out = take!(res_queue[myid()])
    value = convert(Float64,out.value[1])
    if value != 13.
      remotecall_fetch(println,1,"in update_network got: "*string(out)*", process: "*string(myid()))
    end
    # print("update network out: "*string(out)*", id: "*string(myid()))
end

function save_network(estimator::NNEstimatorParallel, name::String)
    put!(cmd_queue,QueueCommand(6,myid(),name=name))
end

function load_network(estimator::NNEstimatorParallel, name::String)
    put!(cmd_queue,QueueCommand(7,myid(),name=name))
end

function set_stash_size(estimator::NNEstimatorParallel, stash_size::Int)
   put!(cmd_queue,QueueCommand(1,myid(),trigger=stash_size))
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
