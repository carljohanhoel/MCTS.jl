using PyCall
using POMDPs
include("../src/nn_estimator.jl")

##
n_s = 3
n_a = 4
v_max = 1*1.05
v_min = -v_max
replay_memory_max_size = 55
training_start = 40
rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)

##
function old_call(n::Int)
   for i in 1:n
      a,b = estimator.py_class[:forward_pass](rand(1,3))
   end
end

function new_call(n::Int)
   for i in 1:n
      a = pycall(estimator.py_class[:forward_pass],PyObject,rand(1,3))
   end
end

function new_call2(n::Int)
   for i in 1:n
      a,b = pycall(estimator.py_class[:forward_pass],PyObject,rand(1,3))
   end
end

function new_call3(n::Int)
   for i in 1:n
      a = pycall(estimator.py_class[:forward_pass],PyObject,rand(1,3))
      aa = get(a,PyVector{PyObject},0)
      bb = get(a,PyVector{PyObject},1)
   end
end

## Run twice, since compiling the first time
N = 1000
@time old_call(N)
@time new_call(N)
@time new_call2(N)
@time new_call3(N)

@time old_call(N)
@time new_call(N)
@time new_call2(N)
@time new_call3(N)
