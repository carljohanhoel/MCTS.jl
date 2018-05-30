using PyCall
using MCTS
using POMDPModels

mdp = GridWorld(5,5,
                penalty=-1.,
                rs=[GridWorldState(3,3),GridWorldState(3,1),GridWorldState(5,5),GridWorldState(3,5)],
                )
state = GridWorldState(1,1)

vec_state = MCTS.convert_state(state)

n_s = length(vec_state)
n_a = n_actions(mdp)

rng = MersenneTwister(12)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS")
estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a)

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180530_022108/50001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180530_200610/10001")

allowed_actions = [1.0, 1.0, 1.0, 1.0]

v = estimator.py_class[:estimate_value](vec_state)
p = estimator.py_class[:estimate_distribution](vec_state, allowed_actions)
println(v)
println(p)

estimator.py_class[:debug_save_input](vec_state, allowed_actions)
estimator.py_class[:debug_print_n_calls]()

estimator.py_class[:save_network](dirname(dirname(estimator_path))*"/Logs/ttt")

estimator.py_class[:load_network](dirname(dirname(estimator_path))*"/Logs/ttt")
