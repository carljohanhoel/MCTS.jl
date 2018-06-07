# @everywhere using ParallelDataTransfer
push!(LOAD_PATH,joinpath("./test"))

# @everywhere using PyCall
@everywhere using DummyModule

# addprocs(1)




# @everywhere function test()
#     return rand(1)
# end

p = PyConnection()
# sendto(2,p=p)

state = 1

addprocs(1)

r1 = remotecall(p.py_class[:get_state],1)
fetch(r1)

rr1 = remotecall(get_state,1,p)
fetch(rr1)

r2 = remotecall(p.py_class[:get_state],2)
fetch(r2)

rr2 = remotecall(get_state,2,p)
fetch(rr2)

m1 = @spawnat 2 test()
fetch(m1)

m2 = @spawnat 2 p
fetch(m2)

rr1 = remotecall(test,2)
fetch(rr1)


b = PyConnection()

m1 = @spawnat 2 b
fetch(m1)
