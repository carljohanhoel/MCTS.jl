module DummyModule

using PyCall

export PyConnection, get_state

mutable struct PyConnection
    py_class::PyCall.PyObject
end

function PyConnection()
    unshift!(PyVector(pyimport("sys")["path"]), "/home/cj/2018/Stanford/Code/Multilane.jl/src/")
    # eval(parse(string("@pyimport ", "ParallelJuliaTest", " as python_module")))
    @everywhere @pyimport ParallelJuliaTest as python_module
    py_class = python_module.ParallelJuliaTest(0)
    return PyConnection(py_class)
end

function get_state(py_connection::PyConnection)
    py_connection.py_class[:get_state]()
end

function test()
    return rand(1)
end

end
