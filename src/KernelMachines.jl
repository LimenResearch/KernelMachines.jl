module KernelMachines

using Base: tail
using LinearAlgebra: dot
using ZygoteRules: @adjoint

export KernelMachine

include("utils.jl")
include("kernelmachine.jl")

end # module
