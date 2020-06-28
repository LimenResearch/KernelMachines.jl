module KernelMachines

using Base: tail
using LinearAlgebra: dot
using ZygoteRules: @adjoint

export KernelMachine

include("kernelmachine.jl")

end # module
