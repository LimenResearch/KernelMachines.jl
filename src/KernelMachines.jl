module KernelMachines

using Base: tail
using LinearAlgebra: dot
using ZygoteRules: @adjoint

export KernelMachine, radialkernel

include("utils.jl")
include("kernels.jl")
include("kernelmachine.jl")

end # module
