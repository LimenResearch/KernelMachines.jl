module KernelMachines

using LinearAlgebra: dot
using ZygoteRules: @adjoint

export KernelMachine

include("kernelmachine.jl")

end # module
