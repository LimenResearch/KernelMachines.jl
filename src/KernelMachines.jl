module KernelMachines

using Base: tail
using LinearAlgebra: dot
using ZygoteRules: @adjoint

export DiscreteMachine

include("discretemachine.jl")

end # module
