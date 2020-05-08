module KernelMachines

using Random, Statistics
import LinearAlgebra: dot
using Base: front, tail
using Zygote: gradient
using Flux: @functor, glorot_uniform, Flux
using Optim: optimize, minimizer

export KernelMachine, KernelLayer

include("kernelmachine.jl")
include("regression.jl")

end # module
