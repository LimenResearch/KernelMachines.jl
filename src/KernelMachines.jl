module KernelMachines

using Random, Statistics
import LinearAlgebra: dot
using Base: front, tail
using Zygote: gradient, @adjoint, pullback
using Flux: @functor, glorot_uniform, Flux
using Optim: optimize, minimizer, only_fg!, ConjugateGradient

export KernelMachine, KernelLayer, Splitter, Linear

include("kernelmachine.jl")
include("regression.jl")
include("extras.jl")

end # module
