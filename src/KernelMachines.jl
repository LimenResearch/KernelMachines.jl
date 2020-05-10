module KernelMachines

using Random, Statistics
import LinearAlgebra: dot
import StatsBase: fit, predict
using Base: front, tail
using Zygote: gradient, @adjoint, pullback, Params, @nograd
using Flux: @functor, glorot_uniform, Flux
using NNlib: conv
using Optim: optimize,
             minimizer,
             only_fg!,
             ConjugateGradient,
             OptimizationResults,
             AbstractOptimizer,
             Options,
             default_options

export KernelMachine, KernelLayer
export EquivariantKernelMachine, EquivariantKernelLayer
export Splitter, Linear
export KernelRegression, fit, predict

include("kernelmachine.jl")
include("equivariant.jl")
include("regression.jl")
include("extras.jl")

end # module
