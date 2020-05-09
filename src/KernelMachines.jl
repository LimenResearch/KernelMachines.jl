module KernelMachines

using Random, Statistics
import LinearAlgebra: dot
import StatsBase: fit, predict
using Base: front, tail
using Zygote: gradient, @adjoint, pullback, Params
using Flux: @functor, glorot_uniform, Flux
using Optim: optimize,
             minimizer,
             only_fg!,
             ConjugateGradient,
             OptimizationResults

export KernelMachine, KernelLayer
export Splitter, Linear
export KernelRegression, fit, predict

include("kernelmachine.jl")
include("regression.jl")
include("extras.jl")

end # module
