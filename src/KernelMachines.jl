module KernelMachines

using Base: tail
using LinearAlgebra: dot
using Statistics: mean
using Zygote: @adjoint, pullback
using Optim: optimize,
             minimizer,
             OptimizationResults,
             only_fg!,
             ConjugateGradient,
             AbstractOptimizer,
             default_options,
             Options

import StatsBase: fit!, predict

export KernelMachine, KernelRegression, radialkernel

include("utils.jl")
include("kernels.jl")
include("kernelmachine.jl")
include("regression.jl")

end # module
