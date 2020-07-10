module KernelMachines

using Base: tail
using LinearAlgebra: dot, I
using Statistics: mean
using ChainRulesCore: NO_FIELDS
using Zygote: pullback, @nograd
using Optim: optimize,
             minimizer,
             OptimizationResults,
             only_fg!,
             ConjugateGradient,
             AbstractOptimizer,
             default_options,
             Options

import ChainRulesCore: rrule
import StatsBase: fit!, predict

export KernelMachine, KernelRegression, KernelMachineRegression

export additivegaussiankernel, multiplicativegaussiankernel, gaussiankernel

export fit!, predict

include("utils.jl")
include("kernels.jl")
include("kernelmachine.jl")
include("regression.jl")
include("machineregression.jl")

end # module
