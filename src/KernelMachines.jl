module KernelMachines

import LinearAlgebra: dot
using Flux: @functor, glorot_uniform, Flux
using Zygote: Buffer

export KernelMachine, Linear

include("kernelmachine.jl")
include("linear.jl")

end # module
