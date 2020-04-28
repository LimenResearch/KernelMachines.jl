module KernelNetworks

import LinearAlgebra: dot
using Flux: @functor, glorot_uniform, Flux
using Zygote: Buffer

export KernelLayer, KernelNetwork, Linear

include("kernelnetwork.jl")
include("linear.jl")

end # module
