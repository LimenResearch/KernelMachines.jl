function combined(op, kernel_function)
    function combined_kernel_function(u, v, k=nothing)
        ker = kernel_function(u, v)
        isnothing(k) ? ker : op.(k, ker)
    end
end

# Kernel functions

"""
    gaussiankernel(u, v)

Compute the Gaussian kernel, given by the formula
```
    κ(x,y) = exp(-‖x-y‖²/2).
```
`u` and `v` are matrices, where each column is a datapoint.
"""
function gaussiankernel(u, v)
    uu = sum(abs2, u, dims=1)
    vv = sum(abs2, v, dims=1)
    acc = u' * v
    @. acc = exp(acc - uu' / 2 - vv / 2)
    return acc
end

function rrule(::typeof(gaussiankernel), u, v)
    r = gaussiankernel(u, v)
    # r̄ is short for ∂l / ∂r
    function gaussiankernel_pullback(r̄)
        m = r .* r̄ # pullback the exponential
        ū = v * m'
        ū .-= sum(m', dims=1) .* u
        v̄ = u * m
        v̄ .-= sum(m, dims=1) .* v
        return NO_FIELDS, ū, v̄
    end
    return r, gaussiankernel_pullback
end

"""
    additivegaussiankernel(u, v, k=nothing)

Compute the radial basis function kernel (see [`gaussiankernel`](@ref)) on `u`, `v`.
If `k` is not `nothing`, add it to the result.
"""
const additivegaussiankernel = combined(+, gaussiankernel)

"""
    additivegaussiankernel(u, v, k=nothing)

Compute the radial basis function kernel (see [`gaussiankernel`](@ref)) on `u`, `v`.
If `k` is not `nothing`, multiply the result by it.
"""
const multiplicativegaussiankernel = combined(*, gaussiankernel)
