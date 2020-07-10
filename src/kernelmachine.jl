"""
    KernelMachine(kernel, augmenter, data, cs, dims)

Return a `KernelMachine`. `kernel` is required to have a method `kernel(u, v, k=nothing)`,
where `u` and `v` are matrices (datapoints are columns), and `k` is the kernel evaluated
in the previous hidden space. See [`additivegaussiankernel`](@ref) for an example.
`augmenter` is a matrix mapping the input to the machine space.
`cs` is the matrix of coefficient corresponding to `data`. Finally, `dims` is a tuple
containing the dimensionality of the spaces that compose the machine space. In particular,
`size(augmenter, 1) == sum(dims)`.

Given a kernel machine `km`, `km(v)` returns the value of the stable state of the machine
on the last component of machine space, as well as the square norm of the machine plus the
square Frobenius norm of the augmenter. This second quantity should be used to regularize
the machine.

    KernelMachine(kernel, data::M; dims, init=glorot_uniform)

A simpler constructor. `kernel` is option and defaults to [`additivegaussiankernel`](@ref).
`data` is the training data, or, more generally, the set of anchor points (datapoints are columns).
`dims` is a tuple containing the dimensionality of the spaces that compose the machine space.
`init` is used to initialize the `cs` matrix.
"""
struct KernelMachine{K, M<:AbstractMatrix, N}
    kernel::K
    augmenter::M
    data::M
    cs::M
    dims::NTuple{N, Int}
end

function KernelMachine(kernel, data::M; dims, init=glorot_uniform) where M
    augmenter_size = sum(dims)
    cs_size = augmenter_size - first(dims)
    augmenter::M = init(augmenter_size, size(data, 1))
    cs::M = init(cs_size, size(data, 2))
    return KernelMachine(kernel, augmenter, data, cs, dims)
end

function KernelMachine(data; kwargs...)
    KernelMachine(additivegaussiankernel, data; kwargs...)
end

Base.show(io::IO, d::KernelMachine) = print(io, "KernelMachine {...}")

# TODO: what if `kernel` is operator-valued?
# If it is a relevant use case, it would require a different kernel API.
function consume(kernel, xss, css, indices=axes(first(xss), 2))
    res = first(xss)
    k, sn = nothing, zero(eltype(res))
    # Iteratively update k (kernel), res (result so far), and sn (square norm)
    for (xs, cs) in zip(tail(xss), css)
        k = kernel(res[:, indices], res, k)
        val = cs * k
        res = xs + val
        sn += dot(val[:, indices], cs)
    end
    return res, sn
end

maybe_hcat(x, ::Nothing) = x
maybe_hcat(x, y) = hcat(x, y)

function (dm::KernelMachine)(input)
    kernel, augmenter, data, cs, dims = 
        dm.kernel, dm.augmenter, dm.data, dm.cs, dm.dims
    full = maybe_hcat(data, input)
    xs = augmenter * full
    cost = sum(abs2, augmenter)
    xss, css = split_matrix(xs, dims), split_matrix(cs, tail(dims))
    res, sn = consume(kernel, xss, css, axes(data, 2))
    return res, cost + sn
end
