# TODO: make @functor and trainable
# TODO: what if `kernel` is operator-valued?
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
    KernelMachine(additiveradialkernel, data; kwargs...)
end

Base.show(io::IO, d::KernelMachine) = print(io, "KernelMachine {...}")

function consume(kernel, xss, css, indices=axes(first(xss), 2))
    res = first(xss)
    k, sn = nothing, zero(eltype(res))
    # Iteratively update k (kernel), res (result so far), and sn (square norm)
    for (xs, cs) in zip(tail(xss), css)
        k = kernel(k, res[:, indices], res)
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
    xss, css = slice(xs, dims), slice(cs, tail(dims))
    res, sn = consume(kernel, xss, css, axes(data, 2))
    return res, cost + sn
end
