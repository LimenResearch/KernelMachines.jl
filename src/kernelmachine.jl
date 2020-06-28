function radialkernel(u, v)
    u2 = sum(abs2, u, dims=1)
    v2 = sum(abs2, v, dims=1)
    exp.(u' * v .- u2' ./ 2 .- v2 ./ 2)
end

@adjoint function radialkernel(u, v)
    val = radialkernel(u, v)
    back = function (m)
        a = m .* val
        û = v * a' - sum(a', dims=1) .* u
        v̂ = u * a - sum(a, dims=1) .* v
        return û, v̂
    end
    return val, back
end

radialkernel(k::Nothing, u, v) = radialkernel(u, v)
radialkernel(k, u, v) = k + radialkernel(u, v)

# TODO: make @functor and add simpler constructor
# TODO: what if `kernel` is operator-valued?
struct KernelMachine{K, M<:AbstractMatrix, N}
    kernel::K
    augmenter::M
    data::M
    cs::M
    dims::NTuple{N, Int}
end

function KernelMachine(kernel, data::AbstractMatrix{T}; dims, init=rand) where T
    augmenter_size = sum(dims)
    cs_size = augmenter_size - first(dims)
    augmenter = init(T, augmenter_size, size(data, 1))
    cs = init(T, cs_size, size(data, 2))
    d = convert(typeof(cs), data)
    return KernelMachine(kernel, augmenter, d, cs, dims)
end

KernelMachine(data; kwargs...) = KernelMachine(radialkernel, data; kwargs...)

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

function (dm::KernelMachine)(input)
    kernel, augmenter, data, cs, dims = 
        dm.kernel, dm.augmenter, dm.data, dm.cs, dm.dims
    full = hcat(data, input)
    xs = augmenter * full
    cost = sum(abs2, augmenter)
    xss, css = slice(xs, dims), slice(cs, tail(dims))
    res, sn = consume(kernel, xss, css, axes(data, 2))
    return res, cost + sn
end
