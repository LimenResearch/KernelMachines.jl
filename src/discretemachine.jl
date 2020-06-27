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
struct DiscreteMachine{A, D, K, V, N}
    augmenter::A
    data::D
    kernel::K
    css::NTuple{N, V}
end
Base.show(io::IO, d::DiscreteMachine) = print(io, "DiscreteMachine {...}")

function consume(kernel, xss, css, indices=axes(first(xss), 2))
    k, res = nothing, first(xss)
    sn = zero(eltype(res))
    # Iteratively update k (kernel), res (result so far), and sn (square norm)
    for (xs, cs) in zip(tail(xss), css)
        k = kernel(k, res[:, indices], res)
        val = cs * k
        res = xs + val
        sn += dot(val[:, indices], cs)
    end
    return res, sn
end

function (dm::DiscreteMachine)(input)
    augmenter, data, kernel, css = dm.augmenter, dm.data, dm.kernel, dm.css
    full = hcat(data, input)
    xss, cost = augmenter(full)
    res, sn = consume(kernel, xss, css, axes(data, 2))
    return res, cost + sn
end
