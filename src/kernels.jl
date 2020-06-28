function radialkernel(u, v)
    uu = sum(abs2, u, dims=1)
    vv = sum(abs2, v, dims=1)
    acc = u' * v
    @. acc = exp(acc - uu' / 2 - vv / 2)
    return acc
end

@adjoint function radialkernel(u, v)
    val = radialkernel(u, v)
    back = function (m)
        a = m .* val
        û = v * a'
        @. û -= $sum(a', dims=1) * u
        v̂ = u * a
        @. v̂ -= $sum(a, dims=1) * v
        return û, v̂
    end
    return val, back
end

radialkernel(k::Nothing, u, v) = radialkernel(u, v)
radialkernel(k, u, v) = k + radialkernel(u, v)