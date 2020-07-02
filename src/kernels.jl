function radialkernel(u, v)
    uu = sum(abs2, u, dims=1)
    vv = sum(abs2, v, dims=1)
    acc = u' * v
    @. acc = exp(acc - uu' / 2 - vv / 2)
    return acc
end

@adjoint function radialkernel(u, v)
    r = radialkernel(u, v)
    # ∂r is short for ∂l / ∂r
    function radialkernel_pullback(∂r)
        m = r .* ∂r # pullback the exponential
        ∂u = v * m'
        ∂u .-= sum(m', dims=1) .* u
        ∂v = u * m
        ∂v .-= sum(m, dims=1) .* v
        return ∂u, ∂v
    end
    return r, radialkernel_pullback
end

radialkernel(k::Nothing, u, v) = radialkernel(u, v)
radialkernel(k, u, v) = k + radialkernel(u, v)