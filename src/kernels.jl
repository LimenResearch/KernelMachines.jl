abstract type AbstractKernel end

(ker::AbstractKernel)(u, v) = compute(ker, u, v)

(ker::AbstractKernel)(::Nothing, u, v) = ker(u, v)

function (ker::AbstractKernel)(k1, u, v)
    k2 = compute(ker, u, v)
    return combine(ker, k1, k2)
end

# Example AbstractKernels

struct AdditiveRadialKernel <: AbstractKernel; end

const additiveradialkernel = AdditiveRadialKernel()

combine(::AdditiveRadialKernel, k1, k2) = k1 + k2

compute(::AdditiveRadialKernel, u, v) = radialkernel(u, v)

struct MultiplicativeRadialKernel <: AbstractKernel; end

combine(::MultiplicativeRadialKernel, k1, k2) = k1 .* k2

compute(::MultiplicativeRadialKernel, u, v) = radialkernel(u, v)

const multiplicativeradialkernel = MultiplicativeRadialKernel()

function radialkernel(u, v)
    uu = sum(abs2, u, dims=1)
    vv = sum(abs2, v, dims=1)
    acc = u' * v
    @. acc = exp(acc - uu' / 2 - vv / 2)
    return acc
end

@adjoint function radialkernel(u, v)
    r = radialkernel(u, v)
    # r̄ is short for ∂l / ∂r
    function radialkernel_pullback(r̄)
        m = r .* r̄ # pullback the exponential
        ū = v * m'
        ū .-= sum(m', dims=1) .* u
        v̄ = u * m
        v̄ .-= sum(m, dims=1) .* v
        return ū, v̄
    end
    return r, radialkernel_pullback
end
