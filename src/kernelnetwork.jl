struct KernelLayer{T}
    xs′::T
    cs::T
end

KernelLayer(p::Pair, n::Integer) = KernelLayer(Tuple(p), n)
function KernelLayer((a, b)::Tuple, n::Integer)
    xs′ = glorot_uniform(n, a)
    cs = glorot_uniform(b, n)
    return KernelLayer(xs′, cs)
end

@functor KernelLayer

radialkernel(u′, v) = radialkernel_multiply(u′, v, true)

function radialkernel_multiply(u′, v, ker)
    u2′ = sum(abs2.(u′), dims=2)
    v2 = sum(abs2.(v), dims=1)
    u′v = u′ * v
    @. exp(2u′v - u2′ - v2) * ker
end

(kl::KernelLayer)(::Nothing, v) = kl(true, v)

function (kl::KernelLayer)(k, v)
    ker = radialkernel_multiply(kl.xs′, v, k)
    val = kl.cs * ker
    return ker, val
end

function dot(kl1::KernelLayer, kl2::KernelLayer)
    ker = radialkernel(kl1.xs′, transpose(kl2.xs′))
    coefs = transpose(kl1.cs) * kl2.cs
    return dot(coefs, ker)
end

struct KernelNetwork{T}
    layers::Vector{KernelLayer{T}}
end

function KernelNetwork(sizes::Tuple, n)
    layers = KernelLayer.(zip(Base.front(sizes), Base.tail(sizes)), n)
    return KernelNetwork(layers)
end

@functor KernelNetwork

function (kn::KernelNetwork)(m)
    ls = kn.layers
    buffer = Buffer(m)
    copyto!(buffer, m)
    ker = nothing
    idxs::UnitRange = axes(first(ls).xs′, 2)
    for l in ls
        ker, val = l(ker, buffer[idxs, :])
        idxs = last(idxs) .+ axes(val, 1)
        buffer[idxs, :] += val
    end
    return copy(buffer)
end

function dot(kn1::KernelNetwork, kn2::KernelNetwork)
    sum(map(dot, kn1.layers, kn2.layers))
end
