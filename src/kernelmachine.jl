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

radialkernel(u′, v) = radialkernel_add(u′, v, false)

function radialkernel_add(u′, v, ker)
    u′² = sum(abs2.(u′), dims=2)
    v² = sum(abs2.(v), dims=1)
    u′v = u′ * v
    @. exp(u′v - u′² / 2  - v² / 2) + ker
end

(kl::KernelLayer)(::Nothing, v) = kl(false, v)

function (kl::KernelLayer)(k, v)
    ker = radialkernel_add(kl.xs′, v, k)
    val = kl.cs * ker
    return ker, val
end

function dot(kl1::KernelLayer, kl2::KernelLayer)
    ker = radialkernel(kl1.xs′, transpose(kl2.xs′))
    return dot(kl1.cs * ker, kl2.cs)
end

struct KernelMachine{T}
    layers::Vector{KernelLayer{T}}
end

function KernelMachine(sizes::Tuple, n)
    layers = KernelLayer.(zip(Base.front(sizes), Base.tail(sizes)), n)
    return KernelMachine(layers)
end

@functor KernelMachine

function (km::KernelMachine)(m)
    ls = km.layers
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

function dot(km1::KernelMachine, km2::KernelMachine)
    sum(map(dot, km1.layers, km2.layers))
end
