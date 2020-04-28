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

function update(u′, v, old=true)
    u2′ = sum(abs2.(u′), dims=2)
    v2 = sum(abs2.(v), dims=1)
    u′v = u′ * v
    @. exp(2u′v - u2′ - v2) * old
end

function (kl::KernelLayer)(old, v)
    ker = update(kl.xs′, v, old)
    val = kl.cs * ker
    return ker, val
end

function dot(kl1::KernelLayer, kl2::KernelLayer)
    ker = update(kl1.xs′, transpose(kl2.xs′))
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

function to_buffer(v)
    b = Buffer(v)
    b .= v
    return b
end

function (kn::KernelNetwork)(m)
    ls = kn.layers
    i0, i1 = 1, size(first(ls).xs′, 2)
    buffer = to_buffer(m)
    foldl(ls, init=true) do acc, l
        ker, val = l(acc, buffer[i0:i1, :])
        i0 = i1 + 1
        i1 += size(val, 1)
        buffer[i0:i1, :] += val
        return ker
    end
    return copy(buffer)
end

function dot(kn1::KernelNetwork, kn2::KernelNetwork)
    sum(map(dot, kn1.layers, kn2.layers))
end
