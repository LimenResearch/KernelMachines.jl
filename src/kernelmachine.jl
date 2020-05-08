struct KernelLayer{S, T}
    xs::S
    cs::T
end

KernelLayer(p::Pair, n::Integer) = KernelLayer(Tuple(p), n)
function KernelLayer((a, b)::Tuple, n::Integer)
    xs = glorot_uniform(a, n)
    cs = glorot_uniform(b, n)
    return KernelLayer(xs, cs)
end

@functor KernelLayer

radialkernel(u, v) = radialkernel_add(u, v, false)

function radialkernel_add(u, v, ker)
    u′ = transpose(u)
    u′² = sum(abs2.(u′), dims=2)
    v² = sum(abs2.(v), dims=1)
    u′v = u′ * v
    @. exp(u′v - u′² / 2  - v² / 2) + ker
end

function (kl::KernelLayer)(maybe, v)
    xs, cs = something(kl.xs, v), kl.cs
    k = something(maybe, false)
    ker = radialkernel_add(xs, v, k)
    return ker, cs * ker
end

function dot(kl1::KernelLayer, kl2::KernelLayer)
    ker = radialkernel(kl1.xs, kl2.xs)
    return dot(kl1.cs * ker, kl2.cs)
end

struct KernelMachine{S, T, N}
    layers::NTuple{N, KernelLayer{S, T}}
end

function KernelMachine(dims::Tuple, n)
    ins, outs = front(dims), tail(dims)
    layers = map((in, out) -> KernelLayer((in, out), n), ins, outs)
    return KernelMachine(layers)
end

function KernelMachine(fkm::KernelMachine, res::Tuple)
    ls = map(fkm.layers, front(res)) do l, xs
        KernelLayer(xs, l.cs)
    end
    return KernelMachine(ls)
end

@functor KernelMachine

function consume(layers, m)
    init = (nothing, (first(m),), 1)
    _, res, _ = foldl(layers, init=init) do (k, r, i), layer
        k̂, val = layer(k, last(r))
        î = i + 1
        r̂ = (r..., m[î] + val)
        return k̂, r̂, î
    end
    return res
end

(km::KernelMachine)(m::Tuple) = consume(km.layers, m)

function dot(km1::KernelMachine, km2::KernelMachine)
    sum(map(dot, km1.layers, km2.layers))
end
