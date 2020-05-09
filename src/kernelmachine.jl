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

add(::Nothing, t) = t
add(s, t) = s + t

function (kl::KernelLayer)(k, v)
    xs, cs = something(kl.xs, v), kl.cs
    ker = add(k, radialkernel(xs, v))
    return ker, cs * ker
end

function dot(kl1::KernelLayer, kl2::KernelLayer, k=nothing)
    _, val = kl1(k, kl2.xs)
    return dot(val, kl2.cs)
end

struct KernelMachine{S, T, N}
    layers::NTuple{N, KernelLayer{S, T}}
end

function KernelMachine(dims::Tuple, n)
    ins, outs = front(dims), tail(dims)
    layers = map((in, out) -> KernelLayer((in, out), n), ins, outs)
    return KernelMachine(layers)
end

function KernelMachine(css::NTuple{N, AbstractArray}) where N
    layers = map(cs -> KernelLayer(nothing, cs), css)
    return KernelMachine(layers)
end

function KernelMachine(km::KernelMachine, res::Tuple)
    ls = map(km.layers, front(res)) do l, xs
        KernelLayer(xs, l.cs)
    end
    return KernelMachine(ls)
end

@functor KernelMachine

# return results and partial cs * ker
function consume(layers, m)
    init = (nothing, (first(m),), (), 1)
    _, res, vals, _ = foldl(layers, init=init) do (k, r, v, i), layer
        k̂, val = layer(k, last(r))
        î = i + 1
        r̂ = (r..., add(m[î], val))
        v̂ = (v..., val)
        return k̂, r̂, v̂, î
    end
    return res, vals
end

function (km::KernelMachine)(m::Tuple)
    res, _ = consume(km.layers, m)
    return res
end

function dot(km1::KernelMachine, km2::KernelMachine)
    itr = zip(km1.layers, km2.layers)
    _, s = foldl(itr, init=(nothing, false)) do (k, acc), (l1, l2)
        k̂, v = l1(k, l2.xs)
        return k̂, acc + dot(v, l2.cs)
    end
    return s
end
