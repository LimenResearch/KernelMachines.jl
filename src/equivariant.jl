function adjusted_crosscorr(patch, image)
    s = (size(patch, 1), size(patch, 2))
    @assert all(isodd, s)
    pad = map(t -> div(t, 2), s)
    conv(image, patch, flipped=true, pad=pad)
end

flipchmb(v) = permutedims(v, (1, 2, 4, 3))

@nograd function one_patch(patch)
    fill!(patch[:, :, 1:1, 1:1], 1)
end

function equivariant_radialkernel(patch::AbstractArray{T, 4}, image;
    options...) where {T}

    cross = adjusted_crosscorr(patch, image)
    patch2 = sum(abs2, patch, dims=(1, 2, 3))
    const_patch = one_patch(patch)
    image2_equiv = sum(abs2, image, dims=3)
    image2 = adjusted_crosscorr(const_patch, image2_equiv)
    patch2′ = flipchmb(patch2)
    @. exp(cross - image2 / 2 - patch2′ / 2)
end

# TODO: give abstract type and define layer interface
struct EquivariantKernelLayer{S, T}
    xs::S
    cs::T
end

function EquivariantKernelLayer(sz::Tuple, p::Pair{<:Tuple, <:Tuple}, n::Integer)
    EquivariantKernelLayer(sz, Tuple(p), n)
end
function EquivariantKernelLayer(sz::Tuple, (in, out)::Tuple, n::Integer)
    xs = glorot_uniform(sz..., in, n)
    cs = glorot_uniform(sz..., out, n)
    return EquivariantKernelLayer(xs, cs)
end

@functor EquivariantKernelLayer

function (kl::EquivariantKernelLayer)(k, image)
    xs, cs = kl.xs, kl.cs
    nk = equivariant_radialkernel(xs, image)
    ker = add(k, nk)
    return ker, adjusted_crosscorr(flipchmb(cs), ker)
end

struct EquivariantKernelMachine{S, T, N}
    layers::NTuple{N, EquivariantKernelLayer{S, T}}
end

function EquivariantKernelMachine(sz::Tuple, dims::Tuple, n)
    ins, outs = front(dims), tail(dims)
    layers = map((in, out) -> EquivariantKernelLayer(sz, (in, out), n), ins, outs)
    return EquivariantKernelMachine(layers)
end

@functor EquivariantKernelMachine

function (km::EquivariantKernelMachine)(m::Tuple)
    res, _ = consume(km.layers, m)
    return res
end
