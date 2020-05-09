struct KernelRegression{S, T, M}
    input::S
    output::T
    machine::M
    result::OptimizationResults
end

function get_dims(km::KernelMachine)
    layers = km.layers
    init = (size(first(layers).xs, 1),)
    foldl(layers, init=init) do acc, layer
        (acc..., size(layer.cs, 1))
    end
end

function adjust(X, dims)
    s1, s2 = size(X)
    M = fill!(similar(X, sum(dims), s2), 0)
    M[1:s1, :] .= X
    return M
end

function ranges(dims)
    foldl(dims, init=()) do tup, dim
        l = isempty(tup) ? 0 : last(last(tup))
        (tup..., l+1:l+dim)
    end
end

function fit(
    ::Type{<:KernelRegression},
    X::AbstractArray,
    Y::AbstractArray,
    alg=ConjugateGradient();
    dims,
    cost,
    kwargs...)

    rgs = ranges(dims)
    M = adjust(X, dims)
    Ms = map(rg -> M[rg, :], rgs)

    dim = first(dims)
    C = similar(M, size(M, 1) - dim, size(M, 2))
    C .= glorot_uniform(size(C)...)
    Cs = map(rg -> C[rg .- dim, :], tail(rgs))

    function loss(Cs...)
        km = KernelMachine(Cs)
        r, vals = consume(km.layers, Ms)
        R = vcat(r...) # TODO make more efficient
        Ŷ = R[end + 1 - size(Y, 1):end, :]
        err = mean(abs2, Ŷ - Y)
        norm² = sum(map(dot, vals, Cs))
        return err + cost * norm²
    end
    
    function fg!(_, G, w)
        copy!(Params(Cs), w)
        if isnothing(G)
            l = loss(Cs...)
        else
            l, back = pullback(loss, Cs...)
            gs = back(one(l))
            copy!(G, Params(gs))
        end
        return l
    end

    res = optimize(only_fg!(fg!), vec(C), alg; kwargs...)
    pkm = KernelMachine(Cs)
    km = KernelMachine(pkm, pkm(Ms))
    return KernelRegression(X, Y, km, res)
end

function predict(kr::KernelRegression, X=kr.input)
    
    km = kr.machine
    dims = get_dims(km)
    M = adjust(X, dims)

    rgs = ranges(dims)
    Ms = map(rg -> M[rg, :], rgs)

    r = km(Ms)
    R = vcat(r...)
    sz = size(kr.output, 1)
    return R[end+1-sz:end, :]
end