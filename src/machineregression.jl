mutable struct KernelMachineRegression{M, O, C}
    machine::M
    output::O
    cost::C
    result::Union{OptimizationResults, Nothing}
end

function KernelMachineRegression(X::AbstractArray, Y::AbstractArray;
    kernel=additivegaussiankernel, cost=0, dims)

    machine = KernelMachine(kernel, permutedims(X); dims=(dims..., size(Y, 2)))
    output = permutedims(Y)
    return KernelMachineRegression(machine, output, cost, nothing)
end

Base.show(io::IO, d::KernelMachineRegression) = print(io, "KernelMachineRegression {...}")

loss(k::KernelMachineRegression) = loss(k.machine, k.output, k.cost)

function loss(km::KernelMachine, output, cost)
    pred, sq_norm = km(nothing)
    mean(abs2, pred - output) + cost * sq_norm
end

function updatemachine!(k::KernelMachineRegression, w)
    km = k.machine
    aug, cs = km.augmenter, km.cs
    n_aug, n_cs = length(aug), length(cs)
    copyto!(aug, 1, w, 1, n_aug)
    copyto!(cs, 1, w, n_aug + 1, n_cs)
    return
end

function compute_fg(k::KernelMachineRegression)
    function fg!(_, G, w)
        updatemachine!(k, w)
        if isnothing(G)
            l = loss(k)
        else
            l, back = pullback(loss, k.machine, k.output, k.cost)
            gs, = back(one(l))
            copyto!(G, gs.augmenter)
            copyto!(G, length(gs.augmenter) + 1, gs.cs)
        end
        return l
    end
end

const default_method = ConjugateGradient()

function fit!(k::KernelMachineRegression,
    method::AbstractOptimizer=default_method,
    options::Options=Options(; default_options(method)...))

    fg! = compute_fg(k)
    aug, cs = k.machine.augmenter, k.machine.cs
    init = vcat(vec(aug), vec(cs))
    res = optimize(only_fg!(fg!), init, method, options)
    updatemachine!(k, minimizer(res))
    k.result = res
    return k
end

function predict(k::KernelMachineRegression, input=nothing)
    km = k.machine
    if isnothing(input)
        res, _ = km(nothing)
    else
        nsamples = size(km.data, 2)
        res_full, _ = km(permutedims(input))
        res = res_full[:, nsamples+1:end]
    end
    return permutedims(res)
end