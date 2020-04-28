using Loess, Plots, KernelNetworks
using Optim
using Zygote: gradient
using Flux, LinearAlgebra
using Random: seed!
seed!(0)

xs = 10 .* rand(100)
ys = sin.(xs) .+ rand(100)

model = loess(xs, ys)

us = range(extrema(xs)...; step = 0.1)
vs = predict(model, us)

model = Chain(
    Linear(1, 6),
    KernelNetwork((2, 2, 2), 5),
    Linear(6, 1)) |> f64

ps, re = Flux.destructure(model)

function f(ps, cost)
    model = re(ps)
    s = sum(abs2, vec(model(permutedims(xs))) - ys)
    return s + cost * sum(dot(l, l) for l in model.layers)
end

costs = [1, 10, 100]
results = map(costs) do c
    func = ps -> f(ps, c)
    func! = (G, ps) -> (G .= only(gradient(func, ps)))
    Optim.optimize(func, func!, ps, BFGS())
end

##

theme(:wong)

styles = [:solid, :dash, :dashdot]

plt = scatter(
    xs, ys,
    label = false, legend = :bottomleft,
    color="black", primary=false)

plot!(plt, us, sin.(us) .+ 0.5, label="Truth", linewidth=2,
    color="black")

for (i, (c, r)) in enumerate(zip(costs, results))
    vals = re(r.minimizer)(us') |> vec
    plot!(plt, us, vals, label="Cost = $c", linewidth=2,
        color=Plots.palette(:wong)[1],
        linestyle=styles[i])
end

plot!(plt, us, vs, label = "Loess", linewidth=2,
    color=Plots.palette(:wong)[2])
