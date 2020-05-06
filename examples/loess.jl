using Loess, Plots, KernelMachines
using Plots.Measures
using Optim
using Zygote: gradient
using Flux, LinearAlgebra
using Random: seed!
seed!(0)

xs = 12 .* rand(100) .- 6
ys = sin.(xs) .+ rand(100)

model = loess(xs, ys)

us = range(extrema(xs)...; step = 0.1)
vs = predict(model, us)

model = Chain(
    Linear(1, 6),
    KernelMachine((2, 2, 2), 5),
    Linear(6, 1)) |> f64

ps, re = Flux.destructure(model)

function f(ps, cost)
    model = re(ps)
    s = sum(abs2, vec(model(permutedims(xs))) - ys)
    return s + cost * sum(dot(l, l) for l in model.layers)
end

costs = Any[0.1, 1, 10, 100]
results = map(costs) do c
    func = ps -> f(ps, c)
    func! = (G, ps) -> (G .= only(gradient(func, ps)))
    Optim.optimize(func, func!, ps, BFGS())
end

##

theme(:wong)
default(legendfont=14, tickfont=14, guidefont=14, size=(800, 600))

styles = [:solid, :dash, :dashdot, :dashdotdot]

plt = scatter(
    xs, ys,
    label = false, legend = :topright,
    color="black", primary=false,
    bottom_margin=5mm,
    xlabel = "x",
    ylabel="sine(x) + rand()",
    xlims=(-6.5, 6.5),
    ylims=(-1, 2)
    )

plot!(plt, us, sin.(us) .+ 0.5, label="Truth", linewidth=2,
    color="black")

for (i, (c, r)) in enumerate(zip(costs, results))
    vals = re(r.minimizer)(us') |> vec
    plot!(plt, us, vals, label="Cost $c", linewidth=2,
        color=Plots.palette(:wong)[1],
        linestyle=styles[i])
end

plot!(plt, us, vs, label = "Loess", linewidth=2,
    color=Plots.palette(:wong)[2])
