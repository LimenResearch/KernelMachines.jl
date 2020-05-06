using Plots, KernelMachines
using Plots.Measures
using Optim
using Zygote: gradient
using Flux, LinearAlgebra
using Random: seed!
seed!(0)

xs = 20 .* rand(100) .- 10
ys = sin.(xs) .+ rand(100)

us = range(extrema(xs)...; step = 0.1)

model = Chain(
    Linear(1, 12),
    KernelMachine((3, 3, 3, 3), 30),
    Linear(12, 1)) |> f64

ps, re = Flux.destructure(model)

function f(ps, cost)
    model = re(ps)
    s = sum(abs2, vec(model(permutedims(xs))) - ys)
    return s + cost * sum(dot(l, l) for l in model.layers)
end

costs = [1, 5, 10]
results = map(costs) do c
    func = ps -> f(ps, c)
    func! = (G, ps) -> (G .= only(gradient(func, ps)))
    Optim.optimize(func, func!, ps, BFGS())
end

##

theme(:wong)
default(legendfont=14, tickfont=14, guidefont=14, size=(800, 600))

styles = [:solid, :dash, :dashdot]

plt = scatter(
    xs, ys,
    label = false, legend = :topleft,
    color="black", primary=false,
    bottom_margin=5mm,
    xlabel = "x",
    ylabel="sine(x) + rand()",
    xlims=(-12.5, 12.5),
    ylims=(-1, 2)
    )

plot!(plt, us, sin.(us) .+ 0.5, label="Truth", linewidth=2,
    color="black")

using PlotThemes: wong_palette

for (i, (c, r)) in enumerate(zip(costs, results))
    vals = re(r.minimizer)(us') |> vec
    plot!(plt, us, vals, label="Cost $c", linewidth=2,
        color=wong_palette[1],
        linestyle=styles[i])
end

display(plt)