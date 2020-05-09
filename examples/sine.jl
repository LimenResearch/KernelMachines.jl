using Plots, KernelMachines
using Plots.Measures
using Flux, LinearAlgebra, Statistics
using Random: seed!
seed!(0)

xs = 20 .* rand(100) .- 10
ys = sin.(xs) .+ rand(100)

us = range(extrema(xs)...; step = 0.1)

inputs = permutedims(xs)
outputs = permutedims(ys)

kr = fit(KernelRegression, inputs, outputs, dims=(1, 2, 2, 2, 1), cost=0.005)

res = predict(kr, us')'
vs = res

##

theme(:wong)
default(legendfont=14, tickfont=14, guidefont=14, size=(800, 600))

plt = scatter(
    xs, ys,
    label = false, legend = :bottomright,
    color="black", primary=false,
    bottom_margin=5mm,
    xlabel = "x",
    ylabel="sine(x) + rand()",
    xlims=(-12.5, 12.5),
    ylims=(-1, 2)
    )

plot!(plt, us, vs, label="KM", linewidth=2)
plot!(plt, us, sin.(us) .+ 0.5, label="Truth", linewidth=2,
    color="black")
display(plt)
