using KernelMachines
using StatsBase, Plots

xs = 20 .* rand(100) .- 10
ys = sin.(xs) .+ rand(100)

us = range(extrema(xs)...; step = 0.1)

kr = KernelRegression(xs, ys, dims=(1, 2, 2, 2, 1), cost=0.005)
fit!(kr)

res = predict(kr, us)

##

theme(:wong)

plt = scatter(
    xs, ys,
    label = false, legend = :bottomright,
    color="black", primary=false,
    xlabel = "x",
    ylabel="sine(x) + rand()",
    xlims=(-12.5, 12.5),
    ylims=(-1, 2)
    )

plot!(plt, us, res, label="KM", linewidth=2)
plot!(plt, us, sin.(us) .+ 0.5, label="Truth", linewidth=2, color="black")