using KernelMachines
using StatsBase, Plots

xs = 20 .* rand(100) .- 10
ys = sin.(xs) .+ rand(100)

us = range(extrema(xs)...; step = 0.1)

krm = KernelMachineRegression(xs, ys;
    kernel=additiveradialkernel,
    dims=(1, 2, 2, 2, 1), cost=0.01)
fit!(krm)

pred_krm = predict(krm, us)
@show mean(abs2, pred_krm .- sin.(us) .- 0.5)

kr = KernelRegression(xs, ys; cost=0.01)
fit!(kr)

pred_kr = predict(kr, us)
@show mean(abs2, pred_kr .- sin.(us) .- 0.5)

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

plot!(plt, us, pred_krm, label="KM", linewidth=2)
plot!(plt, us, pred_kr, label="KR", linewidth=2)
plot!(plt, us, sin.(us) .+ 0.5, label="Truth", linewidth=2, color="black")