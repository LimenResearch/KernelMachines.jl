using KernelMachines
using StatsBase, Plots

func(x) = sin(x^2 / 2)
xs = 8 .* rand(100)
ys = @. func(xs) + rand()

us = range(extrema(xs)...; step = 0.01)

krm = KernelMachineRegression(xs, ys;
    kernel=additiveradialkernel,
    dims=(3, 3, 3), cost=0.0005)
fit!(krm)

pred_krm = predict(krm, us)
@show mean(abs2, pred_krm .- func.(us) .- 0.5)

kr = KernelRegression(xs, ys; cost=0.0005)
fit!(kr)

pred_kr = predict(kr, us)
@show mean(abs2, pred_kr .- func.(us) .- 0.5)

##

theme(:wong)

plt = scatter(xs, ys, color="black", xlabel="", ylabel="", primary=false,
    legend=:bottomleft)

plot!(plt, us, pred_krm, label="KM", linewidth=2)
plot!(plt, us, pred_kr, label="KR", linewidth=2)
plot!(plt, us, func.(us) .+ 0.5, label="Truth", linewidth=2, color="black")