using KernelMachines
using StatsBase, Plots
theme(:wong)

f(x, y) = (2x-1)^2 + 2y + x * y - 3
f((x, y)) = f(x, y)

X = rand(100, 2)
Y = map(f, eachrow(X))

kr = KernelRegression(X, Y, dims=(3, 3, 3))
fit!(kr)

rg = 0:0.01:1
surface(rg, rg, f, clims=(-3, 1))

N = length(rg)
flat = hcat(repeat(rg, inner=N), repeat(rg, outer=N))
f_pred = predict(kr, flat)
surface(rg, rg, reshape(f_pred, N, N), clims=(-3, 1))