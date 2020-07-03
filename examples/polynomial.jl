using KernelMachines
using StatsBase, Plots
theme(:wong)

f(x, y) = (2x-1)^2 + 2y + x * y - 3
f((x, y)) = f(x, y)

N_train = 6
rg_train = range(0, 1, length=N_train)
X = hcat(repeat(rg_train, inner=N_train), repeat(rg_train, outer=N_train))
Y = map(f, eachrow(X))

krm = KernelMachineRegression(X, Y;
    dims=(3, 3, 3), kernel=multiplicativeradialkernel)
fit!(krm)

kr = KernelRegression(X, Y)
fit!(kr)

##

rg = 0:0.01:1
flat = hcat(repeat(rg, inner=N), repeat(rg, outer=N))
N = length(rg)
truth = map(f, eachrow(flat))
surface(rg, rg, reshape(truth, N, N), clims=(-3, 1))

##

pred_krm = predict(krm, flat)
surface(rg, rg, reshape(pred_krm, N, N), clims=(-3, 1))

@show mean(abs2, pred_krm - truth)

##

pred_kr = predict(kr, flat)
surface(rg, rg, reshape(pred_kr, N, N), clims=(-3, 1))

@show mean(abs2, pred_kr - truth)
