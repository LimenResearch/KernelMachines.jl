using KernelMachines
using Plots, CSV, StatsBase, Random

tmp = download("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df = copy(CSV.read(tmp))
passengers = zscore(df.Passengers)

by_year = reshape(passengers, 12, 12)
X = by_year[:, 1:11]
Y = by_year[:, 2:12]

# Must train on extrema, as kernel methods cannot extrapolate
train = [1, 2, 4, 5, 7, 8, 10, 11]
test = [3, 6, 9]

X_train = X[:, train]
Y_train = Y[:, train]

X_test = X[:, test]
Y_test = Y[:, test]

##

Aug_train = vcat(permutedims(train ./ std(1:11)), X_train)
Aug_test = vcat(permutedims(test ./ std(1:11)), X_test)

kr = KernelMachines.fit(Aug_train, Y_train, dims = (1, ntuple(_ -> 1, 24)...), cost = 0.1)

res_train = KernelMachines.predict(kr, Aug_train)
res_test = KernelMachines.predict(kr, Aug_test)

##

using PlotThemes: wong_palette
theme(:wong)
default(legendfont=14, tickfont=14, guidefont=14, size=(800, 600))

plt = plot(passengers, color="black", legend=false, linewidth=2)
train_idxs = map(i -> div(i-1, 12) + 1 in train, 1:132)
test_idxs = map(!, train_idxs)
res = zeros(132)
res[train_idxs] .= vec(res_train)
res[test_idxs] .= vec(res_test)
color = map(t -> wong_palette[t+1], train_idxs)
plot!(plt, 13:144, vec(res), color=color, linewidth=2)