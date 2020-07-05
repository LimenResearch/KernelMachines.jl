# # Fitting a polynomial with kernels
#
# First, let us load the relevant packages and define our problem.
# We want to interpolate a simple polynomial: `(2x-1)² + 2y + xy - 3`.

using KernelMachines, StatsBase, Plots

f(x, y) = (2x-1)^2 + 2y + x * y - 3
f((x, y)) = f(x, y)
rg = 0:0.01:1
N = length(rg)
flat = hcat(repeat(rg, inner=N), repeat(rg, outer=N))
truth = map(f, eachrow(flat))
theme(:wong)
surface(rg, rg, reshape(truth, N, N), clims=(-3, 1))

# Let us generate a `6 x 6` trainig grid.

N_train = 6
rg_train = range(0, 1, length=N_train)
X = hcat(repeat(rg_train, inner=N_train), repeat(rg_train, outer=N_train))
Y = map(f, eachrow(X));

# Now, let us train a Kernel Machine on the problem.

krm = KernelMachineRegression(X, Y;
    dims=(3, 3, 3), kernel=multiplicativegaussiankernel)
fit!(krm)

pred_krm = predict(krm, flat)
surface(rg, rg, reshape(pred_krm, N, N), clims=(-3, 1))

# In this problem, a simple Kernel Ridge regression also performs very well.

kr = KernelRegression(X, Y)
fit!(kr)
pred_kr = predict(kr, flat)
surface(rg, rg, reshape(pred_kr, N, N), clims=(-3, 1))
