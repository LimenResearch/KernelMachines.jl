# # Noisy nonlinear regression
#
# First, let us load the relevant packages and define our problem.
# We want to interpolate a noisy version of `sin(x²/2)`.

using KernelMachines, Statistics, Plots, Random

func(x) = sin(x^2 / 2)
Random.seed!(1234) # for reproducibility
xs = 8 .* rand(100)
ys = @. func(xs) + rand()
scatter(xs, ys, color="black", xlabel="", ylabel="", primary=false)

# Now, let us train a Kernel Machine on the problem. We use the additive Gaussian
# kernel. Other than the kernel, we can choose the dimensionality of the hidden
# spaces, in this case, `(3, 3, 3)`, and the regularization coefficient, `cost=0.0005`.

krm = KernelMachineRegression(
    xs, ys;
    kernel=additivegaussiankernel,
    dims=(3, 3, 3), cost=0.0005
)
fit!(krm)
us = range(extrema(xs)...; step = 0.01)
pred_krm = predict(krm, us)
mean(abs2, pred_krm .- func.(us) .- 0.5)

# Let us compare with Kernel Ridge regression. Other than the kernel, we can choose the
# regularization coefficient, `cost=0.0005`.

kr = KernelRegression(xs, ys; cost=0.0005, kernel=gaussiankernel)
fit!(kr)
pred_kr = predict(kr, us)
mean(abs2, pred_kr .- func.(us) .- 0.5)

# We can then visualize the results. As the Gaussian kernel has only a fixed
# resolution, it works well on the part of the data of comparable spatial resolution,
# but is unable to fit the part at higher frequency.

theme(:wong)

plt = scatter(xs, ys, color="black", xlabel="", ylabel="", primary=false,
    legend=:bottomleft)

plot!(plt, us, pred_krm, label="KM", linewidth=2)
plot!(plt, us, pred_kr, label="KR", linewidth=2)
plot!(plt, us, func.(us) .+ 0.5, label="Truth", linewidth=2, color="black")
plt

# Note that the fitting procedure of `KernelMachineRegression` is done via
# [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). One can therefore
# pass both different optimization mathods and different options, such as
# number of iterations, or required accuracy. The default method is
# `ConjugateGradient`.
#
# !!! note
#
#     The API to pass optimization method and options to Optim.jl may change
#     in the future.
# 
# Here, we try a different optimization method (`GradientDescent`) and
# different options (maximum number of iterations and required relative
# precision).
#

using Optim: Options, GradientDescent, ConjugateGradient

krm_descent = KernelMachineRegression(
    xs, ys;
    kernel=additivegaussiankernel,
    dims=(3, 3, 3), cost=0.0005
)
krm_cg_approx = KernelMachineRegression(
    xs, ys;
    kernel=additivegaussiankernel,
    dims=(3, 3, 3), cost=0.0005
)
fit!(krm_descent, GradientDescent(), Options(iterations=1000))
fit!(krm_cg_approx, ConjugateGradient(), Options(f_reltol=1e-3))
pred_krm_descent = predict(krm_descent, us)
pred_krm_cg_approx = predict(krm_cg_approx, us);

# In this simple test, `ConjugateGradient` with approximate convergence
# conditions is significantly faster and gives a reasonably good solution. 

plt = scatter(xs, ys, color="black", xlabel="", ylabel="", primary=false,
    legend=:bottomleft)

plot!(plt, us, pred_krm, label="CG", linewidth=2)
plot!(plt, us, pred_krm_descent, label="Descent", linewidth=2)
plot!(plt, us, pred_krm_cg_approx, label="CG approx", linewidth=2)
plot!(plt, us, func.(us) .+ 0.5, label="Truth", linewidth=2, color="black")
plt