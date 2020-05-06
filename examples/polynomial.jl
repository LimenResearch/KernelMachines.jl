using KernelMachines
using LinearAlgebra, Statistics
using Flux: params
using Flux

using Zygote: gradient
using Random: seed!
seed!(0)

f(x, y) = (2x-1)^2 + 2y + x * y - 3
f((x, y)) = f(x, y)

model_kernel = Chain(Linear(2, 12), KernelMachine((3, 3, 3, 3), 30), Linear(12, 1))
model_perceptron = Chain(Dense(2, 8, relu), Dense(8, 16, relu), Dense(16, 16, relu), Dense(16, 8, relu), Dense(8, 1))

model_kernel(rand(Float32, 2, 10))
model_perceptron(rand(Float32, 2, 10))

N = 100
inputs = rand(Float32, 2, N)
outputs = map(f, eachcol(inputs)) |> permutedims
inputs′ = rand(Float32, 2, N)
outputs′ = map(f, eachcol(inputs′)) |> permutedims

_dot(m1::Dense, m2::Dense) = dot(m1.W, m2.W) + dot(m1.b, m2.b)
_dot(a, b) = dot(a, b)

function train!(model, inputs, outputs, inputs′, outputs′)
    ps = params(model)
    opt = ADAM(0.002)
    for i in 1:N
        gs = gradient(ps) do
            mean(abs2, model(inputs) - outputs) + 1f-2 * sum(_dot(l, l) for l in model.layers)
        end
        Flux.update!(opt, ps, gs)
    end
    @show loss_train = mean(abs2, model(inputs) - outputs)
    @show loss_test = mean(abs2, model(inputs′) - outputs′)
    return (loss_train, loss_test)
end

loss_kernel = [train!(model_kernel, inputs, outputs, inputs′, outputs′) for i in 1:50]
loss_perceptron = [train!(model_perceptron, inputs, outputs, inputs′, outputs′) for i in 1:50]

##

using Plots
using Plots.Measures
using PlotThemes: wong_palette

theme(:wong)
default(legendfont=14, tickfont=14, guidefont=14, titlefont=18, size=(800, 600))

plot([last.(loss_kernel) last.(loss_perceptron)], label = ["Kernel Machine" "Multilayer Perceptron"],
    xlabel = "Epoch", ylabel = "MSE", linewidth=2, bottom_margin=5mm, ylims=(0, 0.05))

plot!([first.(loss_kernel) first.(loss_perceptron)], color = permutedims(wong_palette[1:2]),
    label = "", linestyle = :dash, linewidth=2)

##

rg = 0:0.02:1

plt1 = surface(rg, rg, f, clims = (-3, 1),
    title="Truth", colorbar = false, axisratio = 1)
plt2 = surface(rg, rg, only∘model_kernel∘vcat, clims = (-3, 1),
    title = "Kernel Machine", colorbar = false, axisratio = 1)
plt3 = surface(rg, rg, only∘model_perceptron∘vcat, clims = (-3, 1),
    title = "Multilayer Perceptron", colorbar = false, axisratio = 1)

plot(plt1, plt2, plt3, size = (1200, 400),
    layout = (1, 3), frame = :none, top_margin=5mm, ticks=false)