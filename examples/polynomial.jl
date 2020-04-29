using KernelNetworks
using LinearAlgebra, Statistics
using Flux: params
using Flux

using Zygote: gradient
using Random: seed!
seed!(0)

f(x, y) = (2x-1)^2 + 2y + x * y - 3
f((x, y)) = f(x, y)

##

model_kernel = Chain(Linear(2, 7), KernelNetwork((2, 3, 2), 6), Linear(7, 1))
model_perceptron = Chain(Dense(2, 5, relu), Dense(5, 5, relu), Dense(5, 5, relu), Dense(5, 1))

model_kernel(rand(Float32, 2, 10))
model_perceptron(rand(Float32, 2, 10))

N = 100
inputs = rand(Float32, 2, N)
outputs = map(f, eachcol(inputs)) |> permutedims
inputs′ = rand(Float32, 2, N)
outputs′ = map(f, eachcol(inputs′)) |> permutedims

function train!(model, inputs, outputs, inputs′, outputs′)
    ps = params(model)
    opt = ADAM(0.0002)
    for i in 1:N
        gs = gradient(ps) do
            sum(abs2, model(inputs) - outputs)
        end
        Flux.update!(opt, ps, gs)
    end
    loss_train = mean(abs2, model(inputs) - outputs)
    loss_test = mean(abs2, model(inputs′) - outputs′)
    return (loss_train, loss_test)
end

loss_kernel = [train!(model_kernel, inputs, outputs, inputs′, outputs′) for i in 1:3000]
loss_perceptron = [train!(model_perceptron, inputs, outputs, inputs′, outputs′) for i in 1:3000]

##

# using BSON
# BSON.@save "saved/polynomial.bson" loss_kernel loss_perceptron model_kernel model_perceptron 
# BSON.@load "saved/polynomial.bson" loss_kernel loss_perceptron model_kernel model_perceptron 

##

using Plots
using Plots.Measures

theme(:wong)
default(legendfont=14, tickfont=14, guidefont=14, titlefont=18, size=(800, 600))

plot([last.(loss_kernel) last.(loss_perceptron)], label = ["Kernel Network" "Multilayer Perceptron"], yscale=:log10,
    xlabel = "Epoch", ylabel = "Loss", linewidth=2, bottom_margin=5mm)

plot!([first.(loss_kernel) first.(loss_perceptron)], color = permutedims(Plots.palette(:wong)[1:2]), yscale=:log10,
    label = "", linestyle = :dash, linewidth=2)

##

plt1 = heatmap(0:0.05:1, 0:0.05:1, f, clims = (-3, 1),
    title="Truth", colorbar = false, axisratio = 1)
plt2 = heatmap(0:0.05:1, 0:0.05:1, only∘model_kernel∘vcat, clims = (-3, 1),
    title = "Kernel Network", colorbar = false, axisratio = 1)
plt3 = heatmap(0:0.05:1, 0:0.05:1, only∘model_perceptron∘vcat, clims = (-3, 1),
    title = "Multilayer Perceptron", colorbar = false, axisratio = 1)

plot(plt1, plt2, plt3, size = (1200, 400),
    layout = (1, 3), frame = :none, top_margin=5mm)
