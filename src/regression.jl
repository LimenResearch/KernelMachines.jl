mutable struct KernelRegression{K, I, O, C}
    kernel::K
    input::I
    output::O
    cs::Union{Nothing, O}
    cost::C
end

"""
    KernelRegression(X::AbstractArray, Y::AbstractArray;
        kernel=gaussiankernel)

Return a `KernelRegression` object, to perform Kernel Ridge regression.
`X` and `Y` are training input and output. `Y` can be either a vector or a matrix.
`kernel` is required to have a method `kernel(u, v)`, where `u` and `v` are matrices
(datapoints are columns).
See [`additivegaussiankernel`](@ref) for an example. `cost` is used to regularize.

Use `fit!(kr::KernelRegression)` to fit the `KernelRegression` object, and
`predict(kr::KernelRegression, input)` to get the model prediction on a dataset.
If no `input` is given, it will default to the training data.
"""
function KernelRegression(X::AbstractArray, Y::AbstractArray;
    kernel=gaussiankernel, cost=0)

    return KernelRegression(kernel, X, Y, nothing, cost)
end

function fit!(kr::KernelRegression)

    input, output, kernel, cost = kr.input, kr.output, kr.kernel, kr.cost
    m = kernel(input', input')
    m += cost * size(input, 1) * I

    kr.cs = m \ output
    return kr
end

function predict(kr::KernelRegression, input=kr.input)
    return kr.kernel(input', kr.input') * kr.cs
end
