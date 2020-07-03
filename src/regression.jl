mutable struct KernelRegression{K, I, O, C}
    kernel::K
    input::I
    output::O
    cs::Union{Nothing, O}
    cost::C
end

function KernelRegression(X::AbstractArray, Y::AbstractArray;
    kernel=radialkernel, cost=0)

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
