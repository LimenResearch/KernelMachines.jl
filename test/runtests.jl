using KernelMachines: KernelMachine
using Test
using FiniteDiff: finite_difference_gradient
using Zygote: gradient

@testset "KernelMachine" begin
    dims = (2, 3, 2)
    data = rand(5, 10)
    input = rand(5, 50)
    dm = KernelMachine(data; dims=dims)
    g_auto = gradient(input) do input
        r, n = dm(input)
        return sum(r) + n
    end
    g_num = finite_difference_gradient(input) do input
        r, n = dm(input)
        return sum(r) + n
    end
    @test isapprox(first(g_auto), g_num)
end