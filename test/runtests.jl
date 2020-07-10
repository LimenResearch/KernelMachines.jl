using KernelMachines
using Test
using FiniteDiff: finite_difference_gradient
using Zygote: gradient

@testset "utils" begin
    s = rand(10, 3)
    slices = KernelMachines.split_matrix(s, (2, 3, 5))
    @test length(slices) == 3
    @test slices[1] == s[1:2, :]
    @test slices[2] == s[3:5, :]
    @test slices[3] == s[6:10, :]
end

@testset "kernelmachine" begin
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