using KernelMachines: DiscreteMachine, radialkernel
using Test
using FiniteDiff: finite_difference_gradient
using Zygote: gradient

@testset "DiscreteMachine" begin
    augs = (rand(2, 5), rand(3, 5), rand(2, 5))
    data = rand(5, 10)
    input = rand(5, 50)
    css = (rand(3, 10), rand(2, 10))
    kernel = radialkernel
    augmenter = function (t)
        res = map(v -> v*t, augs)
        cost = sum(augs) do aug
            sum(abs2, aug)
        end
        return res, cost
    end
    dm = DiscreteMachine(augmenter, data, kernel, css)
    g_auto = gradient(input) do val
        r, n = dm(val)
        return sum(r) + n
    end
    g_num = finite_difference_gradient(input) do val
        r, n = dm(val)
        return sum(r) + n
    end
    @test isapprox(first(g_auto), g_num)
end