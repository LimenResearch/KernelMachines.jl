using KernelNetworks
using Test
using FiniteDiff: finite_difference_gradient
using Zygote: gradient
using Flux: destructure, params

@testset "layers" begin
    kl = KernelLayer(2 => 3, 10)
    v = rand(2)
    _f = v -> sum(last(kl(true, v)))
    gs_auto = only(gradient(_f, v))
    gs_num = finite_difference_gradient(_f, v)
    @test isapprox(gs_auto, gs_num, rtol=1e-6)
end

@testset "KernelNetworks.jl" begin
    kn = KernelNetwork((2, 3, 3, 2), 20)
    ps, re = destructure(kn)
    v = rand(10, 20)
    _f = ps -> sum(re(ps)(v))
    gs_auto = only(gradient(_f, ps))
    gs_num = finite_difference_gradient(_f, ps)
    @test isapprox(gs_auto, gs_num, rtol=1e-4)
end
