using KernelNetworks
using Test
using FiniteDiff: finite_difference_gradient
using Zygote: gradient
using Flux: destructure, params, f64
using LinearAlgebra

@testset "layers" begin
    kl = KernelLayer(2 => 3, 10)
    v = rand(2)
    _f = v -> sum(last(kl(true, v)))
    gs_auto = only(gradient(_f, v))
    gs_num = finite_difference_gradient(_f, v)
    @test isapprox(gs_auto, gs_num, rtol=1e-6)
end

@testset "networks" begin
    kn = KernelNetwork((2, 3, 3, 2), 20)
    ps, re = destructure(kn)
    v = rand(10, 20)
    _f = ps -> sum(re(ps)(v))
    gs_auto = only(gradient(_f, ps))
    gs_num = finite_difference_gradient(_f, ps)
    @test isapprox(gs_auto, gs_num, rtol=1e-4)
end

@testset "scalar product" begin
    kl1 = KernelLayer(2 => 3, 10) |> f64
    kl2 = KernelLayer(2 => 3, 10) |> f64
    ker = KernelNetworks.radialkernel(kl1.xs′, transpose(kl2.xs′))
    coefs = transpose(kl1.cs) * kl2.cs
    rows1, rows2 = eachrow(kl1.cs), eachrow(kl2.cs)
    d = dot(kl1, kl2)
    d1 = sum(dot(row1, ker, row2) for (row1, row2) in zip(rows1, rows2))
    d2 = dot(transpose(kl1.cs), ker, transpose(kl2.cs))
    d3 = dot(coefs, ker)
    @test d ≈ d1
    @test d ≈ d2
    @test d ≈ d3
    # test gradient of dot product numerically
    gs = gradient(dot, kl1, kl2)
    xs1 = finite_difference_gradient(v -> dot(KernelLayer(v, kl1.cs), kl2), kl1.xs′)
    @test isapprox(gs[1].xs′, xs1, rtol=1e-4)
    cs1 = finite_difference_gradient(v -> dot(KernelLayer(kl1.xs′, v), kl2), kl1.cs)
    @test isapprox(gs[1].cs, cs1, rtol=1e-4)
    xs2 = finite_difference_gradient(v -> dot(KernelLayer(v, kl2.cs), kl1), kl2.xs′)
    @test isapprox(gs[2].xs′, xs2, rtol=1e-4)
    cs2 = finite_difference_gradient(v -> dot(KernelLayer(kl2.xs′, v), kl1), kl2.cs)
    @test isapprox(gs[2].xs′, xs2, rtol=1e-4)
end