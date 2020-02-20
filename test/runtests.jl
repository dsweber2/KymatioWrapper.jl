using DataFrames, PyCall
using KymatioWrapper
using Test
py"""
import torch
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio import Scattering1D
from kymatio import Scattering2D
from kymatio.scattering2d.filter_bank import filter_bank

torch.cuda.is_available()
"""



@testset "scattering.jl" begin
    useGpu = py"torch.cuda.is_available()"
    J=6; Q=16; N = 2^13-2045
    T = 2^ceil.(Int, log2(N))
    Scatter = py"Scattering1D($J,$T,$Q)"
    if useGpu
        Scatter.cuda()
    end
    t = (range(0,2π,length=N));
    x = zeros(N, 5, 3);
    for (ii, ω)=enumerate(floor.(Int, range(1,150,length=5))), jj=1:3
        x[:,ii,jj] = sin.(ω/jj .* t) + cos.(100*ω/jj .* t.^2)
    end
    pX = cat(x,zeros(T-N,5,3),dims=1); size(pX)
    if useGpu
        scX = Scatter.forward(py"torch.from_numpy($(permutedims(pX,(2,3,1)))).cuda()")
    else
        scX = Scatter.forward(py"torch.from_numpy($(permutedims(pX,(2,3,1))))")
    end
    res = scX.cpu().numpy()
    s = Scattering(N, Q=16,useGpu=useGpu)
    s.scatter.shape
    sx = s(x)
    size(sx)
    size(permutedims(res, (4,3,1,2)))
    @test sx ≈ permutedims(res, (4,3,1,2))
    # Write your own tests here.
    s.scatter.shape
    s = Scattering((26,26),useGpu=useGpu)
    N = (26,26); J=2
    x = randn(N..., 4, 1); size(x)
    w= cat(x, zeros(6,26,4,1),dims=1); size(w)
    wx = cat(w,zeros(32,6,4,1),dims=2); size(wx)
    if useGpu
        Scatter = py"Scattering2D(2,shape=(32,32), L=8).cuda()"
        res = Scatter.forward(py"torch.from_numpy($(permutedims(wx,(3,4,1,2)))).cuda()")
    else
        Scatter = py"Scattering2D(2,shape=(32,32), L=8)"
        res = Scatter.forward(py"torch.from_numpy($(permutedims(wx,(3,4,1,2)))).contiguous()")
    end

    res = res.cpu().numpy()
    @test s(x) ≈ permutedims(res, (5,4,3,1:2...))
end


@testset "filter_bank" begin
    N = 450
    nψ₁ = 23
    nψ₂ = 6
    ψ₁,ψ₂,ϕ₃,params = filter_bank(N)
    @test size(ψ₁) == (2^ceil(log2(N)), nψ₁)
    @test size(ψ₂) == (2^ceil(log2(N)), nψ₂)
    @test size(ϕ₃) == (2^ceil(log2(N)),)
    @test size(params[1])==(nψ₁,3)
    @test size(params[2])==(nψ₂,3)
    N = (25,25); J=2; L = 8
    JL = J*L+1
    ψ,params, ϕₙ = filter_bank(N)
    @test size(ψ)==(N...,JL)
    @test size(params) == (JL, 2)
    @test length(ϕₙ)==J
    @test size(ϕₙ[1]) ==N
    @test size(ϕₙ[2]) == N .>>1
end
