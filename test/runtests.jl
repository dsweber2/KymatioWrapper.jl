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
    sx = s(x)
    size(sx)
    size(permutedims(res, (4,3,1,2)))
    @test sx ≈ permutedims(res, (4,3,1,2))
    # Write your own tests here.

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


N=M = 32; J=3; L=8
filters_set = py"filter_bank($M,$M,4,L=$L)"
ϕₙ = Dict()
filters_set[]
for ii=1:J
    ϕₙ[ii] = py"($(filters_set)['phi'][ii-1][..., 0])".numpy()
end
py"""
from kymatio.scattering2d.utils import fft2
"""
heatmap(py"($(filters_set)['psi'][1][0][..., 0])".numpy())
filters_set["psi"][1][0].numpy()
keys(filters_set["phi"])
keys(filters_set["psi"][2])
size(filters_set["psi"][1][0].numpy())
heatmap(filters_set["psi"][1][0].numpy()[:,:,1])
heatmap(filters_set["psi"][1][0].numpy()[:,:,2])
using Plots
filters_set["psi"][24]["j"]
size(filters_set["phi"][0])
N = (26,26)
N = (32, 32)
filter_bank(N,4)

function filter_bank(N,J=floor(minimum(log2.(N))/2),L=8)
    filters_set = py"filter_bank($(N[1]),$(N[2]),$J,L=$L)"
    nfilters = length(filters_set["psi"])+1
    ψ = zeros(N,M, nfilters)
    params = -1 .* ones(nfilters,2)
    ψ[:,:,1] = py"($(filters_set)['phi'][0][..., 0])".numpy()
    params[1,1] = filters_set["psi"][1]["j"]
    params[1,2] = NaN
    for ii=1:(nfilters-1)
        ψ[:,:,nfilters-ii+1] = py"($(filters_set)['psi'][$(ii-1)][0][..., 0])".numpy()
        params[nfilters-ii+1, 1] = filters_set["psi"][ii]["j"]
        params[nfilters-ii+1, 2] = filters_set["psi"][ii]["theta"]
    end
    params = DataFrame(params)
    names!(params, [:j,:θ])
    ϕ₂ = py"($(filters_set)['phi'][1][..., 0])".numpy()
    ϕ₃ = py"($(filters_set)['phi'][2][..., 0])".numpy()
    return ψ,params, ϕ₂, ϕ₃
end
ψ₁,ψ₂,ϕ₃,σξ = scattering_filter_factory(234)
size(ψ₁)
plot(heatmap(ψ₂[1:128,:]), heatmap(ψ₁[1:128,:]), heatmap(daughters),layout=(3,1))

using Wavelets
daughters,ω = computeWavelets(234, wavelet(WT.Morlet()))
size(daughters)

using Shearlab

