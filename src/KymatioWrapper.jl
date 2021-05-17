module KymatioWrapper

using DataFrames
using Interpolations
using PyCall

export filter_bank, scattering_filter_factory, Scattering, computeJ, numpyify

function __init__()
    py"""
    import torch
    from kymatio.scattering1d.filter_bank import scattering_filter_factory
    from kymatio import Scattering1D
    from kymatio import Scattering2D
    from kymatio.scattering2d.filter_bank import filter_bank

    torch.cuda.is_available()
    """
end


struct Scattering{D}
    scatter::PyObject
    useGpu::Bool
end

######################### various utils #########################
next2(N) = 2^ceil.(Int, log2(N))
function padTo(x, T;dims=ndims(x))
    szx = size(x)
    toFill = zeros(eltype(x), szx[1:dims - 1]..., T - szx[dims], szx[dims + 1:end]...)
    cat(x, toFill, dims=dims) # assumes the *last* dimension is the one you're using
end
function padTo(x, N, M; dims=ndims(x))
    w = padTo(x, N, dims=ndims(x) - 1)
    padTo(w, M, dims=ndims(x))
end
computeJ(N) = max(floor(Int, min(log2.(N)...) / 2), 3)
"""
Convert any torch pyobjects to numpy (and thus julia) arrays
"""
function numpyify(a::AbstractArray)
    A = copy(a)
    for d in A
        for x in d
            if length(x) == 2 && typeof(x[2]) <: PyObject
                if x[2].is_cuda
                    xp = x[2].clone().detach().cpu()
                else
                    xp = x
                end
                d[x[1]] = (xp.numpy())[:,1]
            end
        end
    end
    return A
end
function numpyify(d::Dict)
    D = copy(d)
    for x in D
        if length(x) == 2 && typeof(x[2]) <: PyObject
            if x[2].is_cuda
                xp = x[2].clone().detach().cpu()
            else
                xp = x[2]
            end
            D[x[1]] = (xp.numpy())[:,1]
        end
    end
    return D
end
################################################################


"""
    ψ₁, ψ₂, ϕ₃, σξ = filter_bank(N::Real, J=floor(log2(N)/2), Q=8)
    ψ, params, ϕₙ = filter_bank(N::Real, J=floor(log2(N)/2), Q=8)

Slightly different format than the native one. Note that it rounds your length
    up to the next power of 2. ψ₁ is (length×nframes), and starts with the
    averaging (phi_f[0] in the original), and increases in frequency, as in
    Wavelets.jl, which is backwards from the native version. Same for ψ₂ with
    phi_f[1]. The final averaging stands alone in ϕ₃, and the parameters are in
    the array of dataframes σξ.
"""

function filter_bank(N::Real, J::Real=computeJ(N), Q::Real=8)
    T = ceil(Int, log2(N))
    phi_f, psi1_f, psi2_f, _ = py"scattering_filter_factory($T,$J,$Q)";
    ψ₁, params1 = extractDictionaries(phi_f, psi1_f, 0)
    ψ₂, params2 = extractDictionaries(phi_f, psi2_f, 1)
    if 2 in keys(phi_f)
        ϕ₃ = upInterp(phi_f[2], 2^2)
    else
        ϕ₃ = zeros(T)
    end
    σξ = [DataFrame(params1'), DataFrame(params2')]
    [rename!(x, [:σ,:j,:ξ]) for x in σξ]
    return (ψ₁, ψ₂, ϕ₃, σξ)
end

function filter_bank(ky::Scattering{1})
    sc = ky.scatter
    phi_f, psi1_f, psi2_f = (numpyify(sc.phi_f), numpyify(sc.psi1_f),
                             numpyify(sc.psi2_f))
    ψ₁, params1 = extractDictionaries(phi_f, psi1_f, 0)
    ψ₂, params2 = extractDictionaries(phi_f, psi2_f, 1)
    if 2 in keys(phi_f)
        ϕ₃ = upInterp(phi_f[2], 2^2)
    else
        ϕ₃ = zeros(2^(sc.J_pad))
    end
    σξ = [DataFrame(params1'), DataFrame(params2')]
    [rename!(x, [:σ,:j,:ξ]) for x in σξ]
    return (ψ₁, ψ₂, ϕ₃, σξ)
end
function filter_bank(ky::Scattering{2})
    N = 641
    filter_bank(N, ky.scatter.J, ky.scatter.L)
end


function filter_bank(N, J=computeJ(N), L=8)
    filters_set = py"filter_bank($(N[1]),$(N[2]),$J,L=$L)"
    nfilters = length(filters_set["psi"]) + 1
    ψ = zeros(N..., nfilters)
    params = -1 .* ones(nfilters, 2)
    ψ[:,:,1] = py"($(filters_set)['phi'][0][..., 0])" .* (py"($(filters_set)['phi'][0][..., 0])")'
    params[1,1] = filters_set["psi"][1]["j"]
    params[1,2] = NaN
    for ii = 1:(nfilters - 1)
        ψ[:,:,nfilters - ii + 1] = py"($(filters_set)['psi'][$(ii-1)][0][..., 0])" .* (py"($(filters_set)['psi'][$(ii-1)][0][..., 0])")'
        params[nfilters - ii + 1, 1] = filters_set["psi"][ii]["j"]
        params[nfilters - ii + 1, 2] = filters_set["psi"][ii]["theta"]
    end
    params = DataFrame(params)
    rename!(params, [:j,:θ])
    ϕₙ = Dict()
    for ii = 1:J
        ϕₙ[ii] = py"($(filters_set)['phi'][$ii-1][..., 0])"
    end
    return ψ, params, ϕₙ
end

scattering_filter_factory(N,J,Q) = filter_bank(N, J, Q)

function extractDictionaries(phi_f, psi_f, m)
    nfreq = length(psi_f) + 1
    ψ = zeros(length(psi_f[1][0]), nfreq)
    params = zeros(3, nfreq)
    ψ[:,1] = upInterp(phi_f[m], 2^m)
    params[1,1] = phi_f["sigma"]
    params[2,1] = phi_f["j"]
    params[3,1] = phi_f["xi"]
    for (ii, ψi) in enumerate(psi_f)
        ψ[:, nfreq - ii + 1] .= ψi[0]
        params[1,nfreq - ii + 1] = ψi["sigma"]
        params[2,nfreq - ii + 1] = ψi["j"]
        params[3,nfreq - ii + 1] = ψi["xi"]
    end
    return (ψ, params)
end

"""
    upInterp(ϕ,k)
given a vector of values, do a periodic cubic interpolation with k times as
    many entries
"""
function upInterp(ϕ::AbstractArray{T,1}, k) where T
    itr = interpolate(ϕ, BSpline(Cubic(Periodic(OnGrid()))))
     x = range(1, length(ϕ), length=k * length(ϕ))
    return itr(x)
end
upInterp(ϕ::AbstractArray{T,2}, k) where T = upInterp(ϕ[:,1], k + 1)


# 1D version
function Scattering(N::Int64; J=computeJ(N), Q=8,
                    max_order=2, average=true, oversampling=0,
                    vectorize=true, useGpu=true)
    if J < 2.13
        @warn "silently does the wrong thing if J<2.13."
    end
    T = N
    scatterer = py"Scattering1D($J, $T, $Q, $(max_order), $(average), $(oversampling), $(vectorize),backend='torch')"
    if useGpu
        cuVersion = scatterer.cuda()
    end
    Scattering{1}(scatterer, useGpu)
end

"""
    compute_meta_scattering(J::Int,Q,max_order=2)
    compute_meta_scattering(s::Scattering{1})
what's on the tin.
"""
function compute_meta_scattering(J::Int, Q, max_order=2)
    dictMeta = py"Scattering1D.compute_meta_scattering($(J), $(Q), $(max_order))"
    dictMeta
    meta = DataFrame(j1=Float64[], j2=Float64[], key=Tuple{Vararg{Int64}}[],
                    σ1=Float64[], σ2=Float64[], order=Int[], ξ1=Float64[],
                    ξ2=Float64[], n1=Int[], n2=Int[])
    σ = dictMeta["sigma"].numpy()
    σ = DataFrame(σ₁=σ[:,1], σ₂=σ[:,2])
    order = DataFrame(order=dictMeta["order"].numpy())
    n = dictMeta["n"].numpy()
    n = DataFrame(n1=n[:,1], n2=n[:,2])
    j = dictMeta["j"].numpy()
    j = DataFrame(j1=j[:,1], j2=j[:,2])
    key = DataFrame(key=dictMeta["key"])
    return hcat(σ, n, j, key, order)
end
function compute_meta_scattering(s::Scattering{1})
    compute_meta_scattering(s.scatter.J, s.scatter.Q, s.scatter.max_order)
end

function (s::Scattering{1})(x)
    x = Float32.(permutedims(x, (2:ndims(x)..., 1)))
    flipped = true
    if s.useGpu
        res = s.scatter.forward(py"torch.from_numpy($x).cuda().contiguous()")
    else
        res = s.scatter.forward(py"torch.from_numpy($x).contiguous()")
    end
    res = res.cpu().numpy()
    n = ndims(res)
    if flipped
        n = ndims(res)
        return permutedims(res, (n, n - 1, 1:(n - 2)...))
    else
        return res
    end
end
################################################################
# 2D version
################################################################



function Scattering(N::Tuple{Int64,Int64}; J=computeJ(N),
                    L=8, max_order=2, useGpu=true)
    # if you're using Julia size conventions, switch that around
    T = next2.(N)
    scatterer = py"Scattering2D($J, shape=$T, L=$L, max_order=$max_order)"
    if useGpu
        scatterer.cuda()
    end
    Scattering{2}(scatterer, useGpu)
end



function (s::Scattering{2})(x)
    @assert maximum(s.scatter.shape .% 2 .== 0) "s has shape = $(s.scatter.shape)"
    flipped = false
    if next2(size(x, 1)) == s.scatter.shape[1]
        x = Float32.(permutedims(x, ((3:ndims(x))..., 1, 2)))
        flipped = true
    end
    x = padTo(x, s.scatter.shape...)
    torchX =
    if s.useGpu
        res = s.scatter.forward(py"torch.from_numpy($x).cuda()")
    else
        res = s.scatter.forward(py"torch.from_numpy($x).contiguous()")
    end

    res = res.cpu().numpy()
    if flipped
        n = ndims(res)
        return permutedims(res, (n, n - 1, n - 2, 1:(n - 3)...))
    else
        return res
    end
end







end
