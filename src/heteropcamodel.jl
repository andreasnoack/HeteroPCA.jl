using LinearAlgebra, Statistics
using StatsBase: CoefTable
import Base: size, show
import Statistics: mean, var
import StatsAPI: predict, r2
import LinearAlgebra: eigvals, eigvecs

"""
    HeteroPCAModel

Variant of PCA that jointly estimates the low‑rank common component **and**
heteroscedastic idiosyncratic noise variance.  Missing cells are allowed;
they are mean‑centered and then imputed with zero internally (i.e. the
"zero‑fill after demeaning" strategy proposed by Cai et al., 2019).

`proj` gives an **orthonormal** basis of the estimated factor space,
`prinvars` are the k largest singular values of the estimated low‑rank matrix,
and `noisevars` stores the final diagonal (noise) estimates.
"""
struct HeteroPCAModel{T<:Real}
    mean::Vector{T}      # d‑vector of variable means
    proj::Matrix{T}      # d×k orthonormal basis
    prinvars::Vector{T}      # length‑k singular values (≈ variance explained)
    tprinvar::T              # sum(prinvars)
    tvar::T              # total variance in data
    noisevars::Vector{T}      # length‑d heteroscedastic noise variances
    converged::Bool
    iterations::Int
end

# Short, PCA‑compatible helpers --------------------------------------------------

r2(M::HeteroPCAModel) = M.tprinvar / M.tvar
loadings(M::HeteroPCAModel) = sqrt.(principalvars(M))' .* projection(M)
noisevars(M::HeteroPCAModel) = M.noisevars

fullmean(d::Int, mv::AbstractVector{T}) where T = (isempty(mv) ? zeros(T, d) : mv)

## properties
"""
    size(M)

Returns a tuple with the dimensions of input (the dimension of the observation space)
and output (the dimension of the principal subspace).
"""
size(M::HeteroPCAModel) = size(M.proj)


"""
mean(M::HeteroPCAModel)

Returns the mean vector (of length `d`).
"""
mean(M::HeteroPCAModel) = fullmean(size(M.proj, 1), M.mean)

"""
    projection(M::HeteroPCAModel)

Returns the projection matrix (of size `(d, p)`). Each column of the projection matrix corresponds to a principal component.
The principal components are arranged in descending order of the corresponding variances.
"""
projection(M::HeteroPCAModel) = M.proj

"""
    eigvecs(M::HeteroPCAModel)

Get the eigenvalues of the PCA model `M`.
"""
eigvecs(M::HeteroPCAModel) = projection(M)

"""
    principalvars(M::HeteroPCAModel)

Returns the variances of principal components.
"""
principalvars(M::HeteroPCAModel) = M.prinvars
principalvar(M::HeteroPCAModel, i::Int) = M.prinvars[i]

"""
    eigvals(M::HeteroPCAModel)

Get the eigenvalues of the PCA model `M`.
"""
eigvals(M::HeteroPCAModel) = principalvars(M)

"""
    tprincipalvar(M::HeteroPCAModel)

Returns the total variance of principal components, which is equal to `sum(principalvars(M))`.
"""
tprincipalvar(M::HeteroPCAModel) = M.tprinvar

"""
    tresidualvar(M::HeteroPCAModel)

Returns the total residual variance.
"""
tresidualvar(M::HeteroPCAModel) = M.tvar - M.tprinvar

"""
    var(M::HeteroPCAModel)

Returns the total observation variance, which is equal to `tprincipalvar(M) + tresidualvar(M)`.
"""
var(M::HeteroPCAModel) = M.tvar

"""
    r2(M::HeteroPCAModel)
    principalratio(M::HeteroPCAModel)

Returns the ratio of variance preserved in the principal subspace, which is equal to `tprincipalvar(M) / var(M)`.
"""
const principalratio = r2

"""
    predict(M::HeteroPCAModel, x::AbstractVector; λ = 0.0)

Project a **single observation vector** `x` onto the estimated factor space,
handling missing entries by solving a weighted least‑squares problem on the
observed coordinates only.

`λ` adds a small ridge (Tikhonov) regularisation to keep the system well‑posed
when fewer than `rank` coordinates are observed.
"""
function predict(M::HeteroPCAModel, x::AbstractVector{T}; λ::Real=0.0) where {T}
    obs = map(!ismissing, x)
    U = projection(M)
    k = size(U, 2)

    if count(obs) == 0
        return fill(zero(T), k)               # no information at all
    end

    Uobs = @view U[obs, :]
    μobs = @view M.mean[obs]
    zobs = coalesce.(x[obs] - μobs, zero(T))

    G = Uobs' * Uobs
    if λ != 0.0
        G .+= λ * I(k)
    end
    return G \ (Uobs' * zobs)
end


"""
    predict(M::HeteroPCAModel, X::AbstractMatrix; λ = 0.0)

Column‑wise version: returns a `k × n` matrix whose *j*‑th column is
`predict(M, X[:, j]; λ = λ)`.
"""
function predict(M::HeteroPCAModel, X::AbstractMatrix{T}; λ::Real=0.0) where {T}
    k, n = size(projection(M), 2), size(X, 2)
    Z = Matrix{T}(undef, k, n)
    @inbounds for j in 1:n
        Z[:, j] = predict(M, @view(X[:, j]); λ=λ)
    end
    return Z
end

"""
    reconstruct(M::HeteroPCAModel, y)

Inverse transform from factor space back to the original *d*‑space.
"""
function reconstruct(M::HeteroPCAModel, y::AbstractVecOrMat{T}) where {T}
    return M.proj * y .+ M.mean
end

function show(io::IO, M::HeteroPCAModel)
    idim, odim = size(M)
    print(io, "HeteroPCAModel(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
end

function show(io::IO, ::MIME"text/plain", M::HeteroPCAModel)
    idim, odim = size(M)
    print(io, "HeteroPCAModel(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
    ldgs = loadings(M)
    rot = diag(ldgs' * ldgs)
    ldgs = ldgs[:, sortperm(rot, rev=true)]
    ldgs_signs = sign.(sum(ldgs, dims=1))
    replace!(ldgs_signs, 0 => 1)
    ldgs = ldgs * diagm(0 => ldgs_signs[:])
    print(io, "\n\nPattern matrix (unstandardized loadings):\n")
    cft = CoefTable(ldgs, string.("PC", 1:odim), string.("", 1:idim))
    show(io, MIME"text/plain"(), cft)
    print(io, "\n\n")
    print(io, "Importance of components:\n")
    λ = eigvals(M)
    prp = λ ./ var(M)
    prpv = λ ./ sum(λ)
    names = ["SS Loadings (Eigenvalues)",
        "Variance explained", "Cumulative variance",
        "Proportion explained", "Cumulative proportion"]
    cft = CoefTable(vcat(λ', prp', cumsum(prp)', prpv', cumsum(prpv)'),
        string.("PC", 1:odim), names)
    show(io, MIME"text/plain"(), cft)
end
