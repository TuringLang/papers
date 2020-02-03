using Random: seed!
using Turing.RandomMeasures
using StatsFuns
seed!(1)

include("data.jl")

m0 = 0
s0 = 10

data = get_data(m0 = m0, s0 = s0)

using Turing

Turing.setadbackend(:forward_diff)

# custom mixture model distribution
struct IMM{T<:Real,V<:Real} <: ContinuousUnivariateDistribution
    z::AbstractVector{T}
    m::Vector{V}
end

function Distributions.logpdf(d::IMM{<:Real,<:Real}, x::Vector{<:Real})
    return sum(logpdf.(Normal.(d.m[d.z]), x))
end

@model imm(y, alpha, ::Type{T}=Vector{Int}, ::Type{M}=Vector{Float64}) where {T,M} = begin
    N = length(y)
    rpm = DirichletProcess(alpha)

    z = T(undef, N)
    cluster_counts = T(undef, N)
    fill!(cluster_counts, 0)

    for i in 1:N
        z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
        cluster_counts[z[i]] += 1
    end

    K = sum(cluster_counts .> 0)

    # m ~ MvNormal(fill(m0, K), s0)
    m = M(undef, K)
    for k = 1:K
        m[k] ~ Normal(m0, s0)
    end

    y ~ IMM(z, m)
end

model = imm(data["y"], 1.0)

step_size = 0.01
n_steps = 4

alg = Gibbs(PG(20, :z), HMC(step_size, n_steps, :m))

# no Gibbs
alg = SMC(10)

#include("../infer_turing.jl")

;
