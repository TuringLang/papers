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
    v::Vector{T}
    m::Vector{V}
end

function Distributions.logpdf(d::IMM{<:Real,<:Real}, x::Real)
    lv = log.(d.v)
    lp = logpdf.(Normal.(d.m), x) .+ lv
    return logsumexp(lp) - sum(lv)
end

@model truncated_imm(y, K, alpha) = begin
    N = length(y)
    rpm = DirichletProcess(alpha)

    v ~ Multi(StickBreakingProcess(rpm), K)
    m ~ MvNormal(ones(K)*m0, s0)
    w = vcat(v[1], v[2:end] .* cumprod(1 .- v[1:end-1]))

    # This breaks for some reason...
    # println(v)

    y ~ Multi(IMM(w, m), N)
end

model = truncated_imm(data["y"], 10, 1.0)

step_size = 0.0001
n_steps = 4

include("../infer_turing.jl")

;
