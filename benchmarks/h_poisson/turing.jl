using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

import Distributions: logpdf, DiscreteUnivariateDistribution
using SpecialFunctions: loggamma

struct LogPoisson{T<:Real} <: DiscreteUnivariateDistribution
    alpha::T
end

function logpdf(lp::LogPoisson, k::Int)
    return k * lp.alpha - exp(lp.alpha) - loggamma(k + 1)
end

@model h_poisson(y, x, idx, N, Ns) = begin
    a0 ~ Normal(0, 10)
    a1 ~ Normal(0, 1)
    a0_sig ~ truncated(Cauchy(0, 1), 0, Inf)
    a0s ~ Multi(Normal(0, a0_sig), Ns)    # FIXME: this is broken with Tracker
    alpha = a0 .+ a0s[idx] .+ a1 * x
    y ~ ArrayDist(LogPoisson.(alpha))
end

model = h_poisson(data["y"], data["x"], data["idx"], data["N"], data["Ns"])

step_size = 0.01
n_steps = 4

include("../infer_turing.jl")

;
