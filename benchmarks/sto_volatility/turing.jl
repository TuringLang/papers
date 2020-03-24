using Random: seed!
seed!(1)

include("data.jl")

data = get_data(500)

using ReverseDiff, Memoization, Turing
using Turing.Core: arraydist

Turing.setadbackend(:reversediff)
Turing.setcache(true)

@model sto_volatility(y, ::Type{Tv}=Vector{Float64}) where {Tv} = begin
    T = length(y)
    
    ϕ ~ Uniform(-1, 1)
    σ ~ truncated(Cauchy(0, 5), 0, Inf)
    μ ~ Cauchy(0, 10)

    h = Tv(undef, T)
    h[1] ~ Normal(μ, σ / sqrt(1 - ϕ^2))
    for t in 2:T
        h[t] ~ Normal(μ + ϕ * (h[t-1] - μ), σ)
    end

    y ~ arraydist(Normal.(0, exp.(h / 2)))
end

model = sto_volatility(data["y"])

step_size = 0.0002
n_steps = 4

include("../infer_turing.jl")

;
