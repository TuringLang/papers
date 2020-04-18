using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data(500)

using Turing

@model sto_volatility(y, ::Type{Tv}=Vector{Float64}) where {Tv} = begin
    T = length(y)

    μ ~ Cauchy(0, 10)
    ϕ ~ Uniform(-1, 1)
    σ ~ truncated(Cauchy(0, 5), 0, Inf)

    h = Tv(undef, T)
    h[1] ~ Normal(μ, σ / sqrt(1 - ϕ^2))
    y[1] ~ Normal(0, exp(h[1] / 2))
    for t in 2:T
        h[t] ~ Normal(μ + ϕ * (h[t-1] - μ), σ)
        y[t] ~ Normal(0, exp(h[t] / 2))
    end
end

model = sto_volatility(data["y"])

step_size = 0.0002
n_steps = 4
test_zygote = false
test_tracker = true

include("../infer_turing.jl")

;
