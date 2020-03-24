using Random: seed!
seed!(1)

include("data.jl")

data = get_data(500)

using Turing

Turing.setadbackend(:reverse_diff)

@model sto_volatility(y) = begin
    T = length(y)
    
    ϕ ~ Uniform(-1, 1)
    σ ~ truncated(Cauchy(0, 5), 0, Inf)
    μ ~ Cauchy(0, 10)

    h_std ~ MvNormal(T, 1.0)
    h = σ .* h_std
    h[1] /= sqrt(1 - ϕ^2)
    h .+= μ
    for t in 2:T
        h[t] += ϕ * (h[t-1] - μ)
    end

    y ~ MvNormal(exp.(h ./ 2))
end

model = sto_volatility(data["y"])

step_size = 0.0002
n_steps = 4

include("../infer_turing.jl")

;
