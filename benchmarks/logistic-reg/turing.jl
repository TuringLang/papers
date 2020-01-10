using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

@model logistic_reg(X, y) = begin
    D, N = size(X)
    w ~ Multi(Normal(0, 1), D)
    p = logistic.(X' * w)
    y ~ ArrayDist(Bernoulli.(p))
end

model = logistic_reg(data["X"], data["y"])

alg = HMC(0.1, 4)

include("../infer_turing.jl")

;
