using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

@model logistic_reg(X, y) = begin
    D, N = size(X)
    w ~ MvNormal(fill(0, D), 1)
    p = logistic.(X' * w)
    y .~ Bernoulli.(p)
end

model = logistic_reg(data["X"], data["y"])

include("../infer_turing.jl")

;
