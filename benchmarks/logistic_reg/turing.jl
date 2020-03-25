using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing
using Turing.Core: filldist, arraydist

@model logistic_reg(X, y) = begin
    D, N = size(X)
    w ~ filldist(Normal(0, 1), D)
    p = logistic.(X' * w)
    y ~ arraydist(Bernoulli.(p))
end

model = logistic_reg(data["X"], data["y"])

step_size = 0.1
n_steps = 4

include("../infer_turing.jl")

;
