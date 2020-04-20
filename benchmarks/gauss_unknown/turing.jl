using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Memoization, Turing
using Turing.Core: filldist

@model gauss_unknown(y) = begin
    N = length(y)
    m ~ Normal(0, 1)
    s ~ truncated(Cauchy(0, 5), 0, Inf)
    y ~ filldist(Normal(m, s), N)
end

model = gauss_unknown(data["y"])

step_size = 0.01
n_steps = 4
test_zygote = true
test_tracker = true

include("../infer_turing.jl")

;
