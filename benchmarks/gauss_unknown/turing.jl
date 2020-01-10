using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

@model gauss_unknown(y) = begin
    N = length(y)
    m ~ Normal(0, 1)
    s ~ truncated(Cauchy(0, 5), 0, Inf)
    y ~ Multi(Normal(m, s), N)
end

model = gauss_unknown(data["y"])

alg = HMC(0.01, 4)

include("../infer_turing.jl")

;
