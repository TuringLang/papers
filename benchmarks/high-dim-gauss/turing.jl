using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

@model high_dim_gauss(D) = begin
    m ~ Multi(Normal(0, 1), D)
end

model = high_dim_gauss(data["D"])

alg = HMC(0.1, 4)

include("../infer_turing.jl")

;
