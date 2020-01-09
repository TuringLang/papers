using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

@model high_dim_gauss(D) = begin
    m ~ Multi(Normal(), D)
end

model = high_dim_gauss(data["D"])

include("../infer_turing.jl")

;
