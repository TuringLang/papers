using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

include("model.jl")

model = get_model(data["D"])

alg = HMC(0.1, 4)
n_samples = 2_000

include("../infer.jl")

;
