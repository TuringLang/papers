using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

include("model.jl")

Turing.setadbackend(:reverse_diff)

model = get_model(data["D"])

chain = sample(model, HMC(0.1, 4), 2_000, progress_style=:plain)
