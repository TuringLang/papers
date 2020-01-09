using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

include("model.jl")

Turing.setadbackend(:reverse_diff)

model = get_model(data["D"])

alg = HMC(0.1, 4)
n_samples = 2_000

if "--benchmark" in ARGS
    times = []
    for i in 1:5
        t = @elapsed chain = sample(model, alg, n_samples, progress=false)
        push!(times, t)
    end
    println(times)
else
    chain = sample(model, alg, n_samples, progress_style=:plain)
end

;