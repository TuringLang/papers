using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

include("model.jl")

Turing.setadbackend(:reverse_diff)

model = get_model(data["image"], data["label"], data["C"])

chain = sample(model, HMC(0.1, 4), 2_000, progress_style=:plain)

# Save result

m_data = chain[:m].value.data

m_bayes = mean(
    map(
        i -> reconstruct(pca, Matrix{Float64}(reshape(m_data[i,:,1], D_pca, 10))), 
        1_000:100:2_000
    )
)

bson("result.bson", m_bayes=m_bayes)
