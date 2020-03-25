using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing
using Turing.Core: filldist, arraydist

@model naive_bayes(image, label, D, N, C, ::Type{T}=Float64) where {T<:Real} = begin
    m ~ filldist(Normal(0, 10), D, C)
    image ~ arraydist(Normal.(m[:,label], 1))
end 

model = naive_bayes(data["image"], data["label"], data["D"], data["N"], data["C"])

step_size = 0.1
n_steps = 4

include("../infer_turing.jl")

# Save result

if !isnothing(chain)
    using BSON

    m_data = chain[:m].value.data

    m_bayes = mean(
        map(
            i -> reconstruct(pca, Matrix{Float64}(reshape(m_data[i,:,1], D_pca, 10))), 
            1_000:100:2_000
        )
    )

    bson("result.bson", m_bayes=m_bayes)
end