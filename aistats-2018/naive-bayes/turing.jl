using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

@model naive_bayes(image, label, D, N, C, ::Type{T}=Float64) where {T<:Real} = begin
    m = Matrix{T}(undef, D, C)
    for c = 1:C
        m[:,c] ~ MvNormal(fill(0, D), 10)
    end

    Threads.@threads for n = 1:N
        image[:,n] ~ MvNormal(m[:,label[n]], 1)
    end
end

model = naive_bayes(data["image"], data["label"], data["D"], data["N"], data["C"])

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