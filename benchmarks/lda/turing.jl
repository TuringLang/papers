using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing
using Turing.Core: filldist

@model lda(K, V, M, N, w, doc, alpha, beta, ::Type{T}=Float64) where {T} = begin
    theta = Matrix{T}(undef, K, M)
    theta ~ filldist(Dirichlet(alpha), M)
    phi ~ filldist(Dirichlet(beta), K)
    log_phi_dot_theta = log.(phi * theta)
    @logpdf() += sum(log_phi_dot_theta[CartesianIndex.(w, doc)])
end

model = lda(data["K"], data["V"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])

step_size = 0.01
n_steps = 4

include("../infer_turing.jl")

;
