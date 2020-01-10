using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

@model lda(K, V, M, N, w, doc, alpha, beta, ::Type{T}=Float64) where {T} = begin
    theta = Matrix{T}(undef, K, M)
    for m in 1:M
        theta[:,m] ~ Dirichlet(alpha)
    end
    phi = Matrix{T}(undef, V, K)
    for k in 1:K
        phi[:,k] ~ Dirichlet(beta)
    end

    log_phi_dot_theta = log.(phi * theta)
    @logpdf() += mapreduce(n -> log_phi_dot_theta[w[n],doc[n]], +, 1:N)
end

model = lda(data["K"], data["V"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])

step_size = 0.01
n_steps = 4

include("../infer_turing.jl")

;
