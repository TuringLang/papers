using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

using StatsFuns: logsumexp

@model lda(K, M, N, w, doc, alpha, beta) = begin
    phi ~ Multi(Dirichlet(beta), K)
    theta ~ Multi(Dirichlet(alpha), M)

    log_phi_dot_theta = log.(phi * theta)
    @logpdf() += mapreduce(n -> log_phi_dot_theta[w[n],doc[n]], +, 1:N)
end

model = lda(data["K"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])

step_size = 0.001
n_steps = 4

include("../infer_turing.jl")

;
