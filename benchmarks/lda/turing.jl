using DrWatson
@quickactivate "TuringExamples"

using ReverseDiff, Memoization, Zygote, Turing
using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

@model lda(K, V, M, N, w, doc, alpha, beta) = begin
    theta ~ filldist(Dirichlet(alpha), M)
    phi ~ filldist(Dirichlet(beta), K)
    log_phi_dot_theta = log.(phi * theta)
    Turing.acclogp!(_varinfo, sum(log_phi_dot_theta[CartesianIndex.(w, doc)]))
end

model = lda(data["K"], data["V"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])

step_size = 0.01
n_steps = 4
test_zygote = true
test_tracker = true

include("../infer_turing.jl")

;
