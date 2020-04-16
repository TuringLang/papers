using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Memoization, Turing
using DistributionsAD: MvCategorical
using StatsFuns: logsumexp

@model hmm_semisup(K, T_unsup, w, z, u, alpha, beta) = begin
    theta ~ filldist(Dirichlet(alpha), K)
    phi ~ filldist(Dirichlet(beta), K)
    w ~ MvCategorical(phi[:, z])
    z[2:end] ~ MvCategorical(theta[:, z[1:end-1]])

    TF = eltype(theta)
    acc = similar(alpha, TF, K)
    gamma = similar(alpha, TF, K)
    temp_gamma = similar(alpha, TF, K)
    for k in 1:K
        gamma[k] = log(phi[u[1],k])
    end
    for t in 2:T_unsup
        for k in 1:K
            for j in 1:K
                acc[j] = gamma[j] + log(theta[k,j]) + log(phi[u[t],k])
            end
            temp_gamma[k] = logsumexp(acc)
        end
        gamma .= temp_gamma
    end
    @logpdf() += logsumexp(gamma)
end

model = hmm_semisup(data["K"], data["T_unsup"], data["w"], data["z"], data["u"], data["alpha"], data["beta"])

step_size = 0.0001
n_steps = 4
test_zygote = false
test_tracker = false

include("../infer_turing.jl")

;
