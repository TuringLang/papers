using DrWatson
@quickactivate "TuringExamples"

using ReverseDiff, Memoization, Zygote, Turing
using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

#=
@model lda_unvectorized(K, V, M, N, w, doc, alpha, beta, ::Type{T}=Float64) where {T} = begin
    theta = Matrix{T}(undef, K, M)
    for i in 1:M
        theta[:,i] ~ Dirichlet(alpha)
    end
    phi = Matrix{T}(undef, V, K)
    for i in 1:K
        phi[:,i] ~ Dirichlet(beta)
    end
    for i in 1:N
        temp = zero(T)
        for k in 1:K
            temp += phi[w[i],k] * theta[k,doc[i]]
        end
        Turing.acclogp!(_varinfo, log(temp))
    end
end
=#

@model lda_unvectorized(K, V, M, N, w, doc, alpha, beta, ::Type{T}=Float64) where {T} = begin
    theta = Matrix{T}(undef, K, M)
    for i in 1:M
        theta[:,i] ~ Dirichlet(alpha)
    end
    phi = Matrix{T}(undef, V, K)
    for i in 1:K
        phi[:,i] ~ Dirichlet(beta)
    end
    log_phi_dot_theta = log.(phi * theta)
    for i in 1:length(w)
        Turing.acclogp!(_varinfo, log_phi_dot_theta[w[i], doc[i]])
    end
end

model = lda_unvectorized(data["K"], data["V"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])

step_size = 0.01
n_steps = 4
test_zygote = false
test_tracker = false

include("../infer_turing.jl")

;
