############################################################
############################################################
##                                        |  _            ##
##  -+-      _  .  _   _     . |        __| |_| |\/| /--\ ##
##   |  |_| |   | | | |_| *  | |_   -- \__| |_, |  | \__/ ##
##                     _|   _|                            ##
##                                                        ##
################# Last update : Fri 21 Jul #################
               ##          By : Kai Xu     ##
               ##   Turing.jl : 0.3.0      ##
               ##       Julia : 0.5.2      ##
               ##############################
               ##############################

using JLD                 # for loading training data
using Plots               # for visualization
using Distributions       # for using all distributions
using Turing              # yeah - it's Turing.jl

################################
# 1. Hidden Markov Model (HMM) #
################################

# Load data
hmm_data = load("hmm-data.jld")["data"]

# Visualize data
plot(1:hmm_data["N"], hmm_data["obs"],
     line=:scatter, lab="obs", m=:circle)
plot!(1:hmm_data["N"], hmm_data["hid_ground"],
      line=:scatter, lab="hid (ground true)", m=:diamond)

# Define a HMM model with data as
#          K := number of hidden stats (=3)
#          N := number of observations (=20)
#   T_ground := transition matrix
# sig_ground := noise level
#        obs := observations
@model hmm_model(K, N, T_ground, sig_ground, obs) = begin
  hid = tzeros(Int, N)
  hid[1] ~ Categorical(ones(Float64, K) / K)
  obs[1] ~ Normal(hid[1], sqrt(sig_ground))
  for t in 2:N
    hid[t] ~ Categorical(T_ground[:, hid[t - 1]])
    obs[t] ~ Normal(hid[t], sqrt(sig_ground))
  end
end

N_samples = 100 # number of samples to generate

# Sample from "hmm_model" with data as "hmm_data" using
# "PG" with "10" particles for "100" iterations
hmm_chn = sample(hmm_model(data=hmm_data),
                 PG(10, 100))

# Visualize inference result
plot!(1:hmm_data["N"], hmm_chn[:hid][N_samples],
      line=:scatter, lab="hid (HMM)", m=:star4)

##############################################
# 2. Bayesian Hidden Markov Model (BayesHMM) #
##############################################

# Define a HMM model with data as
#       K := number of hidden stats (=3)
#       N := number of observations (=20)
#       a := param for noise prior (Inv-Gamma)
#       b := param for noise prior (Inv-Gamma)
#   alpha := param for transition matrix prior (Dirichlet)
#     obs := observations
@model bayes_hmm_model(K, N, a, b, alpha, obs) = begin
  # Inverse gamma prior on noise level
  sig ~ InverseGamma(a, b)

  # Dirichlet prior on transition matrix
  T = Vector{Vector{Real}}(K)
  for k = 1:K
    T[k] ~ Dirichlet(alpha)
  end

  hid = tzeros(Int, N)
  hid[1] ~ Categorical(ones(Float64, K) / K)
  obs[1] ~ Normal(hid[1], sqrt(sig))
  for t in 2:N
    hid[t] ~ Categorical(T[hid[t - 1]])
    obs[t] ~ Normal(hid[t], sqrt(sig))
  end
end

# Sample from "bayes_hmm_model" with data as "hmm_data" using
# "Gibbs" sampler for "N_samples" iterations by combing
# "PG" with "10" particles for "hid", and
# "HMC" with leapfrog params "0.2" and "3" for "T" and "sig"
bayes_hmm_chn = sample(bayes_hmm_model(data=hmm_data),
                       Gibbs(N_samples, # N_samples is same as before, 100
                             PG(10, 1, :hid),
                             HMC(1, 0.2, 3, :T, :sig)))

plot!(1:hmm_data["N"], bayes_hmm_chn[:hid][N_samples],
      line=:scatter, lab="hid (BayesHMM)", m=:star5)

########################################
# 3. Latent Dirichelt Allocation (LDA) #
########################################

# Load data
lda_data = load("lda-data.jld")["data"]

# Define the LDA model with parameters:
#   K := topic num (=4)        w := word instances
#   V := vocabulary (=10)    doc := doc instances
#   M := doc num (=50)         β := topic prior
#   N := count of words (=755) α := word prior
@model lda_model(K, V, M, N, w, doc, β, α) = begin
  θ = Vector{Vector{Real}}(M)
  for m = 1:M
    θ[m] ~ Dirichlet(α)
  end

  ϕ = Vector{Vector{Real}}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  z = tzeros(Int, N)
  for n = 1:N
    z[n] ~ Categorical(θ[doc[n]])
  end

  for n = 1:N
    w[n] ~ Categorical(ϕ[z[n]])
  end
end

N_samples = 10

lda_chn = sample(lda_model(data=lda_data),
                 Gibbs(N_samples,
                       PG(20, 1, :z),
                       HMC(1, 0.15, 5, :θ, :ϕ)))

###########################
# Optimized version below #

# Pre-compute indics
lda_data["lidx"] =
  (lda_data["doc"] .- 1) .* lda_data["V"] .+ lda_data["w"]

# Collapsed version with vectorization for (huge) speed-up
@model lda_model_vec(K, V, M, N, w, doc, β, α, lidx) = begin
  θ = Matrix{Real}(K, M)
  θ ~ [Dirichlet(α)]

  ϕ = Matrix{Real}(V, K)
  ϕ ~ [Dirichlet(β)]

  log_ϕ_dot_θ = log(ϕ * θ)
  # lp = mapreduce(n -> log_ϕ_dot_θ[w[n], doc[n]], +, 1:N)
  lp = sum(broadcast_getindex(log_ϕ_dot_θ, lidx))
  Turing.acclogp!(vi, lp)
end

N_samples = 1000

lda_vec_chn = sample(lda_model_vec(data=lda_data),
                     HMC(N_samples, 0.15, 5))

ϕ = mean(lda_vec_chn[:ϕ])

bar(lda_data["words"], [ϕ], layout=@layout([a; b; c; d]),
    labels=["topic 1" "topic 2" "topic 3" "topic 4"],
    legendfont=font(7), xrotation=15)

########################################
# 4. Bayesian Neural Network (BayesNN) #
########################################

# Load data
bayes_nn_data = load("bayes-nn-data.jld")["data"]

# Visualization data
vis_nn_data() = plot(bayes_nn_data["xs"][:,1],
                     bayes_nn_data["xs"][:,2],
                     group=bayes_nn_data["ts"], linetype=:scatter,
                     xlims=(-6, 6), ylims=(-6, 6))
vis_nn_data()

# Define sigmoid function
σ(x) = 1.0 ./ (1.0 + exp(-x))

# Define a feed-forward neural network
#   layer: input -+> hidden_1 -+> hidden_2 -+> output
#     dim:   2    |     3      |     2      |    1
#               tanh         tanh        sigmoid
nn_fwd(x, W₁, b₁, W₂, b₂, wₒ, bₒ) = begin
    h₁ = tanh(W₁' * x + b₁)
    h₂ = tanh(W₂' * h₁ + b₂)
    σ(wₒ' * h₂ + bₒ)[1]   # convert length-1 vector to scalar
end

alpha = 0.25              # regularizatin term
sigma = sqrt(1.0 / alpha) # std of the Gaussian prior

# Define the Bayes NN model with parameters:
#    N := num of data points (=40)
#   xs := data points
#   ts := true labels
@model bayes_nn(N, xs, ts) = begin
    W₁ = Matrix{Real}(2, 3)
    W₁ ~ [MvNormal(zeros(2), sigma .* ones(2))]
    b₁ ~ MvNormal(zeros(3), sigma .* ones(3))

    W₂ = Matrix{Real}(3, 2)
    W₂ ~ [MvNormal(zeros(3), sigma .* ones(3))]
    b₂ ~ MvNormal(zeros(2), sigma .* ones(2))

    wₒ ~ MvNormal(zeros(2), sigma .* ones(2))
    bₒ ~ Normal(0, sigma)

    for n = 1:N
        ts[n] ~ Bernoulli(nn_fwd(xs[n,:], W₁, b₁, W₂, b₂, wₒ, bₒ))
    end
end

N_samples = 2000

bayes_nn_ch = sample(bayes_nn(data=bayes_nn_data),
                     HMC(N_samples, 0.05, 4));

na2mat(na) = begin
    ncol = length(na); nrow = length(na[1])
    mat = Matrix{eltype(na[1])}(nrow, ncol)
    for n = 1:ncol mat[:, n] = na[n] end
    mat
end

_, map_idx = findmax(bayes_nn_ch[:lp])

nn_params = [map(na2mat, bayes_nn_ch[:W₁]), bayes_nn_ch[:b₁],
             map(na2mat, bayes_nn_ch[:W₂]), bayes_nn_ch[:b₂],
             bayes_nn_ch[:wₒ], bayes_nn_ch[:bₒ]]

map_predict(x, nn_params, map_idx) =
  nn_fwd(x, map(p -> p[map_idx], nn_params)...)

vis_nn_data()
contour!(linspace(-6, 6), linspace(-6, 6),
         (x, y) -> map_predict([x, y], nn_params, map_idx),
         xlims=(-6, 6), ylims=(-6, 6), title="BayesNN prediction (MAP)")

savefig("bnn-map.png")

bayes_predict(x, n_end, nn_params) =
  mean([nn_fwd(x, map(p -> p[i], nn_params)...) for i in 1:25:n_end])

vis_nn_data()
contour!(linspace(-6, 6), linspace(-6, 6),
         (x, y) -> bayes_predict([x, y], N_samples, nn_params),
         xlims=(-6, 6), ylims=(-6, 6), title="BayesNN prediction (Bayes)")

############################
# 5. Differential Equation #
############################

# On Jupyter Notebook
