using Distributions

K = 3         # number of hidden states
N = 20        # number of time stamps
a, b = 3, 0.5 # Inv-Gamma parameters

# Transition matrix
alpha = collect(ones(K))  # trans prior
T = rand(Dirichlet(alpha), K)

# Noise level
sig = rand(InverseGamma(a, b))

hid = Vector{Int}(N)      # hidden states
obs = Vector{Float64}(N)  # observed states

hid[1] = rand(1:K);
for t in 2:N
  hid[t] = rand(Categorical(T[:,hid[t - 1]]))
end
for t in 1:N
  obs[t] = rand(Normal(hid[t], sqrt(sig)))
end

hmm_data = Dict(
  "K" => K,
  "N" => N,
  "a" => a,
  "b" => b,
  "alpha" => alpha,
  "obs" => obs,
  "T_ground" => T,
  "hid_ground" => hid,
  "sig_ground" => sig)

# Save to file

using JLD

save("hmm-data.jld", "data", hmm_data)
