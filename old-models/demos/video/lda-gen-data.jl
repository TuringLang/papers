using Distributions

V = 10  # words: river, stream, bank, money, loan, bayes, gaussian, inference, neural, network
words = ["river", "stream", "bank", "money", "loan", "bayes", "gaussian", "inference", "neural", "network"]
K = 4   # topics: river, bank, bayes, nn

phi = [0.331 0.331 0.331 0.001 0.001 0.001 0.001 0.001 0.001 0.001;
       0.001 0.001 0.331 0.331 0.331 0.001 0.001 0.001 0.001 0.001;
       0.001 0.001 0.001 0.001 0.001 0.331 0.331 0.331 0.001 0.001;
       0.001 0.041 0.001 0.001 0.001 0.151 0.101 0.201 0.201 0.301]'

M = 50 # docs
avg_doc_length = 15
doc_length = rand(Poisson(avg_doc_length), M)
N = sum(doc_length)

α = ones(K) ./ K
β = ones(V) ./ V

theta = rand(Dirichlet(α), M)

w = Vector{Int}(N)
doc = Vector{Int}(N)
n = 1
for m in 1:M
  for i in 1:doc_length[m]
    z = rand(Categorical(theta[:,m]))
    w[n] = rand(Categorical(phi[:,z]))
    doc[n] = m;
    n = n + 1;
  end
end

lda_data = Dict(
  "V" => V,
  "K" => K,
  "words" => words,
  "M" => M,
  "N" => N,
  "w" => w,
  "doc" => doc,
  "α" => α,
  "β" => β)

# Save to file

using JLD

save("lda-data.jld", "data", lda_data)
