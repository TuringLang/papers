using DrWatson
@quickactivate "TuringExamples"

# Ref: https://github.com/stan-dev/example-models/blob/master/misc/cluster/lda/lda.stan

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

const model_str = "
data {
  int<lower=2> K;               // num topics
  int<lower=2> V;               // num words
  int<lower=1> M;               // num docs
  int<lower=1> N;               // total word instances
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  simplex[K] theta[M];   // topic dist for doc m
  simplex[V] phi[K];     // word dist for topic k
}
model {
  for (m in 1:M) {
    theta[m] ~ dirichlet(alpha);  // prior
  }
  for (k in 1:K) {
    phi[k] ~ dirichlet(beta);     // prior
  }
  for (n in 1:N) {
    for (k in 1:K) {
      target += log(theta[doc[n],k] * phi[k,w[n]]);
    }
  }
}
"

step_size = 0.01
n_steps = 4

include("../infer_stan.jl")

;
