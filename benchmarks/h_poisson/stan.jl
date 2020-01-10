using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int N;
  int y[N];
  int Ns;
  int idx[N];
  real x[N];
}
parameters {
  real a0;
  vector[Ns] a0s;
  real a1;
  real<lower=0> a0_sig;
}
model {
  vector[N] mu;
  a0 ~ normal(0, 10);
  a1 ~ normal(0, 1);
  a0_sig ~ cauchy(0, 1);
  a0s ~ normal(0, a0_sig);
  for(i in 1:N) mu[i] = exp(a0 + a0s[idx[i]] + a1 * x[i]);
  y ~ poisson(mu);
}
"

alg = CmdStan.Hmc(
    CmdStan.Static(0.04),
    CmdStan.diag_e(),
    0.01,
    0.0,
)

include("../infer_stan.jl")

;