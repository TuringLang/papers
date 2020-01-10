using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real m;
  real<lower=0> s;
}
model {
  m ~ normal(0, 1);
  s ~ cauchy(0, 5);
  y ~ normal(m, s);
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