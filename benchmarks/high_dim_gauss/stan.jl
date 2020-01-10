using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

const model_str = "
data {
  int D;
}
parameters {
  real m[D];
}
model {
  for (d in 1:D)
    m[d] ~ normal(0, 1);
}
"

alg = CmdStan.Hmc(
    CmdStan.Static(0.4),
    CmdStan.diag_e(),
    0.1,
    0.0,
)

include("../infer_stan.jl")

;