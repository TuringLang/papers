include("data.jl")

const high_dim_gauss_str = "
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

using CmdStan

high_dim_gauss_model = Stanmodel(
    name="HighDimGauss", 
    model=high_dim_gauss_str, 
    nchains=1,
    Sample(
        algorithm=CmdStan.Hmc(
            CmdStan.Static(0.4),
            CmdStan.diag_e(),
            0.1,
            0.0,
        ),
        num_warmup=0,
        num_samples=2_000,
        adapt=CmdStan.Adapt(engaged=false),
        save_warmup=true,
    ),
)

@time status, chain = stan(high_dim_gauss_model, get_data(), summary=false)
