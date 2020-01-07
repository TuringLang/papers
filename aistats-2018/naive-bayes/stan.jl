include("data.jl")

const naive_bayes_str = "
data {
  int C;
  int D;
  int N;
  matrix[D,N] image;
  int<lower=1,upper=C> label[N];
}
parameters {
  matrix[D,C] m;
}
model {
  for (c in 1:C)
    for (d in 1:D)
        m[d,c] ~ normal(0, 10);
      
  for (n in 1:N)
    for (d in 1:D)
        image[d,n] ~ normal(m[d,label[n]], 1);
}
"

using CmdStan

naive_bayes_model = Stanmodel(
    name="NaiveBayes", 
    model=naive_bayes_str, 
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

status, chain = stan(naive_bayes_model, get_data(), summary=true)
