using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

const model_str = "
data {
  int D;
  int N;
  matrix[D,N] X;
  int<lower=0,upper=1> y[N];
}
transformed data {
  matrix[N,D] XT;
  XT = X';
}
parameters {
  vector[D] w;
}
transformed parameters {
  vector[N] v;
  vector[N] p;
  v = XT * w;
  for (n in 1:N)
    p[n] = inv_logit(v[n]);
}
model {
  w ~ normal(0, 1);
  y ~ bernoulli(p);
}
"

include("../infer_stan.jl")

;