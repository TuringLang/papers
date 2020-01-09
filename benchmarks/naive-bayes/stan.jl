using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

const model_str = "
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

include("../infer_stan.jl")

;