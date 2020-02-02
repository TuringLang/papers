using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using CmdStan

# See:
# https://ecosang.github.io/blog/study/dirichlet-process-with-stan/

data["C"] = 100

stan_model = "
data{
  int<lower=0> C;//num of cludter
  int<lower=0> N;//data num
  real y[N];
}

parameters {
  real mu_cl[C]; //cluster mean
  real <lower=0,upper=1> v[C];
  real<lower=0> sigma_cl[C]; // error scale
  //real<lower=0> alpha; // hyper prior DP(alpha,base)
}

transformed parameters{
  simplex [C] pi;
  pi[1] = v[1];
  // stick-break process based on The BUGS book Chapter 11 (p.294)
  for(j in 2:(C-1)){
      pi[j]= v[j]*(1-v[j-1])*pi[j-1]/v[j-1];
  }
  pi[C]=1-sum(pi[1:(C-1)]); // to make a simplex.
}

model {
  real alpha = 1;
  real a=0.001;
  real b=0.001;
  real ps[C];
  sigma_cl ~ inv_gamma(a,b);
  mu_cl ~ normal(0,5);
  v ~ beta(1,alpha);

  for(i in 1:N){
    for(c in 1:C){
      ps[c]=log(pi[c])+normal_lpdf(y[i]|mu_cl[c],sigma_cl[c]);
    }
    target += log_sum_exp(ps);
  }

}
"

step_size = 0.01
n_steps = 4

include("../infer_stan.jl")

;
