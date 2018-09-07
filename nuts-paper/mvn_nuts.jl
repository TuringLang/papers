using Distributions
using Turing

# 250-dimensional multivariate normal (MVN)
@model mvn_nuts(A, d) = begin
  Θ ~ d
end

# Sample a precision matrix A from a Wishart distribution
# with identity scale matrix and 250 degrees of freedome
dim = 250
A   = rand(Wishart(250, Matrix{Float64}(I, 250, 250)))
d   = MvNormal(zeros(250), A)
chain = sample(mvn_nuts(A, d), NUTS(1000, 0.65))
