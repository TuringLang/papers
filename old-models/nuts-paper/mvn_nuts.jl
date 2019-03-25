using Distributions, LinearAlgebra
using Turing

# 250-dimensional multivariate normal (MVN)
@model mvn_nuts(A, d) = begin
  Θ ~ d
end

# Sample a precision matrix A from a Wishart distribution
# with identity scale matrix and 250 degrees of freedome
dim2 = 25
A   = rand(Wishart(dim2, Matrix{Float64}(I, dim2, dim2)))
d   = MvNormal(zeros(dim2), A)
chain = sample(mvn_nuts(A, d), NUTS(1000, 0.65))
