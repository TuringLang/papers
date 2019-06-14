#Import Turing, Distributions, LinearAlgebra and DataFrames
using Turing, Distributions, LinearAlgebra, DataFrames

# Import MCMCChain, Plots, StatsPlots, GraphPlot, GraphRecipes, PyCall and Statistics for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots, GraphPlot, GraphRecipes, PyCall, Statistics

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

# Set hyperparameters- alpha is the Dirichlet parameter and eta is the probability matrix between clusters 
global eta
K = 2
alpha = 0.1 * ones(K)
eta = Symmetric(rand(K,K))

# Generate data using randomly generated cluster assignments
N = 3
clusters = rand(1:K, N)
graph = zeros(N,N)
for i=1:N
    graph[i, i] = 1
    for j = 1:i-1
        cluster_i = clusters[i]
        cluster_j = clusters[j]
        eta_ij = eta[cluster_i, cluster_j]
        if(rand(Binomial(1,eta_ij), 1)[1] == 1)
            graph[i, j] = 1
            graph[j, i] = 1
        end
    end
end
#Visualisation using matplotlib backend
pyplot()
heatmap(graph, title = "Heatmap of Original Graph")
gui()
graphplot(graph, reuse = false, title = "Original Graph")
gui()

# Define the model
@model mmsb(alpha, eta, graph, N) = begin
    pi = Vector{Vector}(undef, N)
    for n=1:N
        pi[n] ~ Dirichlet(alpha)
    end
    
    for n=1:N
        for m=1:n-1
            #Setting lower bound to 0 avoid domain errors in the Bernoulli distribution
            val = max((pi[n])'*eta*pi[m], 0)
            graph[n, m] ~ Bernoulli(val)
        end
    end
end;

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Sample using HMC
chain = sample(mmsb(alpha, eta, graph, N), HMC(50000, 0.01, 10) )

plot(chain, reuse = false, title = "Sampler Plot")
gui()

global maxm, count, i, clusters_pred

# Find the predicted clusters using the new parameters
clusters_pred = Array{Int64}(undef, N)
df = DataFrame(chain[:pi])
means = colwise(mean, df)

#Finding the index of the max value in each interval of size K (which represents the probabilities for each node) to get the predicted cluster assignments
count = 1
i = 1
maxm = means[1]
for mean in means
    global maxm, count, i, clusters_pred
    if(mean >= maxm)
        clusters_pred[i] = count
        maxm = mean
    end
    count += 1
    if(count == K + 1)
        count = 1
        maxm = -Inf
        i += 1
    end
end

#Reconstructing the predicted graph using predicted cluster assignments
global graph_pred
graph_pred = zeros(N,N)
for i=1:N
    global graph_pred, clusters_pred, eta
    graph_pred[i, i] = 1
    for j = 1:i-1
        global graph_pred, clusters_pred, eta
        cluster_i = clusters_pred[i] 
        cluster_j = clusters_pred[j]
        eta_ij = eta[cluster_i, cluster_j]
        if(rand(Binomial(1,eta_ij), 1)[1] == 1)
            graph_pred[i, j] = 1
            graph_pred[j, i] = 1
        end
    end
end
heatmap(graph_pred, title = "Heatmap of Predicted Graph")
gui()

global non_zero
graph_diff = graph - graph_pred
non_zero = 0
for row in graph_diff
    global non_zero
    for ele in row
        if(ele != 0)
            non_zero += 1
        end
    end
end

println("The number of mismatches is ", non_zero)
println("Press ENTER to exit")
read(stdin, Char)
