# Loading

using MLDatasets

image_raw = MNIST.convert2features(MNIST.traintensor(Float64))
label = MNIST.trainlabels() .+ 1

@info "Data size" size.((image_raw, label))

# Pre-processing

using MultivariateStats

D_pca = 40
pca = fit(PCA, image_raw; maxoutdim=D_pca)

image = transform(pca, image_raw)

@info "Processed data size" size(image)

# Data function

get_data(n=1_000) = tuple(image[:,1:n], label[1:n], 10)