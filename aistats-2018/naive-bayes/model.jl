@model naive_bayes(image, label, C, ::Type{T}=Float64) where {T<:Real} = begin
    D, N = size(image)
    
    m = Matrix{T}(undef, D, C)
    for c = 1:C
        m[:,c] ~ MvNormal(fill(0.0, D_pca), fill(10.0, D_pca))
    end

    Threads.@threads for n = 1:N
        image[:,n] ~ MvNormal(m[:,label[n]], fill(1.0, D))
    end
end

# Model function

get_model(args...) = naive_bayes(args...)
