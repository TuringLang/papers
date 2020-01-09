@model naive_bayes(image, label, D, N, C, ::Type{T}=Float64) where {T<:Real} = begin
    m = Matrix{T}(undef, D, C)
    for c = 1:C
        m[:,c] ~ MvNormal(fill(0, D), 10)
    end

    Threads.@threads for n = 1:N
        image[:,n] ~ MvNormal(m[:,label[n]], 1)
    end
end

# Model function

get_model(args...) = naive_bayes(args...)
