using BSON, PyPlot

function make_imggrid(imgs, n_rows, n_cols; gap::Int=1)
    n = length(imgs)
    d_row, d_col = size(first(imgs))
    imggrid = 0.5 * ones(n_rows * (d_row + gap) + gap, n_cols * (d_col + gap) + gap)
    i = 1
    for row = 1:n_rows, col = 1:n_cols
        if i <= n
            i_row = (row - 1) * (d_row + gap) + 1
            i_col = (col - 1) * (d_col + gap) + 1
            imggrid[i_row+1:i_row+d_row,i_col+1:i_col+d_col] .= imgs[i]
        else
            break
        end
        i += 1
    end
    return imggrid
end
    
m_bayes = BSON.load("result.bson")[:m_bayes]
    
imgs = [reshape(m_bayes[:,i], 28, 28)' for i in 1:size(m_bayes, 2)]

plt.figure(figsize=(5, 2))
plt.imshow(make_imggrid(imgs, 2, 5), cmap="gray")
plt.savefig("vis.png")