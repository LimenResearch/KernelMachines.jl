function slice(mat::AbstractMatrix, list::Dims)
    map(cumsum(list), list) do post, diff
        pre = post - diff + 1
        return mat[pre:post, :]
    end
end