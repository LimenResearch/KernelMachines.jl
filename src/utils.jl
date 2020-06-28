function slice(mat::AbstractMatrix, list::Dims)
    map(cumsum(list), list) do post, diff
        pre = post - diff + 1
        return mat[pre:post, :]
    end
end

# from Flux
glorot_uniform(dims...) = (rand(dims...) .- 0.5) .* sqrt(24 / sum(dims))