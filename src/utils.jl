# TODO: decide on slice versus view
# Using view, there would be issues with CuArrays dispatch
function split_matrix(mat::AbstractMatrix, dims::Dims)
    map(cumsum(dims), dims) do post, diff
        pre = post - diff + 1
        return mat[pre:post, :]
    end
end

# from Flux
glorot_uniform(dims...) = (rand(dims...) .- 0.5) .* sqrt(24 / sum(dims))
