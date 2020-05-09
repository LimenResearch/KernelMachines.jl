# MNIST example adapted from https://github.com/FluxML/model-zoo
using KernelMachines
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy
using Base.Iterators: partition
using Printf, BSON

Base.@kwdef mutable struct Args
    lr::Float64 = 3e-3
    epochs::Int = 20
    batch_size = 128
    savepath::String = "./examples/saved" 
end

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

function get_processed_data(args)
    # Load labels and images from Flux.Data.MNIST
    train_labels = MNIST.labels()
    train_imgs = MNIST.images()
    mb_idxs = partition(1:length(train_imgs), args.batch_size)
    train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs] 
    
    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_labels = MNIST.labels(:test)
    test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

    return train_set, test_set

end

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

accuracy(x, y, model) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))

function build_model(args)
    Chain(
        t -> reshape(t, :, last(size(t))),
        Splitter(
            Linear(28*28, 32),
            Linear(28*28, 32),
            Linear(28*28, 32),
            Linear(28*28, 32),
            Linear(28*28, 32),
            Linear(28*28, 32),
            Linear(28*28, 10),
        ),
        KernelMachine((32, 32, 32, 32, 32, 32, 10), 64),
        last,
        softmax,
    )
end

_dot(a, b) = dot(a, b)
_dot(a::Function, b::Function) = false
_dot(a::Chain, b::Chain) = sum(map(_dot, a.layers, b.layers))

function train(; kws...)	
    args = Args(; kws...)

    @info("Loading data set")
    train_set, test_set = get_processed_data(args)

    # Define our model.
    @info("Building model...")
    model = build_model(args)

    # Make sure our model is nicely precompiled before starting our training loop
    model(train_set[1][1])

    # `loss()` calculates the crossentropy loss between our prediction `y_hat`
    # (calculated from `model(x)`) and the ground truth `y`.  We augment the data
    # a bit, adding gaussian random noise to our image to make it more robust.
    function loss(x, y)    
        ŷ = model(x)
        return logitcrossentropy(ŷ, y) + 1f-5 * _dot(model, model)
    end
	
    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    opt = ADAM(args.lr)
	
    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)
	    
        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
            break
        end
	
        # Calculate accuracy:
        acc = accuracy(test_set..., model)
		
        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        x, y = test_set
        @show logitcrossentropy(model(x), y)
        @show _dot(model, model)
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
	
        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(" -> New best accuracy! Saving model out to mnist_km.bson")
            BSON.@save joinpath(args.savepath, "mnist_km.bson") params=cpu.(params(model)) epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end
	
        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
   
            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end
	
        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end
    end
end

# Testing the model, from saved model
function test(; kws...)
    args = Args(; kws...)
    
    # Loading the test data
    _,test_set = get_processed_data(args)
    
    # Re-constructing the model with random initial weights
    model = build_model(args)
    
    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "mnist_km.bson") params
    
    # Loading parameters onto the model
    Flux.loadparams!(model, params)
    
    @show accuracy(test_set...,model)
end

train(batch_size = 64, lr = 1e-3)
test()