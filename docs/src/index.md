# Home

Julia implementation of *discrete kernel machines* [^1].

[^1]: For the theoretical foundations of the framework, see [parametric machines](https://arxiv.org/abs/2007.02777), section 4.

To install, type

```julia
julia> import Pkg

julia> Pkg.add("KernelMachines")
```

in the Julia REPL.

!!! warning

    This package requires Julia 1.5.

To run the examples, you will also need to install Plots and, optionally, Optim:

```julia
julia> import Pkg

julia> Pkg.add(["Plots", "Optim"])
```
