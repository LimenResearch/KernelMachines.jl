using Documenter, KernelMachines, Literate

const example_dir = joinpath(@__DIR__, "..", "examples")
const generated_dir = joinpath(@__DIR__, "src", "generated")

for file in readdir(example_dir)
    Literate.markdown(joinpath(example_dir, file), generated_dir)
end

ENV["GKSwstype"] = "100"

makedocs(
    sitename="Kernel Machines",
    pages = [
        "index.md",
        "kernels.md",
        "Examples" => [
            "generated/multiscale.md",
            "generated/polynomial.md",
        ]
    ])
