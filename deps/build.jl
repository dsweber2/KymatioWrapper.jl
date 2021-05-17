using Conda
using PyCall
using Pkg
root = Symbol(Conda.ROOTENV)
if !Conda.exists("torch", root) || !Conda.exists("torchvision", root)
    println("either pytorch not found, installing in '$root'")
    Conda.runconda(`install pytorch torchvision -c pytorch -p "$root"`)
end
println("torchvision installed")


currentDirectory = pwd()
if Sys.iswindows()
    pipPath = joinpath(String(root), "Scripts", "pip")
else
    pipPath = joinpath(String(root), "bin", "pip")
end
println("using pip $(pipPath)")
run(`$(pipPath) install kymatio`)

try
    Conda.parseconda(`search kymatio`)
catch
    run(`$(pipPath) install kymatio`)
end

cd(currentDirectory)
