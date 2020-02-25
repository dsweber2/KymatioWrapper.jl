using Conda
using PyCall
using Pkg
# ENV["PATH_TO_KYMATIO"] = "/home/dsweber/allHail/projects/kymatio"
# kymatioPath = get(ENV, "PATH_TO_KYMATIO", :none)
# if !(typeof(kymatioPath) <: String)
#     kymatioPath = joinpath(ENV["HOME"], "kymatio")
# end
# if !isfile(joinpath(kymatioPath, "README.md"))
#     run(`git clone https://github.com/kymatio/kymatio $(kymatioPath)`)
# end
root = Symbol(Conda.ROOTENV)
if !Conda.exists("torch", root) || !Conda.exists("torchvision", root)
    println("either pytorch not found, installing")
    Conda.runconda(`install pytorch torchvision -c pytorch`)
end
println("torchvision installed")


currentDirectory = pwd()
pipPath = joinpath(String(root), "bin", "pip")
pythonPath = joinpath(String(root), "bin", "python")
run(`$(pipPath) install kymatio`)

try
    Conda.parseconda(`search kymatio`)
catch
    run(`$(pipPath) install kymatio`)
end

# cd(kymatioPath)
# run(`$(pipPath) install -r requirements.txt`)
# run(`$(pythonPath) setup.py install`)
# run(`$(pipPath) install -r requirements_optional_cuda.txt`)

cd(currentDirectory)
