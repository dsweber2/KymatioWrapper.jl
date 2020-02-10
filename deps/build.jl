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
root = Conda.ROOTENV
if !Conda.exists("torchvision")
    println("either pytorch or torchvision not found, installing both")
    Conda.runconda(`install pytorch torchvision -c pytorch`)
end
println("torchvision installed")

try
    Conda.parseconda(`search kymatio`)
catch
    run(`$(pipPath) install kymatio`)
end

currentDirectory = pwd()
pipPath = joinpath(root, "bin", "pip")
pythonPath = joinpath(root, "bin", "python")
run(`$(pipPath) install kymatio`)

# cd(kymatioPath)
# run(`$(pipPath) install -r requirements.txt`)
# run(`$(pythonPath) setup.py install`)
# run(`$(pipPath) install -r requirements_optional_cuda.txt`)

cd(currentDirectory)


ENV["PYTHON"] = ""
Pkg.build("PyCall")