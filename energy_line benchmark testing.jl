using BenchmarkTools #checks the time needed
#energy_line is perfect for @threads because it computes one nuy for every nux, calculationg the eigenvalue problem
#therefore the different nuy can be seperated with different cores

#please run the Julia rewrite with parallelism.jl and Noldnoldmodified.py translated to julia.jl files first to have the functions defined
function energy_line(nux, dictionary)
    kx = dkx * nux
    for nuy in nuy_list
        ky = dky * nuy
        values, vectors = eigen(hamiltonian(kx, ky))
        sorting = sortperm(values)
        values = real(values[sorting])
        vectors = vectors[:, sorting]
        dictionary[(Int(nux), Int(nuy))] = (values, transpose(vectors))
    end
end

dict_parallel = ThreadSafeDict()

function energy_line_parallel(nux, dictionary)
    kx = dkx * nux
    Threads.@threads for idx in eachindex(nuy_list)
        nuy = nuy_list[idx]
        values, vectors = eigen(hamiltonian(kx, dky * nuy))
        sorting = sortperm(values)
        values = real(values[sorting])
        vectors = vectors[:, sorting]
        dictionary[(Int(nux), Int(nuy))] = (values, transpose(vectors))
    end
end

function energy_line_parallel_local(nux)
    kx = dkx * nux
    local_dict = Dict{Tuple{Int, Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()
    Threads.@threads for idx in eachindex(nuy_list)
        nuy = nuy_list[idx]
        values, vectors = eigen(hamiltonian(kx, dky * nuy))
        sorting = sortperm(values)
        values = real(values[sorting])
        vectors = vectors[:, sorting]
        # Nicht in globales dict schreiben, sondern lokal sammeln
        local_dict[(Int(nux), Int(nuy))] = (values, transpose(vectors))
    end
    return local_dict
end
println("Benchmark for energy_line:")
dict_line = Dict()
benchmark_energy_line = @benchmark energy_line(0, $dict_line)
display(benchmark_energy_line)

println("benchmark for energy_line_parallel")
dict_parallel = Dict()
benchmark_energy_line_parallel = @benchmark energy_line_parallel(0, $dict_parallel)
display(benchmark_energy_line_parallel)

println("benchmark for energy_line_parallel_local") 
nux_test = 0
benchmark_energy_line_parallel_local = @benchmark energy_line_parallel_local($nux_test)
display(benchmark_energy_line_parallel_local)

#the threads couldnt seem to make the energy_line more efficient