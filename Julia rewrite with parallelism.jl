using LinearAlgebra     #for using eigen and sortperm
using SparseArrays      #for using spdiagm
using Serialization     #for using serialize
using Printf            #for using @sprintf 
using Plots             #for plots
using PlotlyJS
plotlyjs() #interactive backend for the plots, but it looks distorted now
using BenchmarkTools    #for looking at the speed of the code
using ThreadSafeDicts   #for parallelism with the dictionary used in energy_line_parallel

# Global variables:
p = nothing
q = nothing #how many lattice sites are in my unit cell
x_unitcells = nothing #how many unitcells in x 
y_unitcells = nothing #and y direction
flux = nothing #strength of the magnetic field
dkx = nothing  
dky = nothing
nux_list = nothing
nuy_list = nothing
kx_list = nothing
ky_list = nothing
matrix_size = nothing
filename = nothing


#This can probably done with another package, but I write meshgrid here
function meshgrid(x::AbstractVector, y::AbstractVector)
    X = repeat(x, 1, length(y))         # X: (length(x), length(y))
    Y = repeat(y', length(x), 1)        # Y: (length(x), length(y))
    return X, Y
end

#sets the basis for the rest of the code. Defines Variables 
function OneParticle(p_val, q_val, x_unitcells_val, y_unitcells_val)
    global p, q, x_unitcells, y_unitcells, flux, dkx, dky, nux_list, nuy_list, kx_list, ky_list, matrix_size, filename

    p = p_val
    q = q_val
    x_unitcells = x_unitcells_val
    y_unitcells = y_unitcells_val

    flux = p / q

    dkx = 2 * pi / x_unitcells
    dky = 2 * pi / y_unitcells

    nux_list = collect(-x_unitcells ÷ 2 : x_unitcells ÷ 2 - 1)
    nuy_list = collect(-y_unitcells ÷ 2 : y_unitcells ÷ 2 - 1)

    kx_list = dkx .* nux_list
    ky_list = dky .* nuy_list

    matrix_size = q
    filename = "/p=$(p)_q=$(q)_lx=$(x_unitcells)_ly=$(y_unitcells).jls"
end

#Creates the bloch hamiltonian equation which has to be diagonalized 
function hamiltonian(kx, ky)
    if matrix_size == 1
        matrix = 2 * (cos(kx) + cos(ky))
    elseif matrix_size == 2
        matrix = [
            -2 * cos(kx)              1 + exp(-im * ky);
             1 + exp(im * ky)         2 * cos(kx)
        ]
    else
        main_diag = [2 * cos(kx + 2π * flux * r) for r in 1:q]
        upper = fill(1.0, q - 1)
        lower = fill(1.0, q - 1)
        corner_top = [exp(-im * ky)]
        corner_bottom = [exp(im * ky)]

        matrix = spdiagm(
            0 => main_diag,
            1 => upper,
           -1 => lower,
            q - 1 => corner_top,
           -(q - 1) => corner_bottom
        )
    end
    return -Matrix(matrix)
end

#energy_line diagonalizes the hamiltonian matrix for one kx value for all ky values
dict_parallel = ThreadSafeDict()

function energy_line_parallel(nux, dictionary)
    kx = dkx * nux
    Threads.@threads for idx in eachindex(nuy_list)
        nuy = nuy_list[idx]
        values, vectors = eigen(hamiltonian(kx, dky * nuy))
        sorting = sortperm(values)
        values = real(values[sorting])
        vectors = vectors[:, sorting]
        dictionary[(Int(nux), Int(nuy))] = (values, vectors)
    end
end

#loops energy line for all kx values and saves the resulting dictionary for further faster callbacks
function energy_spectrum_parallel()
    # make local dicts so it dodesnt write every time (thread-local)
    nthreads = Threads.nthreads()
    dicts = [Dict{Tuple{Int, Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}() for _ in 1:nthreads]

    Threads.@threads for i in eachindex(nux_list)
        nux = nux_list[i]
        energy_line_parallel(nux, dicts[Threads.threadid()])
    end
    #merge all dicts
    energy_dictionary = Dict{Tuple{Int, Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()
    for d in dicts
        merge!(energy_dictionary, d)
    end

    return energy_dictionary
end



#creates our band structure by plotting the eigen values of the diagonalized matrices on every kx and ky point
function plot_3d_parallel()
    nx = length(kx_list)
    ny = length(ky_list)
    full_energy_list = Array{Float64,3}(undef, nx, ny, matrix_size)

    energies = energy_spectrum_parallel()

    Threads.@threads for idx in 1:(nx * ny)
        i = (idx - 1) ÷ ny + 1
        j = (idx - 1) % ny + 1
        full_energy_list[i, j, :] = energies[(nux_list[i], nuy_list[j])][1]
    end

    kxs, kys = meshgrid(kx_list ./ pi, ky_list ./ pi)

    plt = Plots.plot(title = "p/q = $(p)/$(q)", xlabel = "kₓ/π", ylabel = "kᵧ/π", zlabel = "E/t", legend = false)

    for r in 1:matrix_size
        surface!(plt, kys, kxs, full_energy_list[:, :, r])
    end

    display(plt)
end

#gives the probability distribution of the particle  to be in a site in the bloch state of kx and ky in the corresponding band
function plot_state(nux, nuy, band_index=1)
    energies = energy_spectrum_parallel()
    plotted_state = abs2.(energies[(nux, nuy)][2][band_index, :])

    state_labels = 1:matrix_size
    plt2 = Plots.bar(state_labels, plotted_state, legend=false, xlabel="site α", ylabel="|ψ|²", #need to use Plots.bar because its ambiguous otherwise
        title=@sprintf("Prob. dens. for kx/π = %.2f, ky/π = %.2f, q = %d",
        dkx * nux / pi, dky * nuy / pi, q))
    display(plt2) 
end

# Main function
@time begin #gibt die Zeit aus, die gebraucht wird
    OneParticle(1, 4, 165,300)
    println(q)
    plot_3d_parallel()
    plot_state(0, 0, 1) #kx, ky and band_index
    println("the code took ")
end
println("to complete")
