using BenchmarkTools

println("Benchmark for energy_spectrum:")
benchmark_energy_spectrum = @benchmark energy_spectrum(calculate_new=true)
display(benchmark_energy_spectrum)

println("\nBenchmark for energy_spectrum_parallel:")
benchmark_energy_spectrum_parallel = @benchmark energy_spectrum_parallel()
display(benchmark_energy_spectrum_parallel)

#this made it faster by 20%