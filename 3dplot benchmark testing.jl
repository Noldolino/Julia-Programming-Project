using BenchmarkTools

@benchmark plot3d()            
@benchmark plot3d_parallel()   
#this didnt seem to improve the runtime either, it actually got worse