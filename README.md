# Julia-Programming-Project
Rewriting Python Code into Julia and optimizing it

This is a Julia Programming Project, where I will rewrite already existing Python code into Julia code. I will try to make the code more efficient.

How to run the code:
IMPORTANT!!!
Before running either the "Noldnold.py", "Noldnoldmodified.py", "Noldnoldmodified.py translated to Julia.jl" or "Julia rewrite with parallelism.jl", in your Project there first has to exist a folder "./dictionaries/one_particle/".
The code will NOT create this folder, the User has to create it manually.

plot_3d needs 4 inputs: p,q,x_unitcells,y_unitcells.
plot_state() needs 3 inputs: nux, nuy,band_index.
Use INTEGERS for all inputs with 
0 =< nux =< x_unitcells
0 =< nuy =< y_unitcells
0 =< band_index =< q

Read the project report "RSE Julia Poject Report" for more information.
