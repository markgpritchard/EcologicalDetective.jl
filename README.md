# EcologicalDetective.jl

This code recreates some of the functions and pseudocode from the book *The Ecological 
Detective* by Ray Hilborn and Marc Mangel. The code is being developed as a learning 
exercise. It is not associated with the authors or publishers of the book, and is saved here 
in 'as is' form.

This code base uses the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> EcologicalDetective.jl

To (locally) reproduce this project:

0. Download this code base. 
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson")  # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "EcologicalDetective.jl"
```
which auto-activate the project and enable local path handling from DrWatson.

As this code is being used for learning only, the all functions are included in the 
`scripts` files so that they can be seen in order and easily modified.

No formal help text is provided. Comments within the functions should make clear what each 
function does.
