
using DrWatson
@quickactivate "EcologicalDetective.jl"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chapter 5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using CairoMakie
using Distributions: Uniform 
using Random: seed!


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 5.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function sumsquares(xs, ys, A, B=[ 0 ], C=[ 0 ]) 
    S_min = Inf 
    A_star = minimum(A)
    B_star = minimum(B)
    C_star = minimum(C)
    for _A in A, _B in B, _C in C 
        S = calcsumsquares(xs, ys, _A, _B, _C)
        if S < S_min 
            S_min = S 
            A_star = _A
            B_star = _B
            C_star = _C
        end
    end
    return @ntuple A_star B_star C_star S_min
end

function calcsumsquares(xs, ys, _A, _B, _C)
    S = 0.0
    for (x, y) in zip(xs, ys) 
        y_pre = _A + _B * x + _C * x^2
        S += (y_pre - y)^2
    end
    return S 
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 5.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A_true = 1 
B_true = 0.5
C_true = 0.25

xs = 1:10
deterministic_ys = [ A_true + B_true * x + C_true * x^2 for x in xs ]
#10-element Vector{Float64}:
#  1.75
#  3.0
#  4.75
#  7.0
#  9.75
# 13.0
# 16.75
# 21.0
# 25.75
# 31.0
seed!(52)
ys = [ y + rand(Uniform(-3, 3)) for y in deterministic_ys ]
#10-element Vector{Float64}:
# -0.801420830575224        
#  5.220028441934401        
#  4.857563016606784        
#  5.017014395982918
#  8.547298339387682
# 10.249147129646872
# 14.285642857071151
# 23.925754331989822
# 27.76695933239202
# 32.86753067381784

result52 = sumsquares(xs, ys, 0:0.1:3, 0:0.05:2, 0:0.025:1)

@unpack A_star, B_star, C_star = result52
A_star
#0.7
B_star
#0.00
C_star
#0.325

pred_ys = [ A_star + B_star * x + C_star * x^2 for x in xs ]
#10-element Vector{Float64}:
#  1.025
#  2.0
#  3.625
#  5.9
#  8.825
# 12.4
# 16.625
# 21.5
# 27.025
# 33.2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Goodness of fit profiles / figure 5.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function goodnessoffitprofile(xs, ys, vrange; A=:fixed, B=:fixed, C=:fixed)
    gof = zeros(length(vrange))
    for (i, v) in enumerate(vrange)
        @unpack S_min = goodnessoffitprofile_ss(xs, ys, v; A, B, C)
        gof[i] = S_min
    end
    return gof
end

function goodnessoffitprofile_ss(xs, ys, fixedv; A, B, C)
    args = [ goodnessoffitprofile_ssv(Z, fixedv) for Z in [ A, B, C ] ]
    return sumsquares(xs, ys, args...) 
end

function goodnessoffitprofile_ssv(v::Symbol, fixedv)
    @assert v == :fixed 
    return [ fixedv ]  # i.e. as a vector of length 1 
end 

function goodnessoffitprofile_ssv(v::T, fixedv) where T <: Union{<:AbstractVector, <:AbstractRange} 
    return v 
end

## goodness of fit profile when fixing values of A 
goodnessoffitvals_A = goodnessoffitprofile(xs, ys, -5:0.01:5; B=0:0.05:2, C=0:0.025:1)

## fixing B
goodnessoffitvals_B = goodnessoffitprofile(xs, ys, -5:0.01:5; A=0:0.1:3, C=0:0.025:1)

## fixing C
goodnessoffitvals_C = goodnessoffitprofile(xs, ys, -5:0.01:5; A=0:0.1:3, B=0:0.05:2)

fig5_1 = let 
    fig = Figure()
    axs = [ Axis(fig[2 * i - 1, 1]) for i in 1:3 ]
    for (i, gof) in enumerate(
        [ goodnessoffitvals_A, goodnessoffitvals_B, goodnessoffitvals_C ]
    )
        lines!(axs[i], -5:0.01:5, gof; color=:black) 
    end
    for (i, lbl) in enumerate([ "A", "B", "C" ])
        Label(fig.layout[2 * i, 1], lbl; tellwidth=false)
    end
    Label(fig.layout[1:5, 0], "Goodness of fit"; rotation=π/2, tellheight=false)
    for r in [ 1, 3, 5 ] rowgap!(fig.layout, r, 5) end
    fig
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Penalize additional parameters / equation 5.11
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function penalizedsumsquares(xs, ys, args...)
    @unpack S_min = sumsquares(xs, ys, args...) 
    m = length(args)
    n = length(ys)
    return S_min / (n - 2*m)
end  

## "Model 1"
penalizedsumsquares(xs, ys, 0:0.1:3)
#272.6181824007249

## "Model 2"
penalizedsumsquares(xs, ys, 0:0.1:3, 0:0.05:2)
#50.07755955317532

## "Model 3"
penalizedsumsquares(xs, ys, 0:0.1:3, 0:0.05:2, 0:0.025:1)
#8.181118272840548
