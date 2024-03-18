
using DrWatson
@quickactivate "EcologicalDetective.jl"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chapter 6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using BenchmarkTools
using CairoMakie
using DataFrames
using Random: seed!
using StatsBase: fweights, mean, sample

# The functions were initially written with `DataFrames`, then for Pseudocode 6.3 re-written 
# for vectors of egg numbers and clutch sizes, leading to some redundancy. The functions
# using DataFrames are much slower than those using the reformatted data, one of them
# taking 50 seconds vs 0.8 seconds. I have therefore commented some of the function calls so
# they won't be run unintentionally. 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure 6.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

table6_1 = DataFrame(:E => 4:23)
insertcols!(
    table6_1, 
    :C1 => [ table6_1.E[i] in [ 6, 7, 13 ] ? 1 : 0 for i in axes(table6_1, 1) ],
    :C2 => [ 2, 5, 11, 5, 2, 1, 4, 3, 1, 2, 0, 3, 2, 2, 0, 0, 0, 1, 0, 0 ],
    :C3 => [ 1, 1, 3, 1, 1, 0, 3, 4, 6, 4, 3, 4, 6, 4, 2, 6, 2, 1, 0, 1 ],
    :C4 => [ table6_1.E[i] in [ 9, 14 ] ? 1 : 0 for i in axes(table6_1, 1) ],
)

# Nc should be 102 
Nc = sum(sum([ getproperty(table6_1, Symbol("C$x")) for x in 1:4 ]))
#102

averageclutch = [ 
    mean(1:4, fweights([ getproperty(table6_1, Symbol("C$x"))[i] for x in 1:4 ])) 
    for i in axes(table6_1, 1) 
]

fig, ax = scatter(table6_1.E, averageclutch; color=:black, marker=:circle)
lines!(ax, [ 0, 10, 25 ], [ 2.089 + 0.0415 * x for x in [ 0, 10, 25 ] ]; color=:black)
scatter!(ax, zeros(1), zeros(1); markersize=0)  # to plot the origin
Label(fig.layout[2, 1], "Egg complement"; tellwidth=false)
Label(fig.layout[1, 0], "Average clutch"; rotation=π/2, tellheight=false)
fig 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 6.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
function sumsquares_cf(cf, df::DataFrame; C_vec=1:4)
    ssq = 0
    for (i, E) in enumerate(df.E), C in C_vec
        ssq += (C - cf)^2 * getproperty(df, Symbol("C$C"))[i]
    end
    return ssq
end

function sumsquares_cf(cf, e_vec, c_vec)
    ssq = 0
    for (E, C) in zip(e_vec, c_vec)
        ssq += (C - cf)^2
    end
    return ssq
end

ssqcf = [ sumsquares_cf(x, table6_1) / Nc for x in 1:4 ]
#4-element Vector{Float64}:
# 2.6862745098039214
# 0.6274509803921569
# 0.5686274509803921
# 2.5098039215686274


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 6.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function sumsquares_singleswitch(c1::Number, c2::Number, e1::Number, df::DataFrame; C_vec=1:4)
    ssq = 0
    for (i, E) in enumerate(df.E), C in C_vec
        if E <= e1 
            ce = c1 
        else 
            ce = c2 
        end
        ssq += (C - ce)^2 * getproperty(df, Symbol("C$C"))[i]
    end
    return ssq
end

function sumsquares_singleswitch(c1::Number, c2::Number, e1::Number, e_vec, c_vec)
    ssq = 0
    for (E, C) in zip(e_vec, c_vec)
        if E <= e1 
            ce = c1 
        else 
            ce = c2 
        end
        ssq += (C - ce)^2
    end
    return ssq
end

function sumsquares_singleswitch(c1s, c2s, e1s, data...; kwargs...)
    ssq_star, c1_star, c2_star, e1_star = Inf, c1s[1], c2s[1], e1s[1] 
    for c1 in c1s, c2 in c2s, e1 in e1s 
        ssq = sumsquares_singleswitch(c1, c2, e1, data...; kwargs...) 
        if ssq < ssq_star 
            ssq_star, c1_star, c2_star, e1_star = ssq, c1, c2, e1
        end
    end
    return @ntuple ssq_star c1_star c2_star e1_star
end

ssqss = sumsquares_singleswitch(0:4, 0:4, 0:23, table6_1)
#(ssq_star = 34, c1_star = 2, c2_star = 3, e1_star = 8)
ssqss.ssq_star / (Nc - 6)
#0.3541666666666667


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variable-clutch model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

allcombinations(v...) = vec(collect(Iterators.product(v...)))

function testascending(a)
    for i in 2:length(a)
        if a[i] < a[i-1]
            return false
        end
    end
    return true
end
        
function sumsquares_variableclutch(cs, es, df::DataFrame; C_vec=1:4)
    @assert length(cs) == length(es) + 1 "Must have one more clutch size than switching points"
    ssq = 0
    for (C, C_vals) in zip(C_vec, [ getproperty(df, Symbol("C$x")) for x in C_vec ])
        for (i, E) in enumerate(df.E)
            if E > maximum(es) 
                ce = last(cs)
            else
                ce = cs[findfirst(x -> x >= E, es)]
            end
            ssq += (C - ce)^2 * C_vals[i]
        end
    end
    return ssq
end

function sumsquares_variableclutch(cs, es, e_vec, c_vec)
    @assert length(cs) == length(es) + 1 "Must have one more clutch size than switching points"
    ssq = 0
    for (E, C) in zip(e_vec, c_vec)
        if E > maximum(es) 
            ce = last(cs)
        else
            ce = cs[findfirst(x -> x >= E, es)]
        end
        ssq += (C - ce)^2
    end
    return ssq
end

function sumsquares_variableclutch(potentialcs, potentiales, n_switches::Int, data...; kwargs...)
    cs_star = [ potentialcs[1] for _ in 1:(n_switches + 1) ]
    es_star = [ potentiales[1] for _ in 1:n_switches ]
    ssq_star = sumsquares_variableclutch(cs_star, es_star, data...; kwargs...)
    allcs = allcombinations([ potentialcs for _ in 1:(n_switches + 1) ]...) 
    for es in allcombinations([ potentiales for _ in 1:n_switches ]...)
        if testascending(es)
            for cs in allcs
                ssq = sumsquares_variableclutch(cs, es, data...; kwargs...)
                if ssq < ssq_star 
                    ssq_star, cs_star, es_star = ssq, cs, es
                end
            end
        end
    end
    return @ntuple ssq_star cs_star es_star
end

ssqvc2 = sumsquares_variableclutch(0:4, 0:23, 2, table6_1)
#(ssq_star = 34, cs_star = (0, 2, 3), es_star = (0, 8))
# Note that the first switch is at E = 0, so this is the same model as the one that only 
# switches once at E = 8
ssqvc2.ssq_star / (Nc - 10)
#0.3695652173913043

#ssqvc3 = sumsquares_variableclutch(0:4, 0:23, 3, table6_1)
#(ssq_star = 33, cs_star = (2, 3, 2, 3), es_star = (8, 9, 10))
#ssqvc3.ssq_star / (Nc - 12)
#0.36666666666666664


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure 6.3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sv = 1:23 
ssqprofile = [ sumsquares_singleswitch(0:4, 0:4, [ v ], table6_1).ssq_star for v in sv ]

fig, ax = scatter(sv, ssqprofile)
Label(fig.layout[2, 1], "Switching value"; tellwidth=false)
Label(fig.layout[1, 0], "Minimum sum of squares"; rotation=π/2, tellheight=false)

fig


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reformat data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

raw_E = zeros(Int, Nc)
raw_C = zeros(Int, Nc)

j = 1 
for (C, C_vec) in zip(1:4, [ getproperty(table6_1, Symbol("C$i")) for i in 1:4 ])
    for (i, E) in enumerate(table6_1.E)
        k = 0
        while k < C_vec[i]
            raw_E[j] = E 
            raw_C[j] = C 
            k += 1 
            j += 1 
        end
    end
end

# check that the functions used above work equivalently with this dataset 

ssqcf = [ sumsquares_cf(x, raw_E, raw_C) / Nc for x in 1:4 ]
#4-element Vector{Float64}:
# 2.6862745098039214
# 0.6274509803921569
# 0.5686274509803921
# 2.5098039215686274

ssqss = sumsquares_singleswitch(0:4, 0:4, 0:23, raw_E, raw_C)
#(ssq_star = 34, c1_star = 2, c2_star = 3, e1_star = 8)

ssqvc2 = sumsquares_variableclutch(0:4, 0:23, 2, raw_E, raw_C)
#(ssq_star = 34, cs_star = (0, 2, 3), es_star = (0, 8))

ssqvc3 = sumsquares_variableclutch(0:4, 0:23, 3, raw_E, raw_C)
#(ssq_star = 33, cs_star = (2, 3, 2, 3), es_star = (8, 9, 10))
ssqvc3.ssq_star / (Nc - 12)
#0.36666666666666664

# NB, the version with vectors is tremendously faster 
#@benchmark sumsquares_variableclutch(0:4, 0:23, 3, table6_1)
#BenchmarkTools.Trial: 1 sample with 1 evaluation.
# Single result which took 49.456 s (3.34% GC) to evaluate,       
# with a memory estimate of 20.30 GiB, over 648775577 allocations.

@benchmark sumsquares_variableclutch(0:4, 0:23, 3, raw_E, raw_C)
#BenchmarkTools.Trial: 7 samples with 1 evaluation.
# Range (min … max):  783.911 ms … 835.917 ms  ┊ GC (min … max): 4.08% … 3.75%
# Time  (median):     785.742 ms               ┊ GC (median):    4.07%
# Time  (mean ± σ):   794.116 ms ±  19.024 ms  ┊ GC (mean ± σ):  4.06% ± 0.23%
#
#  █▁█             ▁                                           ▁  
#  ███▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#  784 ms           Histogram: frequency by time          836 ms <
#
# Memory estimate: 381.15 MiB, allocs estimate: 8665647.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 6.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function bsdata(Eraw::Vector{S}, Craw::Vector{T}) where {S <: Number, T <: Number}
    @assert length(Eraw) == length(Craw)
    E_bs = zeros(S, length(Eraw))
    C_bs = zeros(T, length(Craw))
    for i in eachindex(E_bs)
        j = sample(1:length(Eraw))
        E_bs[i] = Eraw[j]
        C_bs[i] = Craw[j]
    end
    return @ntuple E_bs C_bs
end

cf2results = zeros(10_000)
cf3results = zeros(10_000)
vc2results = zeros(10_000)

seed!(63)
for i in 1:10_000 
    @unpack E_bs, C_bs = bsdata(raw_E, raw_C)
    cf2results[i] = sumsquares_cf(2, E_bs, C_bs) / Nc
    cf3results[i] = sumsquares_cf(3, E_bs, C_bs) / Nc
    vc2results[i] = sumsquares_singleswitch(0:4, 0:4, 0:23, E_bs, C_bs).ssq_star / (Nc - 6)
end

# count winners cf2results vs cf3results  
cf2wins = 0 
cf3wins = 0 
for i in 1:10_000 
    if cf2results[i] < cf3results[i] 
        cf2wins += 1 
    else
        cf3wins += 1 
    end
end
cf2wins 
#2834
cf3wins 
#7166

cf2wins = 0 
cf3wins = 0 
vc2wins = 0
for i in 1:10_000 
    if cf2results[i] < cf3results[i] && cf2results[i] < vc2results[i]
        cf2wins += 1 
    elseif cf3results[i] < vc2results[i]
        cf3wins += 1 
    else 
        vc2wins += 1
    end
end
cf2wins 
#0
cf3wins 
#1
vc2wins 
#9999
