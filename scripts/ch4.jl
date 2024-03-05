
using DrWatson
@quickactivate "EcologicalDetective.jl"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chapter 4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using BenchmarkTools: @benchmark
using CairoMakie
using Distributions: cdf, pdf, Chisq, NegativeBinomial, Poisson
using Random: seed!
using SpecialFunctions: gamma
using StatsBase: fweights, mean, pweights, sample, std, var


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data and Table 4.2 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t42_tows = [ 237, 125, 240, 73, 70, 152 ]
t42_birds = [ 72, 36, 100, 8, 5, 15 ]
t42_birdspertow = t42_birds ./ t42_tows

mean_t42_birdspertow = mean(t42_birdspertow)
#0.2146943263453124

std_t42_birdspertow = std(t42_birdspertow)
#0.14080941904028482

mean_birdspertow = sum(t42_birds) / sum(t42_tows)
#0.2630992196209587


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Table 4.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t43_birds = 0:17 
t43_captures = [ 807, 37, 27, 8, 4, 4, 1, 3, 1, 0, 0, 2, 1, 1, 0, 0, 0, 1 ]

t43_totalcaptures = sum([ t43_captures[i] * t43_birds[i] for i ∈ eachindex(t43_captures) ])
#250
# NB, was 236 in Table 4.2

mean_t43_birdspertow = t43_totalcaptures / sum(t43_captures)
#0.2787068004459309
std_t43_birdspertow = std(t43_birds, fweights(t43_captures))
#1.2503392181367554


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Equation 4.7
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function p_c(c, k, m)
    f1 = gamma(k + c) / (gamma(k) * factorial(c))
    f2 = (k / (k + m))^k 
    f3 = (m / (m + k))^c 
    return f1 * f2 * f3 
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Equation 4.8 (b)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#=
    v = m + m^2 / k 
    v k = m k + m^2 
    k (v - m) = m^2 
    k = m^2 / (v - m)
=#
parameter_k(m, v) = m^2 / (v - m)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure 4.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t43_m = mean_t43_birdspertow
#0.2787068004459309
t43_k = parameter_k(mean_t43_birdspertow, std_t43_birdspertow^2)
#0.06046627723159262

fig4_1 = let 
    # aggregate values ≥ 8 
    _birds = 0:8 
    _capturesfrequency = [ 
        i <= 8 ? 
            t43_captures[i] : 
            sum(t43_captures[9:end]) 
        for i ∈ eachindex(_birds) 
    ] / sum(t43_captures)

    # predicted catch
    _predictedcatch = [ p_c(x, t43_k, t43_m) for x in t43_birds ]
    _predictedcatchagg = [ 
        i <= 8 ? 
        _predictedcatch[i] : 
            sum(_predictedcatch[9:end]) 
        for i ∈ eachindex(_birds) 
    ]  # these are already probabilities so do not need to divide by the total
    _predictedcapturesfrequencydifference = _predictedcatchagg .- _capturesfrequency

    # used Distributions.Poisson
    _predictedcatch2 = [ pdf(Poisson(t43_m), x) for x in t43_birds ]
    _predictedcatchagg2 = [ 
        i <= 8 ? 
        _predictedcatch2[i] : 
            sum(_predictedcatch2[9:end]) 
        for i ∈ eachindex(_birds) 
    ] 
    _predictedcapturesfrequencydifference2 = _predictedcatchagg2 .- _capturesfrequency

    fig = Figure()
    axs = [ Axis(fig[i, j]) for i in 1:2, j in 1:3 ]
    barplot!(axs[1, 1], _birds, _capturesfrequency)
    delete!(axs[2, 1])
    barplot!(axs[1, 2], _birds, _predictedcatchagg)
    scatter!(axs[2, 2], _birds, _predictedcapturesfrequencydifference)
    barplot!(axs[1, 3], _birds, _predictedcatchagg2)
    scatter!(axs[2, 3], _birds, _predictedcapturesfrequencydifference2)

    linkaxes!(axs[1, :]...)

    Label(fig.layout[0, 1], "Observed data"; fontsize=11.84, halign=:left, tellwidth=false)    
    Label(fig.layout[0, 2], "Predicted data"; fontsize=11.84, halign=:left, tellwidth=false)    
    Label(fig.layout[0, 3], "Poisson"; fontsize=11.84, halign=:left, tellwidth=false)    
    Label(fig.layout[1, 0], "Frequency"; fontsize=11.84, rotation=π/2, tellheight=false)
    Label(
        fig.layout[2, 1], "Residual"; 
        fontsize=11.84, halign=:right, rotation=π/2, tellheight=false, tellwidth=false
    )
    Label(fig.layout[3, 1:3], "Albatrosses caught"; fontsize=11.84, tellwidth=false)    

    fig
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Equation 4.9
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# use predicted values rather than proportions 
modelchisq = sum(
    [ 
        (t43_captures[i] - p_c(t43_birds[i], t43_k, t43_m) * sum(t43_captures))^2 / 
        (p_c(t43_birds[i], t43_k, t43_m) * sum(t43_captures))
        for i in eachindex(t43_birds)
    ]
)
#22.04171977027245
1 - cdf(Chisq(17), modelchisq)
#0.18311991466364663

# note that `p_c(t43_birds[i], t43_k, t43_m)` becomes very small but is never 0 so function
# can run without grouping or omitting values (both of which were necessary in the text) but
# this gives very large values for chi squared


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 4.1 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Step 2 (a single simulation)

function probabilities_C_ij(k, m; cmax=17)
    probC_ij_equals_x = [ p_c(x, k, m) for x in 0:cmax ]
    # add a final value for "greater than cmax" 
    finalprob = 1 - sum(probC_ij_equals_x)
    push!(probC_ij_equals_x, finalprob)
    return probC_ij_equals_x
end

# also equip the function to use a Poisson distribution 
probabilities_C_ij_pois(k, m; cmax=17) = [ pdf(Poisson(m), x) for x in 0:(cmax + 1) ]

function sample_C_ij(k, m; cmax=17, probfunction=probabilities_C_ij)
    probC_ij_equals_x = probfunction(k, m; cmax)
    predC_ij = sample(0:(cmax + 1), pweights(probC_ij_equals_x))
    return predC_ij
end

# repeat Step 2 a total of Ntow times 

sample_C_ijs(Ntow, k, m; kwargs...) = [ sample_C_ij(k, m; kwargs...) for _ in 1:Ntow ]

## Step 3 

function calculatemeanandvariance(Ntow, k, m; kwargs...)
    C_ijs = sample_C_ijs(Ntow, k, m; kwargs...)
    M_j = sum(C_ijs) / Ntow
    S_j2 = sum([ (C_ij - M_j)^2 for C_ij in C_ijs ]) / (Ntow - 1)
    return @ntuple M_j S_j2
end

## Step 4 

function calculaterange(Ntow, k, m; tq=1.645, kwargs...)
    @unpack S_j2 = calculatemeanandvariance(Ntow, k, m; kwargs...)
    S_j = sqrt(S_j2)
    r_j = 2 * S_j * tq / sqrt(Ntow)
    return r_j 
end

## Steps 1, 5 and 6  

function countsuccesses(Nsim, d, Ntow, k, m; kwargs...)
    successes = 0
    for j in 1:Nsim  # `j` for consistency with the text, but not used in the code 
        r_j = calculaterange(Ntow, k, m; kwargs...)
        if r_j < d successes += 1 end
    end
    return successes
end

function estimateprogrammesuccess(Ntow; Nsim, d, k, m, kwargs...)
    successes = countsuccesses(Nsim, d, Ntow, k, m; kwargs...)
    return successes / Nsim 
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure 4.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Note that the function takes longer as Ntow increases 

@benchmark estimateprogrammesuccess(10; Nsim=150, d=(t43_m / 4), k=t43_k, m=t43_m)
#BenchmarkTools.Trial: 1738 samples with 1 evaluation.
# Range (min … max):  2.229 ms …  15.319 ms  ┊ GC (min … max): 0.00% … 0.00%
# Time  (median):     2.669 ms               ┊ GC (median):    0.00%
# Time  (mean ± σ):   2.868 ms ± 676.661 μs  ┊ GC (mean ± σ):  0.48% ± 3.43%
#
#      ▄▁█▄         ▂
#  ▇▅▆█████▇▆▄▄▄▃▄▆▄█▇▅▄▃▃▂▂▂▃▂▂▂▂▂▁▁▂▂▂▁▂▁▁▂▂▂▂▂▁▂▁▂▁▂▂▂▁▁▁▂▂ ▃
#  2.23 ms         Histogram: frequency by time        5.71 ms <
#
# Memory estimate: 393.88 KiB, allocs estimate: 3304.

@benchmark estimateprogrammesuccess(1000; Nsim=150, d=(t43_m / 4), k=t43_k, m=t43_m)
#BenchmarkTools.Trial: 15 samples with 1 evaluation.
# Range (min … max):  323.377 ms … 351.154 ms  ┊ GC (min … max): 0.00% … 0.00%
# Time  (median):     333.997 ms               ┊ GC (median):    0.58%
# Time  (mean ± σ):   334.763 ms ±   6.534 ms  ┊ GC (mean ± σ):  0.39% ± 0.33%
#
#  ▁          ▁  ▁ ▁▁▁   ▁▁▁▁   ▁   █         ▁                ▁  
#  █▁▁▁▁▁▁▁▁▁▁█▁▁█▁███▁▁▁████▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#  323 ms           Histogram: frequency by time          351 ms <
#
# Memory estimate: 36.66 MiB, allocs estimate: 300304.

# so plan to do fewer runs at greater numbers 

ntowvec = [ collect(1:1:99); collect(100:10:5000) ]

seed!(1)
chanceofsuccess = [ 
    estimateprogrammesuccess(ntow; Nsim=150, d=(t43_m / 4), k=t43_k, m=t43_m, tq=1.645) 
    for ntow in ntowvec
]

fig4_2, ax1 = lines(ntowvec, chanceofsuccess)

# add a line for the number of tows in the dataset 
vlines!(ax1, sum(t43_captures); color=:black, linestyle=:dot)

fig4_2

# Note that we cannot use the NegativeBinomial distribution for the summarized data as the 
# calculated k parameter is negative 

t42_m = mean_t42_birdspertow 
#0.2146943263453124
_t42_k = parameter_k(mean_t42_birdspertow, std_t42_birdspertow^2)
#-0.23653900227783614

# Instead use Poisson distribution 

seed!(2)
chanceofsuccesspois = [ 
    estimateprogrammesuccess(
        ntow; 
        Nsim=150, d=(t43_m / 4), k=_t42_k, m=t42_m, tq=1.645, 
        probfunction=probabilities_C_ij_pois
    ) 
    for ntow in ntowvec
]

let 
    ax2 = Axis(fig4_2[2, 1])
    lines!(ax2, ntowvec, chanceofsuccesspois)
    vlines!(ax2, sum(t43_captures); color=:black, linestyle=:dot)

    Label(fig4_2.layout[1:2, 0], "Chance of success"; fontsize=11.84, rotation=π/2, tellheight=false)
    Label(fig4_2.layout[3, 1], "Number of tows observed"; fontsize=11.84, tellwidth=false) 
    Label(fig4_2.layout[1, 2], "Negative binomial"; fontsize=11.84, rotation=-π/2, tellheight=false)
    Label(fig4_2.layout[2, 2], "Poisson"; fontsize=11.84, rotation=-π/2, tellheight=false)
end

fig4_2
