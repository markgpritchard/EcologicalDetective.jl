
using DrWatson
@quickactivate "EcologicalDetective.jl"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chapter 10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using BenchmarkTools
using CairoMakie
using DataFrames
using Distributions: cdf, Chisq

#using Distributions: Normal, Poisson
#using Random: seed!
#using StatsBase: sample

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 10.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## 1. Input the data

data = DataFrame(
    :Year => 1965:1:1987,
    :CPUE => [ 
        1.78, 1.31, 0.91, 0.96, 0.88, 0.9, 0.87, 0.72, 0.57, 0.45, 0.42, 0.42, 0.49, 0.43, 
        0.4, 0.45, 0.55, 0.53, 0.58, 0.64, 0.66, 0.65, 0.63 
    ],
    :Catch => [
        94, 212, 195, 383, 320, 402, 366, 606, 378, 319, 309, 389, 277, 254, 170, 97, 91, 
        177, 216, 229, 211, 231, 223
    ]
)

## 2. Starting estimates 
# will be given below

## 3. Functions to minimize negative log likelihood 

function estimate_B_est(C_vector, r, K)
    B_est_vector = zeros(length(C_vector))
    for i in eachindex(C_vector)
        if i == 1 
            B_est_vector[i] = K 
        else
            B_est_vector[i] = estimate_B_est_tplus1(B_est_vector[i-1], C_vector[i-1], r, K)
        end
    end
    return B_est_vector 
end

estimate_B_est_tplus1(B_est, C_t, r, K) = B_est + r * B_est * (1 - B_est / K) - C_t
estimate_I_est(B_est_vector, q) = q .* B_est_vector 

# CPUE taken as true index of abundance 

function estimate_negloglikelihood(
    I_est_vector::Vector{<:Number}, I_vector::Vector{<:Number}, σ_V
)
    return [ 
        estimate_negloglikelihood(I_est, I_t, σ_V) 
        for (I_est, I_t) in zip(I_est_vector, I_vector)
    ]
end

function estimate_negloglikelihood(I_est::Number, I_t::Number, σ_V)
    return log(σ_V) + 0.5 * log(2π) + (log(I_est) - log(I_t))^2 / (2 * σ_V^2)
end

# calculate total values across all years 

function calc_negloglikelihood(C_vector, I_vector, r, K, q, σ_V)
    B_est_vector = estimate_B_est(C_vector, r, K)
    if minimum(B_est_vector) < 0 
        return Inf
    end
    
    I_est_vector = estimate_I_est(B_est_vector, q)
    L_vector = estimate_negloglikelihood(I_est_vector, I_vector, σ_V)
    return sum(L_vector)
end

function minimize_negloglikelihood( ;
    C_vector, I_vector, r_vector, K_vector, q_vector, σ_V_vector
)
    r_star = r_vector[1]
    K_star = K_vector[1]
    q_star = q_vector[1]
    σ_V_star = σ_V_vector[1]
    min_nLL = calc_negloglikelihood(C_vector, I_vector, r_star, K_star, q_star, σ_V_star)
    for r in r_vector, K in K_vector, q in q_vector, σ_V in σ_V_vector
        nLL = calc_negloglikelihood(C_vector, I_vector, r, K, q, σ_V)
        if nLL < min_nLL 
            r_star = r
            K_star = K
            q_star = q
            σ_V_star = σ_V
            min_nLL = nLL
        end 
    end 
    return @ntuple r_star K_star q_star σ_V_star min_nLL
end

## Calculate MSY 
Schaefer_calculateMSY(r, K) = r * K / 4

minimize_negloglikelihood(
    C_vector=data.Catch, 
    I_vector=data.CPUE, 
    r_vector=0.01:0.01:1, 
    K_vector=100:100:10_000, 
    q_vector=0.00001:0.00001:0.001, 
    σ_V_vector=0.01:0.01:1
)
#(r_star = 0.4, K_star = 2700, q_star = 0.00044, σ_V_star = 0.12, min_nLL = -15.24874024575675)

MSY = Schaefer_calculateMSY(r_star, K_star)
#270.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 10.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function minimizing_q(I_vector, B_est_vector)
    n = length(B_est_vector)
    expsum = sum([ log(I_t) - log(B_est) for (I_t, B_est) in zip(I_vector, B_est_vector) ])
    return exp(expsum / n)
end

# an alternative version of `calc_negloglikelihood` that uses equation 10.19. It keeps the
# same function name but does not accept a q input

function calc_negloglikelihood(C_vector, I_vector, r, K, σ_V)
    B_est_vector = estimate_B_est(C_vector, r, K)
    
    if minimum(B_est_vector) < 0 
        q = NaN
        nLL = Inf
        return @ntuple nLL q
    end

    q = minimizing_q(I_vector, B_est_vector)
    I_est_vector = estimate_I_est(B_est_vector, q)
    L_vector = estimate_negloglikelihood(I_est_vector, I_vector, σ_V)
    nLL = sum(L_vector)
    return @ntuple nLL q
end

function minimize_negloglikelihood_calcq( ;
    C_vector, I_vector, r_vector, K_vector, σ_V_vector
)
    r_star = r_vector[1]
    K_star = K_vector[1]
    σ_V_star = σ_V_vector[1]
    @unpack q, nLL = calc_negloglikelihood(C_vector, I_vector, r_star, K_star, σ_V_star)
    qstar = q
    min_nLL = nLL
    for r in r_vector, K in K_vector, σ_V in σ_V_vector
        @unpack q, nLL = calc_negloglikelihood(C_vector, I_vector, r, K, σ_V)
        if nLL < min_nLL 
            r_star = r
            K_star = K
            q_star = q
            σ_V_star = σ_V
            min_nLL = nLL
        end 
    end 
    return @ntuple r_star K_star q_star σ_V_star min_nLL
end

## Compare this function to the original one 

@benchmark minimize_negloglikelihood(
    C_vector=data.Catch, 
    I_vector=data.CPUE, 
    r_vector=0.01:0.01:1, 
    K_vector=100:100:10_000, 
    q_vector=0.00001:0.00001:0.001, 
    σ_V_vector=0.01:0.01:1
)
#BenchmarkTools.Trial: 1 sample with 1 evaluation.
# Single result which took 66.633 s (1.11% GC) to evaluate,     
# with a memory estimate of 55.81 GiB, over 249680010 allocations.

@benchmark minimize_negloglikelihood_calcq(
    C_vector=data.Catch, 
    I_vector=data.CPUE, 
    r_vector=0.01:0.01:1, 
    K_vector=100:100:10_000, 
    σ_V_vector=0.01:0.01:1
)
#BenchmarkTools.Trial: 6 samples with 1 evaluation.
# Range (min … max):  879.279 ms … 892.188 ms  ┊ GC (min … max): 0.99% … 0.96%
# Time  (median):     884.495 ms               ┊ GC (median):    0.99%
# Time  (mean ± σ):   885.003 ms ±   4.728 ms  ┊ GC (mean ± σ):  1.00% ± 0.03%
#
#  █         █     █               █        █                  █  
#  █▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#  879 ms           Histogram: frequency by time          892 ms <
#
# Memory estimate: 742.77 MiB, allocs estimate: 3245217.
minimize_negloglikelihood_calcq(
    C_vector=data.Catch, 
    I_vector=data.CPUE, 
    r_vector=0.01:0.01:1, 
    K_vector=100:100:10_000, 
    σ_V_vector=0.01:0.01:1
)
#(r_star = 0.4, K_star = 2700, q_star = 0.00044, σ_V_star = 0.12, min_nLL = -15.248740261830834)

function calculateMSYuncertainty(; C_vector, I_vector, r_vector, K_vector, σ_V_vector)
    # save results in a DataFrame 
    ℓ = length(r_vector)
    results = DataFrame(
        :r => r_vector,
        :minK => Vector{Float64}(undef, ℓ),
        :minσ_V => Vector{Float64}(undef, ℓ),
        :minnLL => Vector{Float64}(undef, ℓ),
        :minMSY => Vector{Float64}(undef, ℓ),
        :chisq => Vector{Float64}(undef, ℓ),
    )

    for (i, r) in enumerate(r_vector)
        @unpack min_nLL, K_star, q_star, σ_V_star = minimize_negloglikelihood_calcq( ;
            C_vector, 
            I_vector, 
            r_vector=[ r ] ,
            K_vector, 
            σ_V_vector
        )
        MSY = Schaefer_calculateMSY(r, K_star)
        results.minK[i] = K_star
        results.minσ_V[i] = σ_V_star
        results.minnLL[i] = min_nLL
        results.minMSY[i] = MSY
    end

    min_minnLL = minimum(results.minnLL)
    for (i, nLL) in enumerate(results.minnLL)
        results.chisq[i] = cdf(Chisq(1), 2 * (nLL - min_minnLL))
    end

    return results
end

results = calculateMSYuncertainty(; 
    C_vector=data.Catch, 
    I_vector=data.CPUE, 
    r_vector=0.2:0.001:0.6, 
    K_vector=2000:10:4000, 
    σ_V_vector=0.01:0.01:1
)

fig10_4 = let 
    fig = Figure()
    axs = [ Axis(fig[2*i-1, 2*j-1]) for i in 1:3, j in 1:2 ]
    
    # for each figure, only plot values for which minnLL < -12 
    inds = findall(x -> x < -12, results.minnLL)

    for (i, xs) in enumerate([ results.r, results.minK, results.minMSY])
        for (j, ys) in enumerate([ results.minnLL, results.chisq ])
            lines!(axs[i, j], xs[inds], ys[inds])
        end

        hlines!(axs[i, 2], 0.95; color=:black, linestyle=:dot)
    end

    Label(
        fig.layout[1:5, 0], "Negative log likelihood"; 
        fontsize=11.84, rotation=π/2, tellheight=false
    )
    Label(
        fig.layout[1:5, 2], "χ² probability"; 
        fontsize=11.84, rotation=π/2, tellheight=false
    )

    for c in [ 1, 3 ] 
        Label(fig.layout[2, c], "r"; fontsize=11.84, tellwidth=false)
        Label(fig.layout[4, c], "k"; fontsize=11.84, tellwidth=false)
        Label(fig.layout[6, c], "MSY"; fontsize=11.84, tellwidth=false)
    end

    for c in [ 1, 3 ] colgap!(fig.layout, c, 5) end
    for r in [ 1, 3, 5 ] rowgap!(fig.layout, r, 5) end

    fig
end


