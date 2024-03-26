
using DrWatson
@quickactivate "EcologicalDetective.jl"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chapter 7
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using CairoMakie
using Distributions: Normal, Poisson
using Random: seed!
using StatsBase: sample

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure 7.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

poislhood(r, k) = exp(-r) * r^k / factorial(k)

r_vec = 0:0.1:16
poislhood_k4 = [ poislhood(r, 4) for r in r_vec ]
poislhood_k6 = [ poislhood(r, 6) for r in r_vec ]

fig_71 = let 
    fig = Figure()
    ax1 = Axis(fig[1, 1])
    lines!(ax1, r_vec, poislhood_k4; label="k = 4")
    lines!(ax1, r_vec, poislhood_k6; label="k = 6")
    
    ax2 = Axis(fig[2, 1])
    lines!(ax2, r_vec, poislhood_k4 ./ poislhood(4, 4))
    lines!(ax2, r_vec, poislhood_k6 ./ poislhood(6, 6))
    
    ax3 = Axis(fig[3, 1])
    lines!(ax3, r_vec, -log.(poislhood_k4))
    lines!(ax3, r_vec, -log.(poislhood_k6))
    
    Label(fig[4, 1], "r"; tellwidth=false)
    Label(fig[1, 0], "Likelihood"; rotation=π/2, tellheight=false)
    Label(fig[2, 0], "Likelihood\nratio"; rotation=π/2, tellheight=false)
    Label(fig[3, 0], "Negative\nlog-likelihood"; rotation=π/2, tellheight=false)
    Legend(fig[0, 1], ax1; orientation=:horizontal)
    linkxaxes!(ax1, ax2, ax3)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 4, 5)
    
    fig        
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure 7.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

normlhood(y, m, σ) = product([ exp(-(yi - m)^2 / (2 * σ^2)) / (σ * sqrt(2π)) for yi in y ])

function normnegloglhood(y, m, σ)
    n = length(y)
    L = n * (log(σ) + 0.5 * log(2π)) + sum([ (yi - m)^2 / (2 * σ^2) for yi in y ])
    return L 
end

heights = [ 171, 168, 180, 190, 169, 172, 162, 181, 181, 177 ]

m_vec = 160:0.1:195

fig_72 = let 
    fig = Figure() 
    ax = Axis(fig[1, 1])

    for obs in [ 2, 4, 7, 10 ]
        nnll = [ normnegloglhood(heights[1:obs], m, 10) for m in m_vec ]
        lines!(ax, m_vec, nnll .- minimum(nnll); label="$obs observations")
    end

    Label(fig[2, 1], "Mean height (cm)"; tellwidth=false)
    Label(fig[1, 0], "Negative log-likelihood"; rotation=π/2, tellheight=false)
    Legend(fig[3, 1], ax; orientation=:horizontal)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)

    fig
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function N_tplus1(N; C, r, K, σ_w)  # where C is the fish caught
    W_t = generate_Wnoise(σ_w)
    return W_t * deterministicN_tplus(N; C, r, K)
end

deterministicN_tplus(N; C, r, K) = N + r * N * (1 - N / K) - C

N_obs(N; σ_v) = N * generate_Wnoise(σ_v)
generate_Wnoise(σ_w) = exp(rand(Normal(0, 1)) * σ_w - σ_w^2 / 2)

Ns = zeros(11)
observedNs = zeros(11)
seed!(71)
for i in eachindex(Ns)
    if i == 1 
        Ns[i] = 1000 
    elseif i == 2 
        Ns[i] = N_tplus1(Ns[i-1]; C=0, r=0.5, K=1000, σ_w=0.1)
    elseif i <= 5 
        Ns[i] = N_tplus1(Ns[i-1]; C=(0.5 * Ns[i-1]), r=0.5, K=1000, σ_w=0.1)
    else
        Ns[i] = N_tplus1(Ns[i-1]; C=(0.01 * Ns[i-1]), r=0.5, K=1000, σ_w=0.1)
    end
    observedNs[i] = N_obs(Ns[i], σ_v=0.1)
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fishingpattern = [ [ 0, 0 ]; [0.5 for _ in 1:3 ]; [ 0.01 for _ in 1:5 ] ]

function deterministicNs(N0; fishing::Vector, r, K)
    Ns = zeros(length(fishing) + 1)
    Ns[1] = K 

    for i in eachindex(Ns)
        i == 1 && continue 
        Ns[i] = deterministicN_tplus(Ns[i-1]; C=(fishing[i-1] * Ns[i-1]), r, K)
    end

    return Ns 
end 

function deviationNs(Ns; fishing::Vector, r, K, σ_v)
    detNs = deterministicNs(K; fishing, r, K)
    #return Ns .- detNs 
    return [ log(Ns[i]) - log(detNs[i]) + σ_v^2 / 2 for i in eachindex(Ns) ]
end

function negloglhood_deviationNs(Ns; fishing, r, K, σ_v)
    ds = deviationNs(Ns; fishing, r, K, σ_v)
    L_t = [ log(σ_v) + 0.5 * log(2π) + d^2 / (2 * σ_v^2) for d in ds ] 
    return sum(L_t)
end

rs = 0:0.01:1
Ks = 1:1:10_000
r_star = rs[1]
K_star = Ks[1] 
minnegloglhood = Inf 

for r in rs, K in Ks 
    nll = negloglhood_deviationNs(observedNs; fishing=fishingpattern, r, K, σ_v=0.1)
    if nll < minnegloglhood
        minnegloglhood = nll 
        r_star = r 
        K_star = K 
    end
end

r_star
#0.91 
K_star
#802

detNs = deterministicNs(K_star; fishing=fishingpattern, r=r_star, K=K_star)

fig7_4 = let 
    fig, ax = scatter(0:10, observedNs)
    scatter!(ax, 0, 0; markersize=0)
    lines!(ax, 0:10, detNs; color=:black)
    Label(fig[2, 1], "Time"; tellwidth=false)
    Label(fig[1, 0], "Abundance"; rotation=π/2, tellheight=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    fig    
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function predictedNs(Ns; fishing::Vector, r, K, σ_w)
    predNs = zeros(length(Ns))
    predNs[1] = Ns[1]

    for i in eachindex(predNs)
        i == 1 && continue 
        predNs[i] = deterministicN_tplus(Ns[i-1]; C=(fishing[i-1] * Ns[i-1]), r, K) 
    end

    #return Ns 
    return [ max(0, N) for N in predNs ]
end 

function deviationNs_73(Ns; fishing::Vector, r, K, σ_w)
    predNs = predictedNs(Ns; fishing, r, K, σ_w)
    return [ i == 1 ? 0 : log(predNs[i]) - log(Ns[i]) + σ_w^2 / 2 for i in eachindex(Ns) ]
end

function negloglhood_deviationNs_73(Ns; fishing, r, K, σ_w)
    ds = deviationNs_73(Ns; fishing, r, K, σ_w)
    L_t = [ log(σ_w) + 0.5 * log(2π) + d^2 / (2 * σ_w^2) for d in ds ] 
    return sum(L_t)
end

rs = 0:0.01:1
Ks = 1:1:10_000
r_star = rs[1]
K_star = Ks[1] 
minnegloglhood = Inf 

for r in rs, K in Ks 
    nll = negloglhood_deviationNs_73(observedNs; fishing=fishingpattern, r, K, σ_w=0.1)
    if nll < minnegloglhood
        minnegloglhood = nll 
        r_star = r 
        K_star = K 
    end
end

r_star
#0.96
K_star
#767

predNs = predictedNs(Ns; fishing=fishingpattern, r=r_star, K=K_star, σ_w=0.1)

fig7_5 = let 
    fig, ax = scatter(0:10, observedNs)
    scatter!(ax, 0, 0; markersize=0)
    lines!(ax, 1:10, predNs[2:end]; color=:black)
    Label(fig[2, 1], "Time"; tellwidth=false)
    Label(fig[1, 0], "Abundance"; rotation=π/2, tellheight=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    fig    
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

indexofabundance(D; p=0, q=0, r=0) = max(0, (p + q * D) / (1 + r * D))

densts = 1:20 
abundanceindex = [ indexofabundance(D; p=-3, q=1, r=0.03) for D in densts ]
seed!(74)
observations = [ rand(Poisson(abi)) for abi in abundanceindex ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function minnegloglhood_indofabund(data, densts; p=zeros(1), q=zeros(1), r=zeros(1))
    p_star = p[1] 
    q_star = q[1]
    r_star = r[1]
    minnegloglhood = Inf 
    for p_i in p, q_i in q, r_i in r 
        nll = negloglhood_indofabund(data, densts; p=p_i, q=q_i, r=r_i)
        if nll < minnegloglhood
            p_star = p_i 
            q_star = q_i
            r_star = r_i
            minnegloglhood = nll 
        end
    end
    return @ntuple p_star q_star r_star minnegloglhood 
end

function negloglhood_indofabund(data, densts; p, q, r)
    I_pred = [ indexofabundance(D; p, q, r) for D in densts ]
    nll = sum([ -log(poislhood(I_pred[i], data[i])) for i in eachindex(data) ])
    return nll
end

tv = let 
    _tv = collect(0.1:0.01:10)
    tv = [ -_tv; collect(-0.1:0.0001:0.1); _tv]
    tv
end

#=
textbookobs = [ 0, 0, 0, 2, 0, 4, 4, 5, 2, 6, 6, 13, 9, 9, 6, 10, 6, 11, 15, 15 ]

# model A 
modelaresults = minnegloglhood_indofabund(textbookobs, densts; q=tv)

# model B
modelbresults = minnegloglhood_indofabund(textbookobs, densts; p=tv, q=tv)

# model C
modelcresults = minnegloglhood_indofabund(textbookobs, densts; q=tv, r=tv)

# model D
modeldresults = minnegloglhood_indofabund(
    textbookobs, densts; 
    p=-10:0.1:10, q=-10:0.1:10, r=-0.1:0.0001:0.1
)

=#

# and with my values from 7.4: 

# model A 
modelaresults = minnegloglhood_indofabund(observations, densts; q=tv)

# model B
modelbresults = minnegloglhood_indofabund(observations, densts; p=tv, q=tv)

# model C
modelcresults = minnegloglhood_indofabund(observations, densts; q=tv, r=tv)

# model D
modeldresults = minnegloglhood_indofabund(
    observations, densts; 
    p=-10:0.1:10, q=-10:0.1:10, r=-0.1:0.0001:0.1
)

# for my data, the "best" is model A, q* = 0.47
fittedabundance = [ indexofabundance(D; p=0, q=0.47, r=0) for D in densts ]

fig_77 = let 
    fig, ax = scatter(densts, observations)
    lines!(ax, densts, abundanceindex; color=:black, linestyle=:dash)
    lines!(ax, densts, fittedabundance; color=:black)
    Label(fig[2, 1], "True abundance"; tellwidth=false)
    Label(fig[1, 0], "Number counted"; rotation=π/2, tellheight=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    fig
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

f_MLE(S, I) = I / S 

function nll_MLE(S, I)
    f = f_MLE(S, I)
    return nll_binom(S, I, f)
end

nll_binom(S, I, f) = -I * log(f) - (S - I) * log(1 - f)

function nll_binom_upperb(S, I)
    L_MLE = nll_MLE(S, I)
    target = L_MLE + 1.92 
    f = f_MLE(S, I)  # starting conditions 
    result = nll_binom(S, I, f)
    while result < target 
        f += 0.001 
        result = nll_binom(S, I, f)
    end
    return @ntuple f result target
end 

f_1 = f_MLE(20, 2)
fb_1 = nll_binom_upperb(20, 2)

f_2 = f_MLE(40, 3)
fb_2 = nll_binom_upperb(40, 3)

f_3 = f_MLE(60, 4)
fb_3 = nll_binom_upperb(60, 4)

f_4 = f_MLE(80, 4)
fb_4 = nll_binom_upperb(80, 4)

f_5 = f_MLE(100, 4)
fb_5 = nll_binom_upperb(100, 4)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.7
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# with known values of r and p 

qs = 0:0.01:10

nll_true_rq = [ negloglhood_indofabund(observations, densts; p=-3, q, r=0.03) for q in qs ]

# unknown r and p 

nll_unknown_rq = [
    minnegloglhood_indofabund(
        observations, densts; p=-10:0.025:10, q, r=-1:0.01:1
    ).minnegloglhood 
    for q in qs 
]

fig_79 = let 
    # clip values at 40 
    inds1 = findall(x -> x <= 40, nll_true_rq)
    fig, ax = lines(qs[inds1], nll_true_rq[inds1]; color=:black, linestyle=:dash)
    inds2 = findall(x -> x <= 40, nll_unknown_rq)
    lines!(ax, qs[inds2], nll_unknown_rq[inds2])
    Label(fig[2, 1], "q"; tellwidth=false)
    Label(fig[1, 0], "Negative log-likelihood"; rotation=π/2, tellheight=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    fig
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudocode 7.8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function bootstrapabundancedata(
    densts::AbstractVector{S}, observations::AbstractVector{T}
) where {S <: Number, T <: Number}
    @assert length(densts) == length(observations)
    densts_bs = zeros(S, length(densts))
    obs_bs = zeros(T, length(observations))
    for i in eachindex(densts_bs)
        j = sample(1:length(densts))
        densts_bs[i] = densts[j]
        obs_bs[i] = observations[j]
    end
    return @ntuple densts_bs obs_bs
end

seed!(78)
bs_q_est = zeros(10_000)
for i in 1:10_000
    @unpack densts_bs, obs_bs = bootstrapabundancedata(densts, observations)
    bs_q_est[i] = minnegloglhood_indofabund(obs_bs, densts_bs; q=qs).q_star
end

cumbs_q_est = [ sum(bs_q_est .<= q) for q in qs ]

fig7_11 = let 
    fig, ax = hist(bs_q_est; bins=30)
    inds = findall(x -> 0 < x < 10_000, cumbs_q_est)
    lines!(ax, qs[inds], cumbs_q_est[inds] ./ 10; color=:black)
    Label(fig[2, 1], "q"; tellwidth=false)
    Label(fig[1, 0], "Frequency"; rotation=π/2, tellheight=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    fig
end

# version with a Poisson-distributed E 

function bootstrapabundancedata_distnuncertainty(
    densts::AbstractVector{S}, observations::AbstractVector{T}, E_mean
) where {S <: Number, T <: Number}
    @assert length(densts) == length(observations)
    densts_bs = zeros(S, length(densts))
    obs_bs = zeros(T, length(observations))
    for i in eachindex(densts_bs)
        j = sample(1:length(densts))
        densts_bs[i] = densts[j]
        #obs_bs[i] = observations[j] + rand(Poisson(E_mean))
        obs_bs[i] = rand(Poisson(observations[j]))
    end
    return @ntuple densts_bs obs_bs
end


seed!(781)
bs_q_est_du = zeros(10_000)
for i in 1:10_000
    @unpack densts_bs, obs_bs = bootstrapabundancedata_distnuncertainty(densts, observations, 1)
    if maximum(obs_bs) > 20 
        bs_q_est_du[i] = minnegloglhood_indofabund(big.(obs_bs), densts_bs; q=qs).q_star
    else
        bs_q_est_du[i] = minnegloglhood_indofabund(obs_bs, densts_bs; q=qs).q_star
    end
end

cumbs_q_est_du = [ sum(bs_q_est_du .<= q) for q in qs ]

fig7_12 = let 
    fig, ax = hist(bs_q_est_du; bins=30)
    inds = findall(x -> 0 < x < 10_000, cumbs_q_est_du)
    lines!(ax, qs[inds], cumbs_q_est_du[inds] ./ 10; color=:black)
    Label(fig[2, 1], "q"; tellwidth=false)
    Label(fig[1, 0], "Frequency"; rotation=π/2, tellheight=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    fig
end
