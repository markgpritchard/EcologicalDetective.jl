using DrWatson
@quickactivate "EcologicalDetective.jl"
using Random
using Distributions
using Plots

## Pseudocode 7.1 ##
# 0. Set seed
Random.seed!(1);

# 1. Set parameters
r = 0.5;
K = 1000;
sigmaW = 0.1;
sigmaV = 0.1;

# 2. set initial pop size
N0 = K;

# 3-5. Loop through 10 years and set the population size for each year
tvec = 1:1:10;
Ct = vcat(zeros(2),0.5 * ones(3), 0.01 * ones(5));
function logistic_equation(r, K, sigmaW, N0, tvec, Ct)
    Nt = vcat([N0],zeros(length(tvec)));
    Ndett = vcat([N0],zeros(length(tvec)));
    Wt = zeros(length(tvec));
    for tt in tvec
        Wt[tt] = exp(randn() * sigmaW - sigmaW^2 / 2)
        Nt[tt+1] = Wt[tt] * (Nt[tt] + r * Nt[tt] * (1 - Nt[tt] / K) - Ct[tt] * Nt[tt])
        Ndett[tt+1] = Ndett[tt] + r * Ndett[tt] * (1 - Ndett[tt] / K) - Ct[tt] * Ndett[tt]
    end
    return (Nt, Ndett)
end
function add_noise(Nt, sigmaV)
    return Nt .* exp.(randn(length(Nt)) * sigmaV .- sigmaV^2 / 2)
end
Nt, Ndett = logistic_equation(r, K, sigmaW, N0, tvec, Ct);
Nobst = add_noise(Nt,sigmaV);
Nobsdett = add_noise(Ndett,sigmaV);

# Plot it
scatter(tvec,Nt,label="process")
scatter!(tvec,Nobst,label="process + obs")
#scatter!(tvec,Ndett,label="deterministic")
#scatter!(tvec,Nobsdett,label="observation")

## Pseudocode 7.2
# 1. Data - Nobst from above
# 2. Loop over r and K and solve the deterministic equation
rvec = 0.1:0.1:0.9;
Kvec = 500:100:1500;
Lt_obs = zeros(length(rvec),length(Kvec));
for ii in 1:length(rvec)
    for jj in 1:length(Kvec)
        r = rvec[ii];
        K = Kvec[jj];
        Ndett = logistic_equation(r, K, sigmaW, N0, tvec, Ct)[2];
        Dt = log.(Nobst) .- log.(Ndett) .+ sigmaV^2 / 2;
        Lt = log(sigmaV) + 0.5 * log(2 * pi) .+ Dt.^2 .* 0.5 ./ sigmaV^2;
        Lt_obs[ii,jj] = sum(Lt);
    end
end
r = rvec[argmin(Lt_obs)[1]]
K = Kvec[argmin(Lt_obs)[2]]
scatter(vcat(0,tvec),Nobst,label="data")
plot!(vcat([0],tvec),logistic_equation(r, K, sigmaW, N0, tvec, Ct)[2],label="observation")

## Pseudocode 7.3
rvec = 0.1:0.1:0.9;
Kvec = 500:100:1500;
Lt_proc = zeros(length(rvec),length(Kvec));
for ii in 1:length(rvec)
    for jj in 1:length(Kvec)
        r = rvec[ii];
        K = Kvec[jj];
        Nt, Ndett = logistic_equation(r, K, sigmaW, N0, tvec, Ct);
        Dt = log.(Nt[2:end]) - log.(Nobst[1:(end-1)]) .+ sigmaW^2 / 2;
        Lt = log(sigmaW) + 0.5 * log(2 * pi) .+ Dt.^2 .* 0.5 ./ sigmaW^2;
        Lt_proc[ii,jj] = sum(Lt);
   end
end
r = rvec[argmin(Lt_proc)[1]]
L = Kvec[argmin(Lt_proc)[2]]
plot!(vcat([0],tvec),logistic_equation(r, K, sigmaW, N0, tvec, Ct)[1],label="process")


## Pseudocode 7.4
q = 1.0;
r = 0.03;
p = -3.0;
D = 1;
function calculateI(p,q,r,D)
    return maximum([0.0, (p + q * D) / (1 + r * D)])
end
Idet = zeros(21);
Idat = zeros(21);
for D in 1:21
    Idet[D] = calculateI(p,q,r,D);
    Idat[D] = rand(Poisson(Idet[D]),1)[1];
end
scatter(1:21,Idat,legend=false)
plot!(1:21,Idet)
