using DrWatson
@quickactivate "EcologicalDetective.jl"

using Distributions
using Plots
using DataFrames

# Write functions in a general way, so that can be applied to any model
function sum_of_squares(X,Yobs,param_mins,param_maxs,param_incs,run_model) 
    n = length(X);
    num_params = length(param_mins);
    # Check the data and Xi are same length, and the param vecs
    if length(Yobs) != n
        println("data and X are different lengths!");
        return 0;
    end
    if length(param_maxs) != num_params | length(param_maxs) != num_params
        println("param vecs not all same length");
        return 0;
    end 
    # First create array of parameters combinations (step 2)
    param_range_vecs = [];
    for i in 1:num_params
        push!(param_range_vecs,param_mins[i]:param_incs[i]:param_maxs[i]);
    end
    param_array = collect(Iterators.product(param_range_vecs...));
    # Loop over them
    Smin = Inf;
    param_min = param_array[1];
    for param in param_array
        Ypred = run_model(X,param);
        S = sum((Ypred .- Yobs).^2.0);
        if S < Smin;
            # Save the sum & param if less than the current minimum
            Smin = S;
            param_min = param;
        end
    end
    return (param_min, Smin);
end

function gen_fake_data_unif(X, param, Wmin, Wmax, run_model)
    out = run_model(X, param);
    noise = rand(Uniform(Wmin,Wmax),length(X));    
    return out .+ noise;
end    

function quadratic_model(X, param)
    out = param[1] .+ param[2] .* X .+ param[3] .* X.^2;
    return out;
end

function linear_model(X, param)
    out = param[1] .+ param[2] .* X;
    return out;
end

function constant_model(X, param)
    out = param[1];
    return out;
end

# Now an example
param_true = [1,0.5,0.25];
X = 1:10;
Wmin = -3; Wmax = 3;
# Gen deterministic output for these params
Ydet = quadratic_model(X,param_true);
# Gen noisy output for these params
Yobs = gen_fake_data_unif(X,param_true,Wmin,Wmax,quadratic_model);
# Fit the model
param_mins = [0,0,0];
param_maxs = [3,2,1];
param_incs = [0.1,0.05,0.025];
ssq_fit = sum_of_squares(X, Yobs, param_mins, param_maxs, param_incs, quadratic_model);
# Not the right value!
Ypred = quadratic_model(X,ssq_fit[1]);

# plot output
plot(X,Ydet,xlabel="X",ylabel="Y",linewidth=1.5,color="black",legend=:topleft,label="Ydet")
plot!(X,Ypred,color="red",linewidth=1.5,label="Ypred")
scatter!(X,Yobs,label="Yobs")

# Show output in table
DataFrame("X"=>X,
          "Y deterministic"=>Ydet,
          "Y observed"=>Yobs,
          "Y predicted"=>Ypred)


# Extra stuff
# goodness of fit profile
function goodness_of_fit_profiles(X,Yobs,param_mins,param_maxs,param_incs,run_model) 
    n = length(X);
    num_params = length(param_mins);
    # Check the data and Xi are same length, and the param vecs
    if length(Yobs) != n
        println("data and X are different lengths!");
        return 0;
    end
    if length(param_maxs) != num_params | length(param_maxs) != num_params
        println("param vecs not all same length");
        return 0;
    end 
    # Loop over the params, getting an profile for each, storing these in a vector
    profile_vecs = [];
    for i in 1:num_params
        param_of_interest_vec = param_mins[i]:param_incs[i]:param_maxs[i]
        profile_temp = zeros(Float64, length(param_of_interest_vec));
        for j in 1:length(param_of_interest_vec)
            # For this parameter value, get the other combinations
            param_range_vecs = [];
            for k in 1:num_params
                if k == i
                    push!(param_range_vecs,[param_of_interest_vec[j]])
                else
                    push!(param_range_vecs,param_mins[k]:param_incs[k]:param_maxs[k]);
                end
            end
            param_array_temp = collect(Iterators.product(param_range_vecs...));
            Smin = Inf;
            for param in param_array_temp
                Ypred = run_model(X,param);
                S = sum((Ypred - Yobs).^2.0);
                if S < Smin;
                    Smin = S;
                end
            end
            profile_temp[j] = Smin;
        end
        push!(profile_vecs,profile_temp)
    end
    return profile_vecs;
end

# Generate the profiles and plot
# (it has jagged lines unless increment is small)
# param_mins=[-3,0,0];
# param_incs=[0.01,0.01,0.01];
# param_maxs=[3,3,1];
Gprofiles = goodness_of_fit_profiles(X, Yobs, param_mins, param_maxs, param_incs, quadratic_model);
param_range_vecs = [];
for i in 1:length(param_mins);
    push!(param_range_vecs,param_mins[i]:param_incs[i]:param_maxs[i]);
end
plot(param_range_vecs,Gprofiles,layout=3,legend=false,xlabel="param",ylabel="Marginal SSQ",color="black",lwd=1.5,yaxis=:log)
println(ssq_fit)



# Model comparison 
function penalizedSSQ(X,Yobs,param_mins,param_maxs,param_incs,run_model)
    n = length(X);
    m = length(param_mins);
    return sum_of_squares(X,Yobs,param_mins,param_maxs,param_incs,run_model)[2] / (n - 2 * m);
end
penalizedSSQs_vec = [];

run_model = constant_model
param_mins_temp = param_mins[1:1];
param_maxs_temp = param_maxs[1:1];
param_incs_temp = param_incs[1:1];
out_temp = penalizedSSQ(X,Yobs,param_mins_temp,param_maxs_temp,param_incs_temp,constant_model);
push!(penalizedSSQs_vec,out_temp);

run_model = linear_model
param_mins_temp = param_mins[1:2];
param_maxs_temp = param_maxs[1:2];
param_incs_temp = param_incs[1:2];
out_temp = penalizedSSQ(X,Yobs,param_mins_temp,param_maxs_temp,param_incs_temp,linear_model);
push!(penalizedSSQs_vec,out_temp);

run_model = quadratic_model
param_mins_temp = param_mins[1:3];
param_maxs_temp = param_maxs[1:3];
param_incs_temp = param_incs[1:3];
out_temp = penalizedSSQ(X,Yobs,param_mins_temp,param_maxs_temp,param_incs_temp,quadratic_model);
push!(penalizedSSQs_vec,out_temp);

println(penalizedSSQs_vec)
