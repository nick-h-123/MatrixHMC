using MatrixHMC
using Revise, Plots, Random, Statistics
using DataFrames, GLM
using FiniteDifferences
using MathLink, Optim
using StatsBase: autocor, tr
using LaTeXStrings

#include("variational_ON.jl")
include("helpers.jl")

m = 10.0
β = 10.0/m
ω = 2*pi/β
Λ = 10
fourPairs = fourierPairs(Λ, 4, 0)
threePairsK = map(k -> fourierPairs(Λ, 3, k), -Λ:Λ)
N = 4
K = 9
g = 1.0
kSize = 2*Λ+1
# helper objects
freeFact = β*map(k -> ((ω*k)^2 + m^2), -Λ:Λ)
# action and gradient
function action(X, useDict=true)
    function free_term_i(Xi) 
        return sum(map(k-> freeFact[k]*real(tr(Xi[k]*Xi[k])), 1:kSize))
    end
    g_term_arr = zeros(length(fourPairs))
    kP = 1
    if g != 0
        kDict = Dict()
        #Threads.@threads 
        for kp in fourPairs
            k1, k2, k3, k4 = map(k->abs(k)+Λ+1, kp)
            #println(k1,", ", k2,", ", k3,", ", k4)
            if k1 == k2 || k3 == k4
                g_term_arr[kP] = 0.0
            else
                p = (1,[k1,k2,k3,k4])
                if haskey(kDict, p)
                    g_term_arr[kP] = kDict[p]
                else
                    Xi_term_1 = sum(map(Xi->comm(Xi[k1],Xi[k2]), X))
                    if k1 == k3 && k3 == k4
                        Xi_term_2 = Xi_term_1
                    else
                        Xi_term_2 = sum(map(Xi->comm(Xi[k3],Xi[k4]), X))
                    end
                    g_term_arr_kP = real(tr(Xi_term_1*Xi_term_2))
                    if useDict
                        kcombos = gen_combos(p)
                        for kcombo in kcombos
                            kDict[kcombo] = kcombo[1]*g_term_arr_kP
                        end
                    end
                    g_term_arr[kP] = g_term_arr_kP
                    kP+=1
                end
            end
        #j+=1
        end
    end
    g_term = sum(g_term_arr)
    return 0.5*sum(map(free_term_i, X)) - β*g^2*g_term
end
# test action
metric = MatrixMetric(N, Λ, K)
Xtest = rand(metric)
action(Xtest)
function benchMarkAction()
    println("Beginning benchmark using dictionary...")
    bmDataD = @benchmark action(Xtest, true)
    println("End benchmark using dictionary...")
    println("Beginning benchmark not using dictionary...")
    bmDataND = @benchmark action(Xtest, false)
    println("End benchmark using dictionary...")
    timesD, timesND = bmDataD.times, bmDataND.times
    println("Using Dict:")
    println("\tMean time = ", mean(timesD)/1e6, " ms")
    println("\tStandard deviation = ", std(timesD)/1e6, " ms")
    println("-----------------------------------------------------")
    println("Not using Dict:")
    println("\tMean time = ", mean(timesND)/1e6, " ms")
    println("\tStandard deviation = ", std(timesND)/1e6, " ms")
    println("-----------------------------------------------------")
    println("Performance ratio = ", mean(timesND)/mean(timesD))

    return
end
benchMarkAction()
# save action into log density for AHMC
ℓπ(X) = -actiοn(X)
# Define gradient of log density
function ∂ℓπ∂φ(X::AbstractVector)
    # return free term for each X^i
    function freeTerm_i(Xi)
        return map(k -> freeFact[k].*Xi[k], 1:kSize)
    end
    # calculate free term for each X^i
    freeTerm = map(freeTerm_i, X)
    # initialize interaction term
    g_term = map(kk->map(k->zeros(ComplexF64, N,N),1:kSize),1:K)
    # calculate interaction term
    g_term_K = map(k->map(kk->zeros(ComplexF64, N,N),1:K),1:kSize)
    Threads.@threads for kp in threePairsK # kp = all 3-pairs adding up to K
        K_kp = sum(kp[1])+Λ+1
        #println(K_kp)
        for kpK in kp # kpK = one 3-pair adding up to K
            k1, k2, k3 = map(k->k+Λ+1, kpK) # unpack pair
            #println(k1,", ",k2,", ",k3)
            Xi_term = map(Xi->Xi[k1], X) # X^i_k1 for each i
            #println(Xi_term)
            Xj2_term = sum(map(Xj->comm(Xj[k2],Xj[k3]), X))
            #println(map(Xi_term_k -> Xi_term_k*Xj2_term, Xi_term))
            g_term_K[K_kp] += map(Xi_term_k -> Xi_term_k*Xj2_term, Xi_term)
        end
    end
    g_term = map(i->map(gik->gik[i], g_term_K), 1:K)
    return -action(X), -(freeTerm - g^2/3*g_term)
end
# test gradient
gradtest = ∂ℓπ∂φ(Xtest)
# setup HMC
# Define a Hamiltonian system
metric = MatrixMetric(N, Λ, K)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂φ)
# Define a leapfrog solver, with initial step size chosen heuristically
initial_X = rand(metric)
rng = MersenneTwister(1234)
initial_ϵ = find_good_stepsize(rng, hamiltonian, initial_X)
# Define integrator
integrator = Leapfrog(initial_ϵ)
# Define an HMC sampler
# Set number of leapfrog steps
n_steps = 2
# Set the number of warmup iterations
proposal = StaticTrajectory(integrator, n_steps)
#adaptor =  StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))
adaptor = StepSizeAdaptor(0.7, integrator)
# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
progmeter=true
verb=true
#MersenneTwister(1234), 
n_samples = 300
n_adapts = 300
throwaway = 300
Xs, stats = sample(hamiltonian, proposal, initial_X, 
                   n_samples+throwaway, adaptor, n_adapts,
                   progress=progmeter, verbose=verb);
function getData(Xs, throwaway)
    function twopt_0(Xt)
        bs = sum(map(Xtk->map(k -> real(tr(Xtk[Λ+1+k]*Xtk[Λ+1+k])), 0:Λ), Xt))
        twopt0 = bs[1] + 2*sum(bs[2:end])
        # calculate coefficients for extensions
        d1 = bs[Λ+1]*Λ^2
        d2 = 0
        # extend sum
        n_ext = 1000 # how long to continue extension (could be automated in future)
        twopt0_ext = twopt0 + extendsum(0, d1, d2, β, n_ext)
        return m*twopt0_ext/(K*N^2)
    end
    Xkeeps = Xs[throwaway+1:end]
    #twopt_data = map(twopt_0, Xkeeps)
    #twopt_0s = map(x->x[1], twopt_data)
    #twopt_0_exts = map(x->x[2], twopt_data)
    twopt_0s = map(twopt_0, Xkeeps)
    twopt_0s_ave = running_ave(twopt_0s)
    twopts_0_ave = expectation_value(twopt_0, Xkeeps, 0.01)
    return twopts_0_ave, twopt_0s, twopt_0s_ave
end
function extendsum(t, d1, d2, β, Λmax=0)
    if Λmax == 0 Λmax = max(Int(Λ*10), 1000) end
    return sum(map(n -> 2*(d1/n^2+d2/n^4)*cos(n*(2*pi/β)*t), Λ+1:Λmax))
end
# generate data
twopt_0, twopt_0s, twopt_0s_ave = getData(Xs, throwaway)
println("mK<X^i(0)X^i(0)>/N^2 = ", round.(twopt_0[1], sigdigits=5))
println("Effective sample size = ", n_samples/twopt_0[3])
plot(twopt_0s, label="value",
     linewidth=1)
plot!(twopt_0s_ave, label="mean",
      title=string("Equal time 2-point: ", "N = ", N, ", g = ", g),
      linewidth=2)
xlabel!("Number of samples")
display(ylabel!(L"\frac{9 m}{N^2} \langle tr \left[X^i(0) X^i(0)\right] \rangle"))
#savefig("twopt_MT.png")
