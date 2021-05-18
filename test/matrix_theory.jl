using MatrixHMC
using Revise, BenchmarkTools, Plots, Random, Statistics
using DataFrames, GLM
using FiniteDifferences
using MathLink, Optim
using StatsBase: autocor, tr
using LaTeXStrings

#include("variational_ON.jl")
include("helpers.jl")

m = 10.0 # mass
β = 10.0/m # total time
ω = 2*pi/β # momentum spacing
Λ = 2 # max momentum index
threePairs = fourierPairs(Λ, 3, 0) # k1 + k2 + k3 = 0
fourPairs = fourierPairs(Λ, 4, 0) # k1 + k2 + k3 + k4 = 0
threePairsK = map(k -> fourierPairs(Λ, 3, k), -Λ:Λ) # k1 + k2 + k3 + k4 = K
N = 2 # size of each matric
Ki = 9 # number of bosonic matrices
K = Ki+1 # num of bosonic + one gauge
g = 1.0 # YM coupling
kSize = 2*Λ+1 # total num of modes
# helper objects
freeFact = β*map(k -> ((ω*k)^2 + m^2), -Λ:Λ) # coefficient for free term
freeFactA = β*map(k -> (ω*k)^2, -Λ:Λ) # coefficient for free term in A gradient
# Calculate action:
function action(u, useDict=true)
    #   u = (X^i, A)
    X = u[1:Ki]
    A = u[K]
    #   freeterm = \sum_k (k^2+m^2) X^i_k X^i_{-k}
    #   commXiXi2term = ([X^i, X^i]^2)_0
    #   commAXi2term = ([A,X^i])_0
    #   dXiCommAXiterm = (dX^i[A,X])
    function free_term_i(Xi) 
        return sum(map(k-> freeFact[k]*real(tr(Xi[k]*Xi[k])), 1:kSize))
    end
    freeterm = 0.5*sum(map(free_term_i, X))
    commXiXi2term_arr = zeros(length(fourPairs))
    commAXi2term_arr = zeros(length(fourPairs))
    dXiCommAXiterm_arr = zeros(length(threePairs))
    kP = 1
    if g != 0
        # initialize dictionaries
        commXiXi2Dict = Dict()
        commAXi2Dict = Dict()
        dXiCommAXiDict = Dict()
        # define function for g^2 term
        function calc_g2term()
            for kp in fourPairs
                k1, k2, k3, k4 = map(k->abs(k)+Λ+1, kp)
                p = (1,[k1,k2,k3,k4])
                kcombos = gen_combos(p)
                if k1 == k2 || k3 == k4
                    commXiXi2term_arr[kP] = 0.0
                else
                    if haskey(commXiXi2Dict, p)
                        commXiXi2term_arr[kP] = commXiXi2Dict[p]
                    else
                        Xi_term_1 = sum(map(Xi->comm(Xi[k1],Xi[k2]), X))
                        if k1 == k3 && k3 == k4
                            Xi_term_2 = Xi_term_1
                        else
                            Xi_term_2 = sum(map(Xi->comm(Xi[k3],Xi[k4]), X))
                        end
                        commXiXi2term_arr_kP = real(tr(Xi_term_1*Xi_term_2))
                        if useDict
                            for kcombo in kcombos
                                commXiXi2Dict[kcombo] = kcombo[1]*commXiXi2term_arr_kP
                            end
                        end
                        commXiXi2term_arr[kP] = commXiXi2term_arr_kP
                    end
                end
                # calculate ([A,X^i]^2)_0 term
                if haskey(commAXi2Dict, p)
                    commAXi2term_arr[kP] = commAXi2Dict[p]
                else
                    AXi_term_1 = sum(map(Xi->comm(A[k1],Xi[k2]), X))
                    if k1 == k3 && k3 == k4
                        AXi_term_2 = sum(map(Xi->comm(A[k1],Xi[k2])^2, X))
                    else
                        AXi_term_2 = sum(map(Xi->comm(A[k1],Xi[k2])*comm(A[k3],Xi[k4]), X))
                    end
                    commAXi2term_arr_kP = real(tr(AXi_term_1*AXi_term_2))
                    if useDict
                        for kcombo in kcombos
                            commAXi2Dict[kcombo] = kcombo[1]*commAXi2term_arr_kP
                        end
                    end
                    commAXi2term_arr[kP] = commAXi2term_arr_kP
                    kP+=1
                end
            end
            g2_term = -sum(commXiXi2term_arr)/4-2*sum(commAXi2term_arr)
            return g2_term
        end
        g2_term = calc_g2term()
        kP = 1 # reset pair counter
        # define function for g^2 term
        function calc_gterm()
            for kp in threePairs
                k1, k2, k3 = map(k->abs(k)+Λ+1, kp)
                p = (1,[k1,k2,k3])
                kcombos = gen_combos(p)
                if haskey(dXiCommAXiDict, p)
                    dXiCommAXiterm_arr[kP] = dXiCommAXiDict[p]
                else
                    dXiCommAXiterm_arr_kP = real(tr(sum(map(Xi -> k1*Xi[k1]*comm(A[k2],Xi[k3]), X))))
                    if useDict
                        for kcombo in kcombos
                            dXiCommAXiDict[kcombo] = kcombo[1]*dXiCommAXiterm_arr_kP
                        end
                    end
                    dXiCommAXiterm_arr[kP] = dXiCommAXiterm_arr_kP
                    kP+=1
                end
            end
            return 2*sum(dXiCommAXiterm_arr)
        end
        g_term = calc_gterm()
    end
    return freeterm + β*g_term + β*g^2*g2_term
end
# test action
metric = MatrixMetric(N, Λ, K)
uTest = rand(metric)
action(uTest)
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
# benchMarkAction()
# save action into log density for AHMC
ℓπ(u) = -actiοn(u)
# Define gradient of log density
function ∂ℓπ∂u(u::AbstractVector)
    #   u = (X^i, A)
    #   freeterm = \sum_k (k^2+m^2) X^i_k X^i_{-k}
    #   commXiXi2term = ([X^i, X^i]^2)_0
    #   commAXi2term = ([A,X^i])_0
    #   dXiCommAXiterm = (dX^i[A,X])
    # unpack
    X = u[1:Ki]
    A = u[K]
    # return free term for each X^i
    function ∂ℓπ∂X()
        function freeTerm_i(Xi)
            return map(k -> freeFact[k].*Xi[k], 1:kSize)
        end
        # calculate free term for each X^i
        freeTerm = map(freeTerm_i, X)
        # initialize interaction term
        g2_term = map(kk->map(k->zeros(ComplexF64, N,N),1:kSize),1:Ki)
        # calculate interaction term
        # array of ([X^j,[X^i,X^j]])_K for K = -Λ,...Λ
        XjcommXiXj_K = map(k->map(kk->zeros(ComplexF64, N,N),1:Ki),1:kSize)
        # array of ([A,[X^i,A]])_K:
        AcommXiA_K = map(k->map(kk->zeros(ComplexF64, N,N),1:Ki),1:kSize)
        Threads.@threads for kp in threePairsK # kp = all 3-pairs adding up to K
            K_kp = sum(kp[1])+Λ+1
            for kpK in kp # kpK = one 3-pair adding up to K
                k1, k2, k3 = map(k->k+Λ+1, kpK) # unpack pair
                # calculate ([X^j_k1,[X^i_k2,X^j_k3]])_K
                XjcommXiXj_K[K_kp] += map(Xi->sum(map(Xj->comm(Xj[k1],comm(Xi[k2],Xj[k3])), X)), X) # X^i_k1 for each i
                # calculate ([A_k1,[X^i_k2,A_k3]])_0
                AcommXiA_K[K_kp] += map(Xi->comm(A[k1],comm(Xi[k2],A[k3])), X)
            end
        end
        # rearrange term
        XjcommXiXj = map(i->map(term->term[i], XjcommXiXj_K), 1:Ki)
        g2_term = XjcommXiXj + 
        return freeTerm - g^2*XjcommXiXj, XjcommXiXj
    end
    δSδX, XjcommXiXj_K = ∂ℓπ∂X()
    function ∂ℓπ∂A()
        freeTermA = freeFactA.*A
        return freeTermA
    end
    δSδA = ∂ℓπ∂A()
    # total gradient
    δSδu = vcat(δSδX, [δSδA])
    return -action(u), -δSδu, XjcommXiXj_K
end
# test gradient
gradtest = ∂ℓπ∂u(uTest)
# setup HMC
# Define a Hamiltonian system
metric = MatrixMetric(N, Λ, K)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂u)
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
