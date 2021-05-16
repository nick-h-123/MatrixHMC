using MatrixAdvancedHMC

using Revise, Plots, Random, Statistics
using DataFrames, GLM
using FiniteDifferences
using MathLink, Optim

#include("variational_ON.jl")
include("helpers.jl")

function massdata(twopt_func, β)
    tmax=β/(2*Λ+1)
    ts = tmax/10:tmax/100:(tmax+tmax/10)
    twopts = map(twopt_func, ts)
    if any(x->x<0, twopts)
        return (0,0)
    end
    log_twopt_func = t->log(twopt_func(t))
    d_log=map(t->forward_fdm(2,1,adapt=1)(log_twopt_func, t), ts)
    return -mean(d_log), std(d_log)
end
function twopt(t, bs, β)
    return bs[1] + sum(map(n -> 2*bs[n+1]*cos(n*(2*pi/β)*t), 1:length(bs)-1))
end
function extendsum(t, d1, d2, β, Λmax=0)
    if Λmax == 0 Λmax = max(Int(Λ*10), 1000) end
    return sum(map(n -> 2*(d1/n^2+d2/n^4)*cos(n*(2*pi/β)*t), Λ+1:Λmax))
end
function ensemble_data(phis, m, λ, β, throwaway=0)
    phisTA = phis[(throwaway+1):end]
    bs = map(sum, map(phisi -> map(phisiN -> (phisiN.*conj(phisiN))[Int((kSize-1)/2)+1:end], phisi), phisTA))
    bs = map(real, bs)
    n_samples = length(phisTA)
    # reshape bs to take averages
    bs_rs = zeros(Float64, (n_samples, Λ+1))
    for i = 1:n_samples
        bs_rs[i, :] = bs[i]
    end
    b_aves = mean(bs_rs, dims=1)
    twopts_0s = map(i -> twopt(0, bs[i], β), 1:n_samples)
    twopts_0s_ave = running_ave(twopts_0s)
    # extend the sum per Hanada
    bΛ = map(bs_i -> bs_i[Λ+1], bs)
    if λ != 0
        bmΛ = map(bs_i -> bs_i[(Λ+1)-1], bs)
        d1 = (bΛ*Λ^4-bmΛ*(Λ-1)^4)/(2*Λ-1)
        d2 = (-bΛ*Λ^2+bmΛ*(Λ-1)^2)*Λ^2*(Λ-1)^2/(2*Λ-1)
        d1_ave = (b_aves[Λ+1]*Λ^4-b_aves[Λ]*(Λ-1)^4)/(2*Λ-1)
        d2_ave = (-b_aves[Λ+1]*Λ^2+b_aves[Λ]*(Λ-1)^2)*Λ^2*(Λ-1)^2/(2*Λ-1)
    else
        d1 = bΛ*Λ^2
        d2 = bΛ*0
        d1_ave = b_aves[Λ+1]*Λ^2
        d2_ave = 0
    end
    twopts_0s_extended = twopts_0s + map(i -> extendsum(0, d1[i], d2[i], β, Int(round(m*1000))), 1:n_samples)
    twopts_0s_extended_ave = running_ave(twopts_0s_extended)
    # return <φ(0)φ(t)>
    twopt_t(t) = twopt(t, b_aves, β)
    twopt_t_extended(t) = twopt_t(t) + extendsum(t, d1_ave, d2_ave, β, Int(round(m*1000)))
    m_eff= massdata(twopt_t_extended, β)
    return twopts_0s_ave, twopts_0s_extended_ave, twopt_t, twopt_t_extended, (m, λ), m_eff, twopts_0s_extended
end
function getGuess(NN)
    function getGuessNN()
        guess = randn(ComplexF64, 2*Λ+1)
        return 0.5*(guess + conj(reverse(guess)))
    end
    return map(n->getGuessNN(), 1:NN)
end

function computeSamples(m, λ, NN, initial_φ, n_samples=1000, n_adapts=min(Int(n_samples*0.1), 1000); progmeter=false, verb=false)
    ω = 2*pi/β
    # helper objects
    freeFact = β*map(k -> ((ω*k)^2 + m^2), -Λ:Λ)
    # action and gradient
    function action(φ)
        free_term = 0.5*sum(map(sum, map(φi -> freeFact.*abs2.(φi), φ)))
        φ4_term = 0
        if λ != 0 
            numPairs = length(fourPairs)
            φ4_term_arr = zeros(ComplexF64, numPairs)
            Threads.@threads for k = 1:numPairs
                ks = fourPairs[k]
                firstSum = sum(map(φN -> φprod(φN, ks[1:2]), φ))
                secondSum = sum(map(φN -> φprod(φN, ks[3:4]), φ))
                φ4_term_arr[k] = firstSum*secondSum
            end
            φ4_term = sum(φ4_term_arr)
            φ4_term = β*λ/(NN*24)*real(φ4_term)
        end
        return free_term + φ4_term
    end
    ℓπ(φ) = -actiοn(φ)
    # Define gradient
    function ∂ℓπ∂φ(φ::AbstractVector)
        freeTerm = map(φN -> freeFact.*φN, φ)
        if λ != 0 
            function ∂ℓπ∂φN(φN)
                function ∂ℓπ∂φNk(Ks)
                    φ4TermNk = 0
                    for ks in Ks
                        φi_k1 = φprod(φN, ks[1])
                        φj_k2xφj_k3 = sum(map(φj -> φprod(φj, ks[2:3]), φ))
                        φ4TermNk += φi_k1*φj_k2xφj_k3
                    end
                    return φ4TermNk
                end
                return map(∂ℓπ∂φNk, threePairsK)
            end
            return -action(φ), -(freeTerm + β*λ/(NN*12)*map(∂ℓπ∂φN, φ))
        end
        return -action(φ),  -freeTerm
    end
    # setup HMC
    # Define a Hamiltonian system
    metric = HermitianMetric(kSize, NN)
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂φ)
    # Define a leapfrog solver, with initial step size chosen heuristically
    rng = MersenneTwister(1234)
    initial_ϵ = find_good_stepsize(rng, hamiltonian, initial_φ)    # Define integrator
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
    phis, stats = sample(hamiltonian, proposal, initial_φ, n_samples, adaptor, n_adapts, progress=progmeter, verbose=verb)
    return phis, stats
end

Λ = 10
NN = 2
λ = 0.0
β = Λ/1.0
n_samples = 100
initial_φ = map(i->zeros(ComplexF64, 2*Λ+1),1:NN)
phis, stats = computeSamples(1.0, λ, NN, initial_φ, n_samples, 2000, progmeter=true)
data = ensemble_data(phis, 1.0, λ, β, 0)
twopts = data[7]./NN
plot(twopts,legend=false, tickfontsize=12, linewidth=1, xguidefontsize=14, yguidefontsize=14, xlabel = "Number of Samples")
savefig("thermalization.png")
dataTherm = ensemble_data(phis, 1.0, λ, β, 50)
twoptsT = dataTherm[7]./NN
plot(twoptsT, legend=false)
maxlag = 500
lags = Array(0:Int(round(maxlag/(100-1))):maxlag)
println(length(lags))
plot(lags, autocor(twoptsT, lags), legend=false, xguidefontsize=14, yguidefontsize=14, yticks = [0.0, 0.25, 0.5, 0.75, 1.0], xlabel = "Number of sweeps", ylabel = "Normalized Autocorrelation", ylims = (-0.2,1.1), tickfontsize=12, linewidth=2)
savefig("autocorr.png")
function mean_lag(data, lag)
    ld = length(data)
    m = 0
    is = Array(1:Int(round.(ld/(lag-1))):ld)
    for i = is
        m += data[i]
    end
    return m/length(is)
end
mean_lag(twoptsT, 10)

"""
initial_φ0 = map(i->zeros(ComplexF64, 2*Λ+1),1:2)
ugh=computeSamples(1.0, 0.1, 2, initial_φ0,50)
"""

#computeSamples(1.0, 1.0, 2, 100)

function compareHMCandVM()
    λtest = 5.0
    Ntest = 10
    n_samples = 2000
    phis, stats = computeSamples(1.0, λtest, Ntest, n_samples; progmeter=true, verb=false)
    phisN = map(n -> map(φ -> φ[n], phis), 1:Ntest)
    data = map(φNs -> ensemble_data(φNs, 1.0, λtest, Λ/1.0), phisN)
    plot(sum(map(n->data[n][1],1:Ntest)), label="HMC", legend=:bottomright)
    plot!(sum(map(n->data[n][2],1:Ntest)), label="HMC_ext")

    phisV, statsV = variationalWarmUp(1.0, λtest, Ntest, n_samples)
    phisVN = map(n -> map(φ -> φ[n], phisV), 1:Ntest)
    dataV = map(φNs -> ensemble_data(φNs, 1.0, λtest, Λ/1.0), phisVN)
    plot!(sum(map(n->dataV[n][1],1:Ntest)), label="VHMC")
    plot!(sum(map(n->dataV[n][2],1:Ntest)), label="VHMC_ext")
    plot!(title=string("O(N) Theory; N = ", Ntest, ", λ = ", λtest, ", m = 1.0"),
        xaxis="Number of samples",
        yaxis="<φ(0)·φ(0)>")
    savefig(string("figures\\HMC_vs_VHMC_twopt0_λ=",λtest,"_N=",Ntest,".png"))
    plot!()

    plot(legend=:bottomright)
    plot!(title=string("O(N) Theory; N = ", Ntest, ", λ = ", λtest, ", m = 1.0"),
        xaxis="Number of samples",
        yaxis="<Action>")
    plot!(running_ave(map(s-> -s[4], stats)), label="HMC")
    plot!(running_ave(map(s-> -s[4], statsV)), label="VHMC")
    savefig(string("figures\\HMC_vs_VHMC_action_λ=",λtest,".png"))
    plot!()
end

function VHMC(m, λ, NN, n_vsamples=1000, n_fsamples=200, n_adapts=min(Int(n_samples*0.1), 1000); progmeter=false, verb=false)
    phiV, statsV = variationalWarmUp(m, λ, NN, n_vsamples)
    phi0 = mean(phiV[end-10:end])
    phi, stats = computeSamples(m, λ, NN, phi0, n_fsamples, n_adapts; progmeter=true)
    return phiV, statsV, phi, stats
end

"""
Ntest=3
n_vsamples = 1000
n_fsamples = 1000
phiV, statsV, phi, stats = VHMC(1.0, 5.0, Ntest, n_vsamples, n_fsamples)
phiFull = vcat(phiV, phi)
phiFN = map(n -> map(φ -> φ[n], phiFull), 1:Ntest)
data = map(φNs -> ensemble_data(φNs, 1.0, 5.0, Λ/1.0), phiFN)
plot(sum(map(n->data[n][1],1:3)), label="VM+HMC", legend=:bottomright)
plot!(sum(map(n->data[n][2],1:3)), label="VM+HMC_ext")
vline!([n_vsamples])

plot(legend=:bottomright)
plot!(title=string("O(N) Theory; N = ", Ntest, ", λ = ", λtest, ", m = 1.0"),
    xaxis="Number of samples",
    yaxis="<Action>")
plot!(running_ave(vcat(map(s-> -s[4], statsV), map(s-> -s[4], stats))), label="HMC")
vline!([n_vsamples], label="End VHMC")

β=Λ/1.0
dataV = map(φNs -> ensemble_data(φNs, 1.0, 5.0, Λ/1.0),  map(n -> map(φ -> φ[n], phiV), 1:Ntest))
plot(0:(β/2)/100:β/2, t->sum(map(n->dataV[n][4](t),1:3)), label="VM", legend=:topright)
plot!(0:(β/2)/100:β/2, t->sum(map(n->data[n][4](t),1:3)), label="VM+HMC")
"""

function computeCouplingEnsemble(m, NN, n_samples, λ_min, λ_max, n_λ; V_warmup=0, throwaway=100)
    # collect samples and data across m values
    λs = Array(λ_min:(λ_max-λ_min)/(n_λ-1):λ_max)
    println("λs = ", λs)
    phis = Array{Any,1}(undef, n_λ)
    phiVs = Array{Any,1}(undef, n_λ)
    data = Array{Any,1}(undef, n_λ)
    twopt0 = Array{Any,1}(undef, n_λ)
    β = Λ/m
    Threads.@threads for i = 1:n_λ
        λi=λs[i]
        if V_warmup != 0
            println("Sampling for λ = ", round.(λs[i], sigdigits=5))
            phiVs[i], statsV, phis[i], stats = VHMC(m, λi, NN, V_warmup, n_samples, 100)
            println("Sampling for λ = ", round.(λs[i], sigdigits=5), " complete...")
            phiFull = vcat(phiVs[i], phis[i])
            phiFN = map(n -> map(φ -> φ[n], phiFull), 1:NN)
            throwaway = V_warmup
            data[i] = map(φNs -> ensemble_data(φNs, m, λi, β, throwaway), phiFN)
        else
            initial_φ = map(i->zeros(ComplexF64, 2*Λ+1),1:NN)
            phis[i], stats = computeSamples(m, λi, NN, initial_φ, n_samples)
            data[i] = ensemble_data(phis[i], m, λi, Λ/m, throwaway)
        end
    end
    return data
end

global Λ = 10
global kSize = 2*Λ+1
global fourPairs = fourierPairs(Λ, 4, 0)
global threePairsK = map(k -> fourierPairs(Λ, 3, k), -Λ:Λ)
mm = 10.0
global β = Λ/mm
λmin = 1.0
λmax = 250.0
NN = 5
nλ = 12
nchains = 10
bigdata = Array{Any,1}(undef, nchains)
Threads.@threads for i = 1:nchains
    bigdata[i] = computeCouplingEnsemble(mm, NN, 1000, λmin, λmax, nλ, V_warmup=false, throwaway= 0)
end
println("WORKED")

function get_twopt0(data)
    twopt0s = map(i -> map(datai -> datai[2], data[i]), 1:nλ)
    twopt0s1 = map(x -> map(y-> y[end], x), twopt0s)
    twopt0s11 = map(mean, twopt0s1)
    return twopt0s, twopt0s11
end

twopts0s=map(dataChain -> map(dataλ -> mm*dataλ[2][end]/NN, dataChain), bigdata)
twopts_ave = mean(twopts0s)
twopts_err = std(twopts0s)/sqrt(nchains)
#scatter(λs, twopts0s, legend=false)
gs = Array(λmin:(λmax-λmin)/(nλ-1):λmax)/NN
scatter(gs, twopts_ave, yerr=twopts_err)

function get_ms(data)
    ms = map(i -> data[i][6][1], 1:nλ)
    return ms
end

m_data = map(get_ms, bigdata)
ms = mean(m_data)
m_errs = std(m_data)/sqrt(nchains)
scatter(gs, ms, yerr=m_errs, legend=false, tickfontsize=10, markersize=6)
savefig("ms_N=2.png")

function get_twopt_t(data)
    twopt_ts = map(datan -> map(i -> t->datan[i][4](t)/NN, 1:nλ), data)
    twopt_ts1 = map(i -> map(twoptugh -> twoptugh[i], twopt_ts), 1:nchains)
    twopt_ts2 = map(twpt-> t->mean(map(twop->twop(t), twpt)), twopt_ts1)
    return twopt_ts2
end

twopt_ts = get_twopt_t(bigdata)

ts = Array(0:(β/2)/(50-1):β/2)
plot(ts, bigdata1[2][4])
plot(ts, twopt_ts[1], label=string("g = ", gs[1]), linewidth = 2, tickfontsize=10, legendfontsize=12)
plot!(ts, twopt_ts[2], label=string("g = ", gs[2]), linewidth = 2)
plot!(ts, twopt_ts[5], label=string("g = ", gs[5]), linewidth = 2)
plot!(ts, twopt_ts[10], label=string("g = ", gs[10]), linewidth = 2)
savefig("ON_twopt_t.png")

function compute_N_Ensemble(m, λ, n_samples, Ns; V_warmup=0, throwaway=0)
    # collect samples and data across m values
    n_N = length(Ns)
    phis = Array{Any,1}(undef, n_N)
    phiVs = Array{Any,1}(undef, n_N)
    data = Array{Any,1}(undef, n_N)
    twopt0 = Array{Any,1}(undef, n_N)
    β = Λ/m
    Threads.@threads for i = 1:n_N
        NN = Ns[i]
        initial_φ = map(i->zeros(ComplexF64, 2*Λ+1), 1:NN)
        phis[i], stats = computeSamples(m, λ, NN, initial_φ, n_samples)
        println("Sampling for N = ", NN, " complete...")
        data[i] = ensemble_data(phis[i], m, λ, Λ/m, throwaway)
    end
    return data
end

Ns = [2,3,5,10]
dataNN1 = compute_N_Ensemble(1.0, 1.0, 500, Ns, throwaway=0)

"""
function compute_Λ_Ensemble(m, λ, NN, Λs, n_samples, n_adapts=0; V_warmup=0, throwaway=0)
    # collect samples and data across m values
    n_Λ = length(Λs)
    phis = Array{Any,1}(undef, n_Λ)
    phiVs = Array{Any,1}(undef, n_Λ)
    data = Array{Any,1}(undef, n_Λ)
    twopt0 = Array{Any,1}(undef, n_Λ)
    for i = 1:n_Λ
        global Λ = Λs[i]
        global kSize = 2*Λ+1
        global fourPairs = fourierPairs(Λ, 4, 0)
        global threePairsK = map(k -> fourierPairs(Λ, 3, k), -Λ:Λ)
        global β = Λ/m
        initial_φ = map(i->zeros(ComplexF64, 2*Λ+1), 1:NN)
        phis[i], stats = computeSamples(m, λ, NN, initial_φ, n_samples, n_adapts)
        println("Sampling for Λ = ", Λ, " complete...")
        data[i] = ensemble_data(phis[i], m, λ, β, throwaway)
    end
    return data
end
function residualPlotΛ(m, λ, NN, Λs, n_samples, n_adapts, n_chains; throwaway=0)
    data = Array{Any,1}(undef, n_chains)
    for i = 1:n_chains
        data[i] = compute_Λ_Ensemble(m, λ, NN, Λs, n_samples, n_adapts; throwaway=2000)
    end
    twoptRes = mean(map(datai->map(dataΛ -> dataΛ[2]/NN .- 0.5/m, datai), data))
    legendΛs = reshape(map(Λi -> string("Λ = ", Λi), Λs),(1,length(Λs)))
    display(plot(twoptRes,legend=:bottomright, label=legendΛs, ylims=(0.0-0.05,0.0+0.05)))
    xaxis!("Number of samples")
    yaxis!("Residual of          ")
    display(plot!())
    savefig(string("Λconv_twopt_",NN,".png"))
    return twoptRes
end
m=1.0
λ=0.0
NN=2
Λs = [4, 8, 10, 20]#, 20]
n_samples = 20000
n_adapts = 1000
n_chains = 10
twoptRes = residualPlotΛ(m, λ, NN, Λs, n_samples, n_adapts, n_chains; throwaway=2000)
"""

function compute_β_Ensemble(m, λ, NN, βs, n_samples, n_adapts=0; V_warmup=0, throwaway=0)
    # collect samples and data across m values
    n_β = length(βs)
    phis = Array{Any,1}(undef, n_β)
    phiVs = Array{Any,1}(undef, n_β)
    data = Array{Any,1}(undef, n_β)
    twopt0 = Array{Any,1}(undef, n_β)
    for i = 1:n_β
        global β = βs[i]
        initial_φ = map(i->zeros(ComplexF64, 2*Λ+1), 1:NN)
        phis[i], stats = computeSamples(m, λ, NN, initial_φ, n_samples, n_adapts)
        println("Sampling for β = ", β, " complete...")
        data[i] = ensemble_data(phis[i], m, β, β, throwaway)
    end
    return data
end
function plot_β_ensemble()
    global Λ = 12
    global kSize = 2*Λ+1
    global fourPairs = fourierPairs(Λ, 4, 0)
    global threePairsK = map(k -> fourierPairs(Λ, 3, k), -Λ:Λ)
    βs = [1.0 2.0 3.0 5.0 10.0 100.0]
    m = 1.0
    λ = 0.0
    NN = 5
    n_samples = 10000
    n_adapts = 1000
    throw = 2000
    data = compute_β_Ensemble(m, λ, NN, βs, n_samples, n_adapts, throwaway=throw)
    twoptRes = map(dat->dat[2]/NN.-0.5,data)
    legendβs = reshape(map(βi -> string("β = ", βi), βs),(1,length(βs)))
    plot(twoptRes, label = legendβs)
end

