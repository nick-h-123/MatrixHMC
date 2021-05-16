using MatrixHMC
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
kSize = 2*Λ+1
NN = 2
λ = 0.0
β = Λ/1.0
n_samples = 1000
initial_φ = map(i->zeros(ComplexF64, 2*Λ+1),1:NN)
phis, stats = computeSamples(1.0, λ, NN, initial_φ, n_samples, 2000, progmeter=true)
data = ensemble_data(phis, 1.0, λ, β, 0)
twopts = data[7]./NN
plot(twopts,legend=false, tickfontsize=12, linewidth=1, xguidefontsize=14, yguidefontsize=14, xlabel = "Number of Samples")
dataTherm = ensemble_data(phis, 1.0, λ, β, 50)
twoptsT = dataTherm[7]./NN
plot(twoptsT, legend=false)
