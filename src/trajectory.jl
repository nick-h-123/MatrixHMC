####
#### Implementation for Hamiltonian dynamics trajectories
####
#### Developers' Notes
####
#### Not all functions that use `rng` require a fallback function with `GLOBAL_RNG`
#### as default. In short, only those exported to other libries need such a fallback
#### function. Internal uses shall always use the explict `rng` version. (Kai Xu 6/Jul/19)

"""
$(TYPEDEF)
A transition that contains the phase point and
other statistics of the transition.

# Fields
$(TYPEDFIELDS)
"""
struct Transition{P<:PhasePoint, NT<:NamedTuple}
    "Phase-point for the transition."
    z       ::  P
    "Statistics related to the transition, e.g. energy."
    stat    ::  NT
end

"Returns the statistics for transition `t`."
stat(t::Transition) = t.stat

abstract type AbstractMCMCKernel end

abstract type AbstractTerminationCriterion end

abstract type StaticTerminationCriterion <: AbstractTerminationCriterion end
"""
$(TYPEDEF)
Static HMC with a fixed number of leapfrog steps.

# Fields
$(TYPEDFIELDS)

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))
"""
struct FixedNSteps <: StaticTerminationCriterion
    "Number of steps to simulate, i.e. length of trajectory will be `L + 1`."
    L::Int
end

"""
$(TYPEDEF)
Standard HMC implementation with a fixed integration time.

# Fields
$(TYPEDFIELDS)

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901)) 
"""
struct FixedIntegrationTime{F<:AbstractFloat} <: StaticTerminationCriterion
    "Total length of the trajectory, i.e. take `floor(λ / integrator_step_size)` number of leapfrog steps."
    λ::F
end

##
## Sampling methods for trajectories.
##

"How to sample a phase-point from the simulated trajectory."
abstract type AbstractTrajectorySampler end

"Samples the end-point of the trajectory."
struct EndPointTS <: AbstractTrajectorySampler end

"""
$(TYPEDEF)

Trajectory slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.

# Fields

$(TYPEDFIELDS)
"""
struct SliceTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Slice variable in log-space."
    ℓu      ::  F
    "Number of acceptable candidates, i.e. those with probability larger than slice variable `u`."
    n       ::  Int
end

Base.show(io::IO, s::SliceTS) = print(io, "SliceTS(ℓu=$(s.ℓu), n=$(s.n))")

"""
$(TYPEDEF)

Multinomial trajectory sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.

# Fields

$(TYPEDFIELDS)
"""
struct MultinomialTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Total energy for the given tree, i.e. the sum of energies of all leaves."
    ℓw      ::  F
end

"""
$(TYPEDEF)

Slice sampler for the starting single leaf tree.
Slice variable is initialized.
"""
SliceTS(rng::AbstractRNG, z0::PhasePoint) = SliceTS(z0, log(rand(rng)) - energy(z0), 1)

"""
$(TYPEDEF)

Multinomial sampler for the starting single leaf tree.
(Log) weights for leaf nodes are their (unnormalised) Hamiltonian energies.

Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp#L226
"""
MultinomialTS(rng::AbstractRNG, z0::PhasePoint) = MultinomialTS(z0, zero(energy(z0)))

"""
$(TYPEDEF)

Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
function SliceTS(s::SliceTS, H0::AbstractFloat, zcand::PhasePoint)
    return SliceTS(zcand, s.ℓu, (s.ℓu <= -energy(zcand)) ? 1 : 0)
end

"""
$(TYPEDEF)

Multinomial sampler for a trajectory consisting only a leaf node.
- tree weight is the (unnormalised) energy of the leaf.
"""
function MultinomialTS(s::MultinomialTS, H0::AbstractFloat, zcand::PhasePoint)
    return MultinomialTS(zcand, H0 - energy(zcand))
end

function combine(rng::AbstractRNG, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    zcand = rand(rng) < s1.n / n ? s1.zcand : s2.zcand
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(zcand::PhasePoint, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(rng::AbstractRNG, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    zcand = rand(rng) < exp(s1.ℓw - ℓw) ? s1.zcand : s2.zcand
    return MultinomialTS(zcand, ℓw)
end

function combine(zcand::PhasePoint, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    return MultinomialTS(zcand, ℓw)
end

mh_accept(rng::AbstractRNG, s::SliceTS, s′::SliceTS) = rand(rng) < min(1, s′.n / s.n)

function mh_accept(rng::AbstractRNG, s::MultinomialTS, s′::MultinomialTS)
    return rand(rng) < min(1, exp(s′.ℓw - s.ℓw))
end

"""
$(TYPEDEF)

Numerically simulated Hamiltonian trajectories.
"""
struct Trajectory{
    TS<:AbstractTrajectorySampler, 
    I<:AbstractIntegrator, 
    TC<:AbstractTerminationCriterion,
}
    "Integrator used to simulate trajectory."
    integrator::I
    "Criterion to terminate the simulation."
    termination_criterion::TC
end

Trajectory{TS}(integrator::I, termination_criterion::TC) where {TS, I, TC} = 
    Trajectory{TS, I, TC}(integrator, termination_criterion)

function Base.show(io::IO, τ::Trajectory{TS}) where {TS}
    print(io, "Trajectory{$TS}(integrator=$(τ.integrator), tc=$(τ.termination_criterion))")
end

nsteps(τ::Trajectory{TS, I, TC}) where {TS, I, TC<:FixedNSteps} = τ.termination_criterion.L
nsteps(τ::Trajectory{TS, I, TC}) where {TS, I, TC<:FixedIntegrationTime} = 
    max(1, floor(Int, τ.termination_criterion.λ / nom_step_size(τ.integrator)))

##
## Kernel interface
##

struct HMCKernel{R, T<:Trajectory} <: AbstractMCMCKernel 
    refreshment::R
    τ::T
end

HMCKernel(τ::Trajectory) = HMCKernel(FullMomentumRefreshment(), τ)

"""
$(SIGNATURES)

Make a MCMC transition from phase point `z` using the trajectory `τ` under Hamiltonian `h`.

NOTE: This is a RNG-implicit fallback function for `transition(GLOBAL_RNG, τ, h, z)`
"""
function transition(τ::Trajectory, h::Hamiltonian, z::PhasePoint)
    return transition(GLOBAL_RNG, τ, h, z)
end

###
### Actual trajectory implementations
###

function transition(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    τ::Trajectory{TS, I, TC},
    h::Hamiltonian,
    z::PhasePoint,
) where {TS<:AbstractTrajectorySampler, I, TC<:StaticTerminationCriterion}
    H0 = energy(z)

    z′, is_accept, α = sample_phasepoint(rng, τ, h, z)
    # Do the actual accept / reject
    z = accept_phasepoint!(z, z′, is_accept)    # NOTE: this function changes `z′` in place in matrix-parallel mode
    # Reverse momentum variable to preserve reversibility
    z = PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ)
    H = energy(z)
    tstat = merge(
        (
            n_steps=nsteps(τ),
            is_accept=is_accept,
            acceptance_rate=α,
            log_density=z.ℓπ.value,
            hamiltonian_energy=H,
            hamiltonian_energy_error=H - H0,
        ),
        stat(τ.integrator),
    )
    return Transition(z, tstat)
end

# Return the accepted phase point
function accept_phasepoint!(z::T, z′::T, is_accept::Bool) where {T<:PhasePoint{<:AbstractVector}}
    if is_accept
        return z′
    else
        return z
    end
end

### Use end-point from the trajectory as a proposal and apply MH correction

function sample_phasepoint(rng, τ::Trajectory{EndPointTS}, h, z)
    z′ = step(τ.integrator, h, z, nsteps(τ))
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    return z′, is_accept, α
end

### Multinomial sampling from trajectory

function randcat(rng::AbstractRNG, zs::AbstractVector{<:PhasePoint}, unnorm_ℓp::AbstractVector)
    p = exp.(unnorm_ℓp .- logsumexp(unnorm_ℓp))
    i = randcat(rng, p)
    return zs[i]
end

# zs is in the form of Vector{PhasePoint{Matrix}} and has shape [n_steps][dim, n_chains]
function randcat(rng, zs::AbstractVector{<:PhasePoint}, unnorm_ℓP::AbstractMatrix)
    z = similar(first(zs))
    P = exp.(unnorm_ℓP .- logsumexp(unnorm_ℓP; dims=2)) # (n_chains, n_steps)
    is = randcat(rng, P')
    foreach(enumerate(is)) do (i_chain, i_step)
        zi = zs[i_step]
        z.θ[:,i_chain] = zi.θ[:,i_chain]
        z.r[:,i_chain] = zi.r[:,i_chain]
        z.ℓπ.value[i_chain] = zi.ℓπ.value[i_chain]
        z.ℓπ.gradient[:,i_chain] = zi.ℓπ.gradient[:,i_chain]
        z.ℓκ.value[i_chain] = zi.ℓκ.value[i_chain]
        z.ℓκ.gradient[:,i_chain] = zi.ℓκ.gradient[:,i_chain]
    end
    return z
end

function sample_phasepoint(rng, τ::Trajectory{MultinomialTS}, h, z)
    n_steps = abs(nsteps(τ))
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    n_steps_fwd = rand_coupled(rng, 0:n_steps) 
    zs_fwd = step(τ.integrator, h, z, n_steps_fwd; fwd=true, full_trajectory=Val(true))
    n_steps_bwd = n_steps - n_steps_fwd
    zs_bwd = step(τ.integrator, h, z, n_steps_bwd; fwd=false, full_trajectory=Val(true))
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims=2)
    end
    unnorm_ℓprob = ℓweights
    z′ = randcat(rng, zs, unnorm_ℓprob)
    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims=2))
    return z′, true, α
end

###
### Advanced HMC implementation with (adaptive) dynamic trajectory length.
###

##
## Variants of no-U-turn criteria
##

"""
    Termination

Termination reasons
- `dynamic`: due to stoping criteria
- `numerical`: due to large energy deviation from starting (possibly numerical errors)
"""
struct Termination
    dynamic::Bool
    numerical::Bool
end

Base.show(io::IO, d::Termination) = print(io, "Termination(dynamic=$(d.dynamic), numerical=$(d.numerical))")
Base.:*(d1::Termination, d2::Termination) = Termination(d1.dynamic || d2.dynamic, d1.numerical || d2.numerical)
isterminated(d::Termination) = d.dynamic || d.numerical

"""
$(SIGNATURES)

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::SliceTS, nt::Trajectory, H0::F, H′::F) where {F<:AbstractFloat}
    return Termination(false, !(s.ℓu < nt.termination_criterion.Δ_max + -H′))
end

"""
$(SIGNATURES)

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::MultinomialTS, nt::Trajectory, H0::F, H′::F) where {F<:AbstractFloat}
    return Termination(false, !(-H0 < nt.termination_criterion.Δ_max + -H′))
end

###
### Initialisation of step size
###

"""
A single Hamiltonian integration step.

NOTE: this function is intended to be used in `find_good_stepsize` only.
"""
function A(h, z, ϵ)
    z′ = step(Leapfrog(ϵ), h, z)
    H′ = energy(z′)
    return z′, H′
end

"Find a good initial leap-frog step-size via heuristic search."
function find_good_stepsize(
    rng::AbstractRNG,
    h::Hamiltonian,
    θ::AbstractArray;
    max_n_iters::Int=100,
)
    # Initialize searching parameters
    ϵ′ = ϵ = Float64(0.1)
    a_min, a_cross, a_max = Float64(0.25), Float64(0.5), Float64(0.75) # minimal, crossing, maximal accept ratio
    d = Float64(2.0)
    # Create starting phase point
    r = rand(rng, h.metric) # sample momentum variable
    z = phasepoint(h, θ, r)
    H = energy(z)

    # Make a proposal phase point to decide direction
    z′, H′ = A(h, z, ϵ)
    ΔH = H - H′ # compute the energy difference; `exp(ΔH)` is the MH accept ratio
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        # `direction` being  `1` means MH ratio too high
        #     - this means our step size is too small, thus we increase
        # `direction` being `-1` means MH ratio too small
        #     - this means our step szie is too large, thus we decrease
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        z′, H′ = A(h, z, ϵ)
        ΔH = H - H′
        DEBUG && @debug "Crossing step" direction H′ ϵ "α = $(min(1, exp(ΔH)))"
        if (direction == 1) && !(ΔH > log(a_cross))
            break
        elseif (direction == -1) && !(ΔH < log(a_cross))
            break
        else
            ϵ = ϵ′
        end
    end
    # Note after the for loop,
    # `ϵ` and `ϵ′` are the two neighbour step sizes across `a_cross`.

    # Bisection step: ensure final accept ratio: a_min < a < a_max.
    # See https://en.wikipedia.org/wiki/Bisection_method

    ϵ, ϵ′ = ϵ < ϵ′ ? (ϵ, ϵ′) : (ϵ′, ϵ)  # ensure ϵ < ϵ′;
    # Here we want to use a value between these two given the
    # criteria that this value also gives us a MH ratio between `a_min` and `a_max`.
    # This condition is quite mild and only intended to avoid cases where
    # the middle value of `ϵ` and `ϵ′` is too extreme.
    for _ = 1:max_n_iters
        ϵ_mid = middle(ϵ, ϵ′)
        z′, H′ = A(h, z, ϵ_mid)
        ΔH = H - H′
        DEBUG && @debug "Bisection step" H′ ϵ_mid "α = $(min(1, exp(ΔH)))"
        if (exp(ΔH) > a_max)
            ϵ = ϵ_mid
        elseif (exp(ΔH) < a_min)
            ϵ′ = ϵ_mid
        else
            ϵ = ϵ_mid
            break
        end
    end

    return ϵ
end

function find_good_stepsize(
    h::Hamiltonian,
    θ::AbstractArray;
    max_n_iters::Int=100,
)
    return find_good_stepsize(GLOBAL_RNG, h, θ; max_n_iters=max_n_iters)
end

"Perform MH acceptance based on energy, i.e. negative log probability."
function mh_accept_ratio(
    rng::AbstractRNG,
    Horiginal::T,
    Hproposal::T,
) where {T<:AbstractFloat}
    α = min(one(T), exp(Horiginal - Hproposal))
    accept = rand(rng, T) < α
    return accept, α
end
