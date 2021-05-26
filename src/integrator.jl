####
#### Numerical methods for simulating Hamiltonian trajectory.
####

# TODO: The type `<:Tuple{Integer,Bool}` is introduced to address
# https://github.com/TuringLang/Turing.jl/pull/941#issuecomment-549191813
# We might want to simplify it to `Tuple{Int,Bool}` when we figured out
# why the it behaves unexpected on Windos 32.

"""
$(TYPEDEF)

Represents an integrator used to simulate the Hamiltonian system.

# Implementation
A `AbstractIntegrator` is expected to have the following implementations:
- `stat`(@ref)
- `nom_step_size`(@ref)
- `step_size`(@ref)
"""
abstract type AbstractIntegrator end

stat(::AbstractIntegrator) = NamedTuple()

"""
    nom_step_size(::AbstractIntegrator)

Get the nominal integration step size. The current integration step size may
differ from this, for example if the step size is jittered. Nominal step size is
usually used in adaptation.
"""
nom_step_size(i::AbstractIntegrator) = step_size(i)

"""
    step_size(::AbstractIntegrator)

Get the current integration step size.
"""
function step_size end

"""
    update_nom_step_size(i::AbstractIntegrator, ϵ) -> AbstractIntegrator

Return a copy of the integrator `i` with the new nominal step size ([`nom_step_size`](@ref))
`ϵ`.
"""
function update_nom_step_size end

abstract type AbstractLeapfrog{T} <: AbstractIntegrator end

step_size(lf::AbstractLeapfrog) = lf.ϵ
jitter(::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, lf::AbstractLeapfrog) = lf
temper(lf::AbstractLeapfrog, r, ::NamedTuple{(:i, :is_half),<:Tuple{Integer,Bool}}, ::Int) = r
stat(lf::AbstractLeapfrog) = (step_size=step_size(lf), nom_step_size=nom_step_size(lf))

update_nom_step_size(lf::AbstractLeapfrog, ϵ) = reconstruct(lf, ϵ=ϵ)

"""
$(TYPEDEF)

Leapfrog integrator with fixed step size `ϵ`.

# Fields

$(TYPEDFIELDS)
"""
struct Leapfrog{T<:AbstractScalarOrVec{<:AbstractFloat}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ       ::  T
end
Base.show(io::IO, l::Leapfrog) = print(io, "Leapfrog(ϵ=$(round.(l.ϵ; sigdigits=3)))")

struct MTLeapfrog{T<:AbstractScalarOrVec{<:AbstractFloat}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ       ::  T
end
Base.show(io::IO, l::MTLeapfrog) = print(io, "Leapfrog(ϵ=$(round.(l.ϵ; sigdigits=3)))")

function step(
    lf::Leapfrog{T},
    h::Hamiltonian,
    z::P,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj} = Val(false)
) where {T<:AbstractScalarOrVec{<:AbstractFloat}, P<:PhasePoint, FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    #ϵ = ϵ'

    res = if FullTraj
        Vector{P}(undef, n_steps)
    else
        z
    end

    @unpack θ, r = z
    @unpack value, gradient = z.ℓπ
    for i = 1:n_steps
        # Tempering
        r = temper(lf, r, (i=i, is_half=true), n_steps)
        # Take a half leapfrog step for momentum variable
        r = r - ϵ / 2 .* gradient
        # Take a full leapfrog step for position variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ .* ∇r
        # Take a half leapfrog step for momentum variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ / 2 .* gradient
        # Tempering
        r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        else
            res = z
        end
        if !isfinite(z)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
    end
    return res
end

function step(
    lf::MTLeapfrog{T},
    h::Hamiltonian,
    z::P,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj} = Val(false)
) where {T<:AbstractScalarOrVec{<:AbstractFloat}, P<:PhasePoint, FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    #ϵ = ϵ'

    res = if FullTraj
        Vector{P}(undef, n_steps)
    else
        z
    end

    @unpack θ, r = z
    @unpack value, gradient = z.ℓπ
    for i = 1:n_steps
        # Tempering
        r = temper(lf, r, (i=i, is_half=true), n_steps)
        # Take a half leapfrog step for momentum variable
        r = r - ϵ / 2 .* gradient
        # Take a full leapfrog step for position variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ .* ∇r
        # Take a half leapfrog step for momentum variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ / 2 .* gradient
        # Tempering
        r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        else
            res = z
        end
        if !isfinite(z)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
    end
    return res
end
