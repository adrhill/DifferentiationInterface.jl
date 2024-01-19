module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DocStringExtensions
using ForwardDiff
using ForwardDiff: Dual, Tag, extract_derivative, extract_derivative!
using LinearAlgebra

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:Real}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    dy = extract_derivative(T, ydual)
    return dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:AbstractArray}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    dy = extract_derivative!(T, dy, ydual)
    return dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Real}
    g = ForwardDiff.gradient(f, x)  # TODO: replace with duals, n times too slow
    new_dy = dot(g, dx)
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ForwardDiff.jacobian(f, x)  # TODO: replace with duals, n times too slow
    mul!(dy, J, dx)
    return dy
end

end
