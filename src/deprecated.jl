# PR 23341
import LinearAlgebra: diagm
@deprecate diagm(A::SparseMatrixCSC) sparse(Diagonal(sparsevec(A)))

# PR #23757
@deprecate spdiagm(x::AbstractVector) sparse(Diagonal(x))
function spdiagm(x::AbstractVector, d::Number)
    depwarn(string("`spdiagm(x::AbstractVector, d::Number)` is deprecated, use ",
        "`spdiagm(d => x)` instead, which now returns a square matrix. To preserve the old ",
        "behaviour, use `sparse(SparseArrays.spdiagm_internal(d => x)...)`"), :spdiagm)
    I, J, V = spdiagm_internal(d => x)
    return sparse(I, J, V)
end
function spdiagm(x, d)
    depwarn(string("`spdiagm((x1, x2, ...), (d1, d2, ...))` is deprecated, use ",
        "`spdiagm(d1 => x1, d2 => x2, ...)` instead, which now returns a square matrix. ",
        "To preserve the old behaviour, use ",
        "`sparse(SparseArrays.spdiagm_internal(d1 => x1, d2 => x2, ...)...)`"), :spdiagm)
    I, J, V = spdiagm_internal((d[i] => x[i] for i in 1:length(x))...)
    return sparse(I, J, V)
end
function spdiagm(x, d, m::Integer, n::Integer)
    depwarn(string("`spdiagm((x1, x2, ...), (d1, d2, ...), m, n)` is deprecated, use ",
        "`spdiagm(d1 => x1, d2 => x2, ...)` instead, which now returns a square matrix. ",
        "To specify a non-square matrix and preserve the old behaviour, use ",
        "`I, J, V = SparseArrays.spdiagm_internal(d1 => x1, d2 => x2, ...); sparse(I, J, V, m, n)`"), :spdiagm)
    I, J, V = spdiagm_internal((d[i] => x[i] for i in 1:length(x))...)
    return sparse(I, J, V, m, n)
end

# methods involving RowVector from base/sparse/linalg.jl, to deprecate
\(::SparseMatrixCSC, ::RowVector) = throw(DimensionMismatch("Cannot left-divide matrix by transposed vector"))
\(::Adjoint{<:Any,<:SparseMatrixCSC}, ::RowVector) = throw(DimensionMismatch("Cannot left-divide matrix by transposed vector"))
\(::Transpose{<:Any,<:SparseMatrixCSC}, ::RowVector) = throw(DimensionMismatch("Cannot left-divide matrix by transposed vector"))
