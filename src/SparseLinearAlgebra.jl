module SparseLinearAlgebra

using Reexport
@reexport using LinearAlgebra
using LinearAlgebra: checksquare
@reexport using SparseArrays
using SparseArrays: SparseMatrixCSCUnion, SparseVectorUnion, getnzval, getrowval, getcolptr
using SuiteSparse

import Base: +, -, *, \, /, adjoint, conj, diff, imag, inv, real, transpose
import LinearAlgebra: adjoint!, diag, dot, factorize, kron, ldiv!, lmul!, mul!, norm, opnorm, rdiv!, rmul!, transpose!, tril, triu

## transpose
adjoint!(  X::SparseMatrixCSC{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = SparseArrays.ftranspose!(X, A, conj)
transpose!(X::SparseMatrixCSC{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = SparseArrays.ftranspose!(X, A, identity)

adjoint(  A::SparseMatrixCSC) = Adjoint(A)
transpose(A::SparseMatrixCSC) = Transpose(A)

Base.copy(A::Adjoint{<:Any,<:SparseMatrixCSC})   = SparseArrays.ftranspose(A.parent, conj)
Base.copy(A::Transpose{<:Any,<:SparseMatrixCSC}) = SparseArrays.ftranspose(A.parent, identity)

function tril!(A::SparseMatrixCSC, k::Integer = 0, trim::Bool = true)
    if !(-A.m - 1 <= k <= A.n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-A.m - 1) and at most $(A.n - 1) in an $(A.m)-by-$(A.n) matrix")))
    end
    SparseArrays.fkeep!(A, (i, j, x) -> i + k >= j, trim)
end
function triu!(A::SparseMatrixCSC, k::Integer = 0, trim::Bool = true)
    if !(-A.m + 1 <= k <= A.n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-A.m + 1) and at most $(A.n + 1) in an $(A.m)-by-$(A.n) matrix")))
    end
    SparseArrays.fkeep!(A, (i, j, x) -> j >= i + k, trim)
end

conj!(A::SparseMatrixCSC) = (@inbounds broadcast!(conj, A.nzval, A.nzval); A)
(-)(A::SparseMatrixCSC) = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), map(-, A.nzval))

# the rest of real, conj, imag are handled correctly via AbstractArray methods
conj(A::SparseMatrixCSC{<:Complex}) =
    SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), conj(A.nzval))
imag(A::SparseMatrixCSC{Tv,Ti}) where {Tv<:Real,Ti} = spzeros(Tv, Ti, A.m, A.n)

## Binary arithmetic and boolean operators
(+)(A::SparseMatrixCSC, B::SparseMatrixCSC) = map(+, A, B)
(-)(A::SparseMatrixCSC, B::SparseMatrixCSC) = map(-, A, B)

(+)(A::SparseMatrixCSC, B::Array) = Array(A) + B
(+)(A::Array, B::SparseMatrixCSC) = A + Array(B)
(-)(A::SparseMatrixCSC, B::Array) = Array(A) - B
(-)(A::Array, B::SparseMatrixCSC) = A - Array(B)

## Structure query functions
issymmetric(A::SparseMatrixCSC) = is_hermsym(A, identity)

ishermitian(A::SparseMatrixCSC) = is_hermsym(A, conj)

function is_hermsym(A::SparseMatrixCSC, check::Function)
    m, n = size(A)
    if m != n; return false; end

    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    tracker = copy(A.colptr)
    for col = 1:A.n
        # `tracker` is updated such that, for symmetric matrices,
        # the loop below starts from an element at or below the
        # diagonal element of column `col`"
        for p = tracker[col]:colptr[col+1]-1
            val = nzval[p]
            row = rowval[p]

            # Ignore stored zeros
            if val == 0
                continue
            end

            # If the matrix was symmetric we should have updated
            # the tracker to start at the diagonal or below. Here
            # we are above the diagonal so the matrix can't be symmetric.
            if row < col
                return false
            end

            # Diagonal element
            if row == col
                if val != check(val)
                    return false
                end
            else
                offset = tracker[row]

                # If the matrix is unsymmetric, there might not exist
                # a rowval[offset]
                if offset > length(rowval)
                    return false
                end

                row2 = rowval[offset]

                # row2 can be less than col if the tracker didn't
                # get updated due to stored zeros in previous elements.
                # We therefore "catch up" here while making sure that
                # the elements are actually zero.
                while row2 < col
                    if nzval[offset] != 0
                        return false
                    end
                    offset += 1
                    row2 = rowval[offset]
                    tracker[row] += 1
                end

                # Non zero A[i,j] exists but A[j,i] does not exist
                if row2 > col
                    return false
                end

                # A[i,j] and A[j,i] exists
                if row2 == col
                    if val != check(nzval[offset])
                        return false
                    end
                    tracker[row] += 1
                end
            end
        end
    end
    return true
end

function istriu(A::SparseMatrixCSC)
    m, n = size(A)
    colptr = A.colptr
    rowval = A.rowval
    nzval  = A.nzval

    for col = 1:min(n, m-1)
        l1 = colptr[col+1]-1
        for i = 0 : (l1 - colptr[col])
            if rowval[l1-i] <= col
                break
            end
            if nzval[l1-i] != 0
                return false
            end
        end
    end
    return true
end

function istril(A::SparseMatrixCSC)
    m, n = size(A)
    colptr = A.colptr
    rowval = A.rowval
    nzval  = A.nzval

    for col = 2:n
        for i = colptr[col] : (colptr[col+1]-1)
            if rowval[i] >= col
                break
            end
            if nzval[i] != 0
                return false
            end
        end
    end
    return true
end

function spdiagm_internal(kv::Pair{<:Integer,<:AbstractVector}...)
    ncoeffs = 0
    for p in kv
        ncoeffs += length(p.second)
    end
    I = Vector{Int}(undef, ncoeffs)
    J = Vector{Int}(undef, ncoeffs)
    V = Vector{promote_type(map(x -> eltype(x.second), kv)...)}(undef, ncoeffs)
    i = 0
    for p in kv
        dia = p.first
        vect = p.second
        numel = length(vect)
        if dia < 0
            row = -dia
            col = 0
        elseif dia > 0
            row = 0
            col = dia
        else
            row = 0
            col = 0
        end
        r = 1+i:numel+i
        I[r] = row+1:row+numel
        J[r] = col+1:col+numel
        copyto!(view(V, r), vect)
        i += numel
    end
    return I, J, V
end

"""
    spdiagm(kv::Pair{<:Integer,<:AbstractVector}...)

Construct a square sparse diagonal matrix from `Pair`s of vectors and diagonals.
Vector `kv.second` will be placed on the `kv.first` diagonal.

# Examples
```jldoctest
julia> spdiagm(-1 => [1,2,3,4], 1 => [4,3,2,1])
5×5 SparseMatrixCSC{Int64,Int64} with 8 stored entries:
  [2, 1]  =  1
  [1, 2]  =  4
  [3, 2]  =  2
  [2, 3]  =  3
  [4, 3]  =  3
  [3, 4]  =  2
  [5, 4]  =  4
  [4, 5]  =  1
```
"""
function spdiagm(kv::Pair{<:Integer,<:AbstractVector}...)
    I, J, V = spdiagm_internal(kv...)
    n = max(SparseArrays.dimlub(I), SparseArrays.dimlub(J))
    return sparse(I, J, V, n, n)
end

## expand a colptr or rowptr into a dense index vector
function expandptr(V::Vector{<:Integer})
    if V[1] != 1 throw(ArgumentError("first index must be one")) end
    res = similar(V, (Int64(V[end]-1),))
    for i in 1:(length(V)-1), j in V[i]:(V[i+1] - 1); res[j] = i end
    res
end


function diag(A::SparseMatrixCSC{Tv,Ti}, d::Integer=0) where {Tv,Ti}
    m, n = size(A)
    k = Int(d)
    if !(-m <= k <= n)
        throw(ArgumentError(string("requested diagonal, $k, must be at least $(-m) ",
            "and at most $n in an $m-by-$n matrix")))
    end
    l = k < 0 ? min(m+k,n) : min(n-k,m)
    r, c = k <= 0 ? (-k, 0) : (0, k) # start row/col -1
    ind = Vector{Ti}()
    val = Vector{Tv}()
    for i in 1:l
        r += 1; c += 1
        r1 = Int(A.colptr[c])
        r2 = Int(A.colptr[c+1]-1)
        r1 > r2 && continue
        r1 = searchsortedfirst(A.rowval, r, r1, r2, Base.Order.Forward)
        ((r1 > r2) || (A.rowval[r1] != r)) && continue
        push!(ind, i)
        push!(val, A.nzval[r1])
    end
    return SparseVector{Tv,Ti}(l, ind, val)
end

function tr(A::SparseMatrixCSC{Tv}) where Tv
    n = checksquare(A)
    s = zero(Tv)
    for i in 1:n
        s += A[i,i]
    end
    return s
end


## rotations

function rot180(A::SparseMatrixCSC)
    I,J,V = findnz(A)
    m,n = size(A)
    for i=1:length(I)
        I[i] = m - I[i] + 1
        J[i] = n - J[i] + 1
    end
    return sparse(I,J,V,m,n)
end

function rotr90(A::SparseMatrixCSC)
    I,J,V = findnz(A)
    m,n = size(A)
    #old col inds are new row inds
    for i=1:length(I)
        I[i] = m - I[i] + 1
    end
    return sparse(J, I, V, n, m)
end

function rotl90(A::SparseMatrixCSC)
    I,J,V = findnz(A)
    m,n = size(A)
    #old row inds are new col inds
    for i=1:length(J)
        J[i] = n - J[i] + 1
    end
    return sparse(J, I, V, n, m)
end


## Uniform matrix arithmetic

(+)(A::SparseMatrixCSC, J::UniformScaling) = A + sparse(J, size(A)...)
(-)(A::SparseMatrixCSC, J::UniformScaling) = A - sparse(J, size(A)...)
(-)(J::UniformScaling, A::SparseMatrixCSC) = sparse(J, size(A)...) - A

# SparseVector

# the rest of real, conj, imag are handled correctly via AbstractArray methods
SparseArrays.@unarymap_nz2z_z2z real Complex
conj(x::SparseVector{<:Complex}) = SparseVector(length(x), copy(SparseArrays.nonzeroinds(x)), conj(nonzeros(x)))
imag(x::AbstractSparseVector{Tv,Ti}) where {Tv<:Real,Ti<:Integer} = SparseVector(length(x), Ti[], Tv[])
SparseArrays.@unarymap_nz2z_z2z imag Complex

### linalg.jl

# Transpose
# (The only sparse matrix structure in base is CSC, so a one-row sparse matrix is worse than dense)
transpose(sv::SparseVector) = Transpose(sv)
adjoint(sv::SparseVector) = Adjoint(sv)

### BLAS Level-1

# axpy

function LinearAlgebra.axpy!(a::Number, x::SparseVectorUnion, y::AbstractVector)
    length(x) == length(y) || throw(DimensionMismatch())
    nzind = SparseArrays.nonzeroinds(x)
    nzval = nonzeros(x)
    m = length(nzind)

    if a == oneunit(a)
        for i = 1:m
            @inbounds ii = nzind[i]
            @inbounds v = nzval[i]
            y[ii] += v
        end
    elseif a == -oneunit(a)
        for i = 1:m
            @inbounds ii = nzind[i]
            @inbounds v = nzval[i]
            y[ii] -= v
        end
    else
        for i = 1:m
            @inbounds ii = nzind[i]
            @inbounds v = nzval[i]
            y[ii] += a * v
        end
    end
    return y
end


# scaling

function rmul!(x::SparseVectorUnion, a::Real)
    rmul!(nonzeros(x), a)
    return x
end
function rmul!(x::SparseVectorUnion, a::Complex)
    rmul!(nonzeros(x), a)
    return x
end
function lmul!(a::Real, x::SparseVectorUnion)
    rmul!(nonzeros(x), a)
    return x
end
function lmul!(a::Complex, x::SparseVectorUnion)
    rmul!(nonzeros(x), a)
    return x
end

(*)(x::SparseVectorUnion, a::Number) = SparseVector(length(x), copy(SparseArrays.nonzeroinds(x)), nonzeros(x) * a)
(*)(a::Number, x::SparseVectorUnion) = SparseVector(length(x), copy(SparseArrays.nonzeroinds(x)), a * nonzeros(x))
(/)(x::SparseVectorUnion, a::Number) = SparseVector(length(x), copy(SparseArrays.nonzeroinds(x)), nonzeros(x) / a)

# dot
function dot(x::AbstractVector{Tx}, y::SparseVectorUnion{Ty}) where {Tx<:Number,Ty<:Number}
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    nzind = SparseArrays.nonzeroinds(y)
    nzval = nonzeros(y)
    s = dot(zero(Tx), zero(Ty))
    for i = 1:length(nzind)
        s += dot(x[nzind[i]], nzval[i])
    end
    return s
end

function dot(x::SparseVectorUnion{Tx}, y::AbstractVector{Ty}) where {Tx<:Number,Ty<:Number}
    n = length(y)
    length(x) == n || throw(DimensionMismatch())
    nzind = SparseArrays.nonzeroinds(x)
    nzval = nonzeros(x)
    s = dot(zero(Tx), zero(Ty))
    @inbounds for i = 1:length(nzind)
        s += dot(nzval[i], y[nzind[i]])
    end
    return s
end

function _spdot(f::Function,
                xj::Int, xj_last::Int, xnzind, xnzval,
                yj::Int, yj_last::Int, ynzind, ynzval)
    # dot product between ranges of non-zeros,
    s = zero(eltype(xnzval)) * zero(eltype(ynzval))
    @inbounds while xj <= xj_last && yj <= yj_last
        ix = xnzind[xj]
        iy = ynzind[yj]
        if ix == iy
            s += f(xnzval[xj], ynzval[yj])
            xj += 1
            yj += 1
        elseif ix < iy
            xj += 1
        else
            yj += 1
        end
    end
    s
end

function dot(x::SparseVectorUnion{<:Number}, y::SparseVectorUnion{<:Number})
    x === y && return sum(abs2, x)
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = SparseArrays.nonzeroinds(x)
    ynzind = SparseArrays.nonzeroinds(y)
    xnzval = nonzeros(x)
    ynzval = nonzeros(y)

    _spdot(dot,
           1, length(xnzind), xnzind, xnzval,
           1, length(ynzind), ynzind, ynzval)
end

norm(x::SparseVectorUnion, p::Real=2) = norm(nonzeros(x), p)

### BLAS-2 / dense A * sparse x -> dense y

# lowrankupdate (BLAS.ger! like)
function LinearAlgebra.lowrankupdate!(A::StridedMatrix, x::AbstractVector, y::SparseVectorUnion, α::Number = 1)
    nzi = SparseArrays.nonzeroinds(y)
    nzv = nonzeros(y)
    @inbounds for (j,v) in zip(nzi,nzv)
        αv = α*conj(v)
        for i in axes(x, 1)
            A[i,j] += x[i]*αv
        end
    end
    return A
end

# * and mul!

function (*)(A::StridedMatrix{Ta}, x::AbstractSparseVector{Tx}) where {Ta,Tx}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, m)
    mul!(y, A, x)
end

mul!(y::AbstractVector{Ty}, A::StridedMatrix, x::AbstractSparseVector{Tx}) where {Tx,Ty} =
    mul!(y, A, x, one(Tx), zero(Ty))

function mul!(y::AbstractVector, A::StridedMatrix, x::AbstractSparseVector, α::Number, β::Number)
    m, n = size(A)
    length(x) == n && length(y) == m || throw(DimensionMismatch())
    m == 0 && return y
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : rmul!(y, β)
    end
    α == zero(α) && return y

    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = nonzeros(x)
    @inbounds for i = 1:length(xnzind)
        v = xnzval[i]
        if v != zero(v)
            j = xnzind[i]
            αv = v * α
            for r = 1:m
                y[r] += A[r,j] * αv
            end
        end
    end
    return y
end

# * and mul!(C, transpose(A), B)

function *(transA::Transpose{<:Any,<:StridedMatrix{Ta}}, x::AbstractSparseVector{Tx}) where {Ta,Tx}
    A = transA.parent
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, n)
    mul!(y, transpose(A), x)
end

mul!(y::AbstractVector{Ty}, transA::Transpose{<:Any,<:StridedMatrix}, x::AbstractSparseVector{Tx}) where {Tx,Ty} =
    (A = transA.parent; mul!(y, transpose(A), x, one(Tx), zero(Ty)))

function mul!(y::AbstractVector, transA::Transpose{<:Any,<:StridedMatrix}, x::AbstractSparseVector, α::Number, β::Number)
    A = transA.parent
    m, n = size(A)
    length(x) == m && length(y) == n || throw(DimensionMismatch())
    n == 0 && return y
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : rmul!(y, β)
    end
    α == zero(α) && return y

    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = nonzeros(x)
    _nnz = length(xnzind)
    _nnz == 0 && return y

    s0 = zero(eltype(A)) * zero(eltype(x))
    @inbounds for j = 1:n
        s = zero(s0)
        for i = 1:_nnz
            s += A[xnzind[i], j] * xnzval[i]
        end
        y[j] += s * α
    end
    return y
end


### BLAS-2 / sparse A * sparse x -> dense y

function densemv(A::SparseMatrixCSC, x::AbstractSparseVector; trans::AbstractChar='N')
    local xlen::Int, ylen::Int
    m, n = size(A)
    if trans == 'N' || trans == 'n'
        xlen = n; ylen = m
    elseif trans == 'T' || trans == 't' || trans == 'C' || trans == 'c'
        xlen = m; ylen = n
    else
        throw(ArgumentError("Invalid trans character $trans"))
    end
    xlen == length(x) || throw(DimensionMismatch())
    T = promote_type(eltype(A), eltype(x))
    y = Vector{T}(undef, ylen)
    if trans == 'N' || trans == 'N'
        mul!(y, A, x)
    elseif trans == 'T' || trans == 't'
        mul!(y, transpose(A), x)
    elseif trans == 'C' || trans == 'c'
        mul!(y, adjoint(A), x)
    else
        throw(ArgumentError("Invalid trans character $trans"))
    end
    y
end

# * and mul!

mul!(y::AbstractVector{Ty}, A::SparseMatrixCSC, x::AbstractSparseVector{Tx}) where {Tx,Ty} =
    mul!(y, A, x, one(Tx), zero(Ty))

function mul!(y::AbstractVector, A::SparseMatrixCSC, x::AbstractSparseVector, α::Number, β::Number)
    m, n = size(A)
    length(x) == n && length(y) == m || throw(DimensionMismatch())
    m == 0 && return y
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : rmul!(y, β)
    end
    α == zero(α) && return y

    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = nonzeros(x)
    Acolptr = A.colptr
    Arowval = A.rowval
    Anzval = A.nzval

    @inbounds for i = 1:length(xnzind)
        v = xnzval[i]
        if v != zero(v)
            αv = v * α
            j = xnzind[i]
            for r = A.colptr[j]:(Acolptr[j+1]-1)
                y[Arowval[r]] += Anzval[r] * αv
            end
        end
    end
    return y
end

# * and *(Tranpose(A), B)

mul!(y::AbstractVector{Ty}, transA::Transpose{<:Any,<:SparseMatrixCSC}, x::AbstractSparseVector{Tx}) where {Tx,Ty} =
    (A = transA.parent; mul!(y, transpose(A), x, one(Tx), zero(Ty)))

mul!(y::AbstractVector, transA::Transpose{<:Any,<:SparseMatrixCSC}, x::AbstractSparseVector, α::Number, β::Number) =
    (A = transA.parent; _At_or_Ac_mul_B!(*, y, A, x, α, β))

mul!(y::AbstractVector{Ty}, adjA::Adjoint{<:Any,<:SparseMatrixCSC}, x::AbstractSparseVector{Tx}) where {Tx,Ty} =
    (A = adjA.parent; mul!(y, adjoint(A), x, one(Tx), zero(Ty)))

mul!(y::AbstractVector, adjA::Adjoint{<:Any,<:SparseMatrixCSC}, x::AbstractSparseVector, α::Number, β::Number) =
    (A = adjA.parent; _At_or_Ac_mul_B!(dot, y, A, x, α, β))

function _At_or_Ac_mul_B!(tfun::Function,
                          y::AbstractVector, A::SparseMatrixCSC, x::AbstractSparseVector,
                          α::Number, β::Number)
    m, n = size(A)
    length(x) == m && length(y) == n || throw(DimensionMismatch())
    n == 0 && return y
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : rmul!(y, β)
    end
    α == zero(α) && return y

    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = nonzeros(x)
    Acolptr = A.colptr
    Arowval = A.rowval
    Anzval = A.nzval
    mx = length(xnzind)

    for j = 1:n
        # s <- dot(A[:,j], x)
        s = _spdot(tfun, Acolptr[j], Acolptr[j+1]-1, Arowval, Anzval,
                   1, mx, xnzind, xnzval)
        @inbounds y[j] += s * α
    end
    return y
end


### BLAS-2 / sparse A * sparse x -> dense y

function *(A::SparseMatrixCSC, x::AbstractSparseVector)
    y = densemv(A, x)
    initcap = min(nnz(A), size(A,1))
    SparseArrays._dense2sparsevec(y, initcap)
end

*(transA::Transpose{<:Any,<:SparseMatrixCSC}, x::AbstractSparseVector) =
    (A = transA.parent; _At_or_Ac_mul_B(*, A, x))

*(adjA::Adjoint{<:Any,<:SparseMatrixCSC}, x::AbstractSparseVector) =
    (A = adjA.parent; _At_or_Ac_mul_B(dot, A, x))

function _At_or_Ac_mul_B(tfun::Function, A::SparseMatrixCSC{TvA,TiA}, x::AbstractSparseVector{TvX,TiX}) where {TvA,TiA,TvX,TiX}
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Tv = promote_type(TvA, TvX)
    Ti = promote_type(TiA, TiX)

    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = nonzeros(x)
    Acolptr = A.colptr
    Arowval = A.rowval
    Anzval = A.nzval
    mx = length(xnzind)

    ynzind = Vector{Ti}(undef, n)
    ynzval = Vector{Tv}(undef, n)

    jr = 0
    for j = 1:n
        s = _spdot(tfun, Acolptr[j], Acolptr[j+1]-1, Arowval, Anzval,
                   1, mx, xnzind, xnzval)
        if s != zero(s)
            jr += 1
            ynzind[jr] = j
            ynzval[jr] = s
        end
    end
    if jr < n
        resize!(ynzind, jr)
        resize!(ynzval, jr)
    end
    SparseVector(n, ynzind, ynzval)
end


# define matrix division operations involving triangular matrices and sparse vectors
# the valid left-division operations are A[t|c]_ldiv_B[!] and \
# the valid right-division operations are A(t|c)_rdiv_B[t|c][!]
# see issue #14005 for discussion of these methods
for isunittri in (true, false), islowertri in (true, false)
    unitstr = isunittri ? "Unit" : ""
    halfstr = islowertri ? "Lower" : "Upper"
    tritype = :(LinearAlgebra.$(Symbol(unitstr, halfstr, "Triangular")))

    # build out-of-place left-division operations
    for (istrans, applyxform, xformtype, xformop) in (
            (false, false, :identity,  :identity),
            (true,  true,  :Transpose, :transpose),
            (true,  true,  :Adjoint,   :adjoint) )

        # broad method where elements are Numbers
        xformtritype = applyxform ? :($xformtype{<:TA,<:$tritype{<:Any,<:AbstractMatrix}}) :
                                    :($tritype{<:TA,<:AbstractMatrix})
        @eval function \(xformA::$xformtritype, b::SparseVector{Tb}) where {TA<:Number,Tb<:Number}
            A = $(applyxform ? :(xformA.parent) : :(xformA) )
            TAb = $(isunittri ?
                :(typeof(zero(TA)*zero(Tb) + zero(TA)*zero(Tb))) :
                :(typeof((zero(TA)*zero(Tb) + zero(TA)*zero(Tb))/one(TA))) )
            LinearAlgebra.ldiv!($xformop(convert(AbstractArray{TAb}, A)), convert(Array{TAb}, b))
        end

        # faster method requiring good view support of the
        # triangular matrix type. hence the StridedMatrix restriction.
        xformtritype = applyxform ? :($xformtype{<:TA,<:$tritype{<:Any,<:StridedMatrix}}) :
                                    :($tritype{<:TA,<:StridedMatrix})
        @eval function \(xformA::$xformtritype, b::SparseVector{Tb}) where {TA<:Number,Tb<:Number}
            A = $(applyxform ? :(xformA.parent) : :(xformA) )
            TAb = $(isunittri ?
                :(typeof(zero(TA)*zero(Tb) + zero(TA)*zero(Tb))) :
                :(typeof((zero(TA)*zero(Tb) + zero(TA)*zero(Tb))/one(TA))) )
            r = convert(Array{TAb}, b)
            # If b has no nonzero entries, then r is necessarily zero. If b has nonzero
            # entries, then the operation involves only b[nzrange], so we extract and
            # operate on solely b[nzrange] for efficiency.
            if nnz(b) != 0
                nzrange = $( (islowertri && !istrans) || (!islowertri && istrans) ?
                    :(b.nzind[1]:b.n) :
                    :(1:b.nzind[end]) )
                nzrangeviewr = view(r, nzrange)
                nzrangeviewA = $tritype(view(A.data, nzrange, nzrange))
                LinearAlgebra.ldiv!($xformop(convert(AbstractArray{TAb}, nzrangeviewA)), nzrangeviewr)
            end
            r
        end

        # fallback where elements are not Numbers
        xformtritype = applyxform ? :($xformtype{<:Any,<:$tritype}) : :($tritype)
        @eval function \(xformA::$xformtritype, b::SparseVector)
            A = $(applyxform ? :(xformA.parent) : :(xformA) )
            LinearAlgebra.ldiv!($xformop(A), copy(b))
        end
    end

    # build in-place left-division operations
    for (istrans, applyxform, xformtype, xformop) in (
            (false, false, :identity,  :identity),
            (true,  true,  :Transpose, :transpose),
            (true,  true,  :Adjoint,   :adjoint) )
        xformtritype = applyxform ? :($xformtype{<:Any,<:$tritype{<:Any,<:StridedMatrix}}) :
                                    :($tritype{<:Any,<:StridedMatrix})

        # the generic in-place left-division methods handle these cases, but
        # we can achieve greater efficiency where the triangular matrix provides
        # good view support. hence the StridedMatrix restriction.
        @eval function ldiv!(xformA::$xformtritype, b::SparseVector)
            A = $(applyxform ? :(xformA.parent) : :(xformA) )
            # If b has no nonzero entries, the result is necessarily zero and this call
            # reduces to a no-op. If b has nonzero entries, then...
            if nnz(b) != 0
                # densify the relevant part of b in one shot rather
                # than potentially repeatedly reallocating during the solve
                $( (islowertri && !istrans) || (!islowertri && istrans) ?
                    :(_densifyfirstnztoend!(b)) :
                    :(_densifystarttolastnz!(b)) )
                # this operation involves only the densified section, so
                # for efficiency we extract and operate on solely that section
                # furthermore we operate on that section as a dense vector
                # such that dispatch has a chance to exploit, e.g., tuned BLAS
                nzrange = $( (islowertri && !istrans) || (!islowertri && istrans) ?
                    :(b.nzind[1]:b.n) :
                    :(1:b.nzind[end]) )
                nzrangeviewbnz = view(b.nzval, nzrange .- (b.nzind[1] - 1))
                nzrangeviewA = $tritype(view(A.data, nzrange, nzrange))
                LinearAlgebra.ldiv!($xformop(nzrangeviewA), nzrangeviewbnz)
            end
            b
        end
    end
end

# helper functions for in-place matrix division operations defined above
"Densifies `x::SparseVector` from its first nonzero (`x[x.nzind[1]]`) through its end (`x[x.n]`)."
function _densifyfirstnztoend!(x::SparseVector)
    # lengthen containers
    oldnnz = nnz(x)
    newnnz = x.n - x.nzind[1] + 1
    resize!(x.nzval, newnnz)
    resize!(x.nzind, newnnz)
    # redistribute nonzero values over lengthened container
    # initialize now-allocated zero values simultaneously
    nextpos = newnnz
    @inbounds for oldpos in oldnnz:-1:1
        nzi = x.nzind[oldpos]
        nzv = x.nzval[oldpos]
        newpos = nzi - x.nzind[1] + 1
        newpos < nextpos && (x.nzval[newpos+1:nextpos] .= 0)
        newpos == oldpos && break
        x.nzval[newpos] = nzv
        nextpos = newpos - 1
    end
    # finally update lengthened nzinds
    x.nzind[2:end] = (x.nzind[1]+1):x.n
    x
end
"Densifies `x::SparseVector` from its beginning (`x[1]`) through its last nonzero (`x[x.nzind[end]]`)."
function _densifystarttolastnz!(x::SparseVector)
    # lengthen containers
    oldnnz = nnz(x)
    newnnz = x.nzind[end]
    resize!(x.nzval, newnnz)
    resize!(x.nzind, newnnz)
    # redistribute nonzero values over lengthened container
    # initialize now-allocated zero values simultaneously
    nextpos = newnnz
    @inbounds for oldpos in oldnnz:-1:1
        nzi = x.nzind[oldpos]
        nzv = x.nzval[oldpos]
        nzi < nextpos && (x.nzval[nzi+1:nextpos] .= 0)
        nzi == oldpos && (nextpos = 0; break)
        x.nzval[nzi] = nzv
        nextpos = nzi - 1
    end
    nextpos > 0 && (x.nzval[1:nextpos] .= 0)
    # finally update lengthened nzinds
    x.nzind[1:newnnz] = 1:newnnz
    x
end


## sparse matrix multiplication

*(A::SparseMatrixCSC{TvA,TiA}, B::SparseMatrixCSC{TvB,TiB}) where {TvA,TiA,TvB,TiB} =
    *(sppromote(A, B)...)
*(A::SparseMatrixCSC{TvA,TiA}, transB::Transpose{<:Any,<:SparseMatrixCSC{TvB,TiB}}) where {TvA,TiA,TvB,TiB} =
    (B = transB.parent; (pA, pB) = sppromote(A, B); *(pA, transpose(pB)))
*(A::SparseMatrixCSC{TvA,TiA}, adjB::Adjoint{<:Any,<:SparseMatrixCSC{TvB,TiB}}) where {TvA,TiA,TvB,TiB} =
    (B = adjB.parent; (pA, pB) = sppromote(A, B); *(pA, adjoint(pB)))
*(transA::Transpose{<:Any,<:SparseMatrixCSC{TvA,TiA}}, B::SparseMatrixCSC{TvB,TiB}) where {TvA,TiA,TvB,TiB} =
    (A = transA.parent; (pA, pB) = sppromote(A, B); *(transpose(pA), pB))
*(adjA::Adjoint{<:Any,<:SparseMatrixCSC{TvA,TiA}}, B::SparseMatrixCSC{TvB,TiB}) where {TvA,TiA,TvB,TiB} =
    (A = adjA.parent; (pA, pB) = sppromote(A, B); *(adjoint(pA), pB))
*(transA::Transpose{<:Any,<:SparseMatrixCSC{TvA,TiA}}, transB::Transpose{<:Any,<:SparseMatrixCSC{TvB,TiB}}) where {TvA,TiA,TvB,TiB} =
    (A = transA.parent; B = transB.parent; (pA, pB) = sppromote(A, B); *(transpose(pA), transpose(pB)))
*(adjA::Adjoint{<:Any,<:SparseMatrixCSC{TvA,TiA}}, adjB::Adjoint{<:Any,<:SparseMatrixCSC{TvB,TiB}}) where {TvA,TiA,TvB,TiB} =
    (A = adjA.parent; B = adjB.parent; (pA, pB) = sppromote(A, B); *(adjoint(pA), adjoint(pB)))

function sppromote(A::SparseMatrixCSC{TvA,TiA}, B::SparseMatrixCSC{TvB,TiB}) where {TvA,TiA,TvB,TiB}
    Tv = promote_type(TvA, TvB)
    Ti = promote_type(TiA, TiB)
    A  = convert(SparseMatrixCSC{Tv,Ti}, A)
    B  = convert(SparseMatrixCSC{Tv,Ti}, B)
    A, B
end

# In matrix-vector multiplication, the correct orientation of the vector is assumed.

function mul!(C::StridedVecOrMat, A::SparseMatrixCSC, B::StridedVecOrMat, α::Number, β::Number)
    A.n == size(B, 1) || throw(DimensionMismatch())
    A.m == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            αxj = α*B[col,k]
            for j = A.colptr[col]:(A.colptr[col + 1] - 1)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    C
end
*(A::SparseMatrixCSC{TA,S}, x::StridedVector{Tx}) where {TA,S,Tx} =
    (T = promote_type(TA, Tx); mul!(similar(x, T, A.m), A, x, one(T), zero(T)))
*(A::SparseMatrixCSC{TA,S}, B::StridedMatrix{Tx}) where {TA,S,Tx} =
    (T = promote_type(TA, Tx); mul!(similar(B, T, (A.m, size(B, 2))), A, B, one(T), zero(T)))

function mul!(C::StridedVecOrMat, adjA::Adjoint{<:Any,<:SparseMatrixCSC}, B::StridedVecOrMat, α::Number, β::Number)
    A = adjA.parent
    A.n == size(C, 1) || throw(DimensionMismatch())
    A.m == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            tmp = zero(eltype(C))
            for j = A.colptr[col]:(A.colptr[col + 1] - 1)
                tmp += adjoint(nzv[j])*B[rv[j],k]
            end
            C[col,k] += α*tmp
        end
    end
    C
end
*(adjA::Adjoint{<:Any,<:SparseMatrixCSC{TA,S}}, x::StridedVector{Tx}) where {TA,S,Tx} =
    (A = adjA.parent; T = promote_type(TA, Tx); mul!(similar(x, T, A.n), adjoint(A), x, one(T), zero(T)))
*(adjA::Adjoint{<:Any,<:SparseMatrixCSC{TA,S}}, B::StridedMatrix{Tx}) where {TA,S,Tx} =
    (A = adjA.parent; T = promote_type(TA, Tx); mul!(similar(B, T, (A.n, size(B, 2))), adjoint(A), B, one(T), zero(T)))

function mul!(C::StridedVecOrMat, transA::Transpose{<:Any,<:SparseMatrixCSC}, B::StridedVecOrMat, α::Number, β::Number)
    A = transA.parent
    A.n == size(C, 1) || throw(DimensionMismatch())
    A.m == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            tmp = zero(eltype(C))
            for j = A.colptr[col]:(A.colptr[col + 1] - 1)
                tmp += transpose(nzv[j])*B[rv[j],k]
            end
            C[col,k] += α*tmp
        end
    end
    C
end
*(transA::Transpose{<:Any,<:SparseMatrixCSC{TA,S}}, x::StridedVector{Tx}) where {TA,S,Tx} =
    (A = transA.parent; T = promote_type(TA, Tx); mul!(similar(x, T, A.n), transpose(A), x, one(T), zero(T)))
*(transA::Transpose{<:Any,<:SparseMatrixCSC{TA,S}}, B::StridedMatrix{Tx}) where {TA,S,Tx} =
    (A = transA.parent; T = promote_type(TA, Tx); mul!(similar(B, T, (A.n, size(B, 2))), transpose(A), B, one(T), zero(T)))

# For compatibility with dense multiplication API. Should be deleted when dense multiplication
# API is updated to follow BLAS API.
mul!(C::StridedVecOrMat, A::SparseMatrixCSC, B::StridedVecOrMat) =
    mul!(C, A, B, one(eltype(B)), zero(eltype(C)))
mul!(C::StridedVecOrMat, adjA::Adjoint{<:Any,<:SparseMatrixCSC}, B::StridedVecOrMat) =
    (A = adjA.parent; mul!(C, adjoint(A), B, one(eltype(B)), zero(eltype(C))))
mul!(C::StridedVecOrMat, transA::Transpose{<:Any,<:SparseMatrixCSC}, B::StridedVecOrMat) =
    (A = transA.parent; mul!(C, transpose(A), B, one(eltype(B)), zero(eltype(C))))

function (*)(X::StridedMatrix{TX}, A::SparseMatrixCSC{TvA,TiA}) where {TX,TvA,TiA}
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    Y = zeros(promote_type(TX,TvA), mX, A.n)
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for multivec_row=1:mX, col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        Y[multivec_row, col] += X[multivec_row, rowval[k]] * nzval[k]
    end
    Y
end

function (*)(D::Diagonal, A::SparseMatrixCSC)
    T = Base.promote_op(*, eltype(D), eltype(A))
    mul!(LinearAlgebra.copy_oftype(A, T), D, A)
end
function (*)(A::SparseMatrixCSC, D::Diagonal)
    T = Base.promote_op(*, eltype(D), eltype(A))
    mul!(LinearAlgebra.copy_oftype(A, T), A, D)
end

# Sparse matrix multiplication as described in [Gustavson, 1978]:
# http://dl.acm.org/citation.cfm?id=355796

*(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = spmatmul(A,B)
*(A::SparseMatrixCSC{Tv,Ti}, B::Adjoint{<:Any,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = spmatmul(A, copy(B))
*(A::SparseMatrixCSC{Tv,Ti}, B::Transpose{<:Any,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = spmatmul(A, copy(B))
*(A::Transpose{<:Any,<:SparseMatrixCSC{Tv,Ti}}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = spmatmul(copy(A), B)
*(A::Adjoint{<:Any,<:SparseMatrixCSC{Tv,Ti}}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = spmatmul(copy(A), B)
*(A::Adjoint{<:Any,<:SparseMatrixCSC{Tv,Ti}}, B::Adjoint{<:Any,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = spmatmul(copy(A), copy(B))
*(A::Transpose{<:Any,<:SparseMatrixCSC{Tv,Ti}}, B::Transpose{<:Any,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = spmatmul(copy(A), copy(B))

function spmatmul(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti};
                  sortindices::Symbol = :sortcols) where {Tv,Ti}
    mA, nA = size(A)
    mB, nB = size(B)
    nA==mB || throw(DimensionMismatch())

    colptrA = A.colptr; rowvalA = A.rowval; nzvalA = A.nzval
    colptrB = B.colptr; rowvalB = B.rowval; nzvalB = B.nzval
    # TODO: Need better estimation of result space
    nnzC = min(mA*nB, length(nzvalA) + length(nzvalB))
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)

    @inbounds begin
        ip = 1
        xb = zeros(Ti, mA)
        x  = zeros(Tv, mA)
        for i in 1:nB
            if ip + mA - 1 > nnzC
                resize!(rowvalC, nnzC + max(nnzC,mA))
                resize!(nzvalC, nnzC + max(nnzC,mA))
                nnzC = length(nzvalC)
            end
            colptrC[i] = ip
            for jp in colptrB[i]:(colptrB[i+1] - 1)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in colptrA[j]:(colptrA[j+1] - 1)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k] != i
                        rowvalC[ip] = k
                        ip += 1
                        xb[k] = i
                        x[k] = nzC
                    else
                        x[k] += nzC
                    end
                end
            end
            for vp in colptrC[i]:(ip - 1)
                nzvalC[vp] = x[rowvalC[vp]]
            end
        end
        colptrC[nB+1] = ip
    end

    deleteat!(rowvalC, colptrC[end]:length(rowvalC))
    deleteat!(nzvalC, colptrC[end]:length(nzvalC))

    # The Gustavson algorithm does not guarantee the product to have sorted row indices.
    Cunsorted = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    C = SparseArrays.sortSparseMatrixCSC!(Cunsorted, sortindices=sortindices)
    return C
end

# Frobenius dot/inner product: trace(A'B)
function dot(A::SparseMatrixCSC{T1,S1},B::SparseMatrixCSC{T2,S2}) where {T1,T2,S1,S2}
    m, n = size(A)
    size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
    r = dot(zero(T1), zero(T2))
    @inbounds for j = 1:n
        ia = A.colptr[j]; ia_nxt = A.colptr[j+1]
        ib = B.colptr[j]; ib_nxt = B.colptr[j+1]
        if ia < ia_nxt && ib < ib_nxt
            ra = A.rowval[ia]; rb = B.rowval[ib]
            while true
                if ra < rb
                    ia += oneunit(S1)
                    ia < ia_nxt || break
                    ra = A.rowval[ia]
                elseif ra > rb
                    ib += oneunit(S2)
                    ib < ib_nxt || break
                    rb = B.rowval[ib]
                else # ra == rb
                    r += dot(A.nzval[ia], B.nzval[ib])
                    ia += oneunit(S1); ib += oneunit(S2)
                    ia < ia_nxt && ib < ib_nxt || break
                    ra = A.rowval[ia]; rb = B.rowval[ib]
                end
            end
        end
    end
    return r
end

## solvers
function fwdTriSolve!(A::SparseMatrixCSCUnion, B::AbstractVecOrMat)
# forward substitution for CSC matrices
    nrowB, ncolB  = size(B, 1), size(B, 2)
    ncol = LinearAlgebra.checksquare(A)
    if nrowB != ncol
        throw(DimensionMismatch("A is $(ncol) columns and B has $(nrowB) rows"))
    end

    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)

    joff = 0
    for k = 1:ncolB
        for j = 1:nrowB
            i1 = ia[j]
            i2 = ia[j + 1] - 1

            # loop through the structural zeros
            ii = i1
            jai = ja[ii]
            while ii <= i2 && jai < j
                ii += 1
                jai = ja[ii]
            end

            # check for zero pivot and divide with pivot
            if jai == j
                bj = B[joff + jai]/aa[ii]
                B[joff + jai] = bj
                ii += 1
            else
                throw(LinearAlgebra.SingularException(j))
            end

            # update remaining part
            for i = ii:i2
                B[joff + ja[i]] -= bj*aa[i]
            end
        end
        joff += nrowB
    end
    B
end

function bwdTriSolve!(A::SparseMatrixCSCUnion, B::AbstractVecOrMat)
# backward substitution for CSC matrices
    nrowB, ncolB = size(B, 1), size(B, 2)
    ncol = LinearAlgebra.checksquare(A)
    if nrowB != ncol
        throw(DimensionMismatch("A is $(ncol) columns and B has $(nrowB) rows"))
    end

    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)

    joff = 0
    for k = 1:ncolB
        for j = nrowB:-1:1
            i1 = ia[j]
            i2 = ia[j + 1] - 1

            # loop through the structural zeros
            ii = i2
            jai = ja[ii]
            while ii >= i1 && jai > j
                ii -= 1
                jai = ja[ii]
            end

            # check for zero pivot and divide with pivot
            if jai == j
                bj = B[joff + jai]/aa[ii]
                B[joff + jai] = bj
                ii -= 1
            else
                throw(LinearAlgebra.SingularException(j))
            end

            # update remaining part
            for i = ii:-1:i1
                B[joff + ja[i]] -= bj*aa[i]
            end
        end
        joff += nrowB
    end
    B
end

ldiv!(L::LowerTriangular{T,<:SparseMatrixCSCUnion{T}}, B::StridedVecOrMat) where {T} = fwdTriSolve!(L.data, B)
ldiv!(U::UpperTriangular{T,<:SparseMatrixCSCUnion{T}}, B::StridedVecOrMat) where {T} = bwdTriSolve!(U.data, B)

(\)(L::LowerTriangular{T,<:SparseMatrixCSCUnion{T}}, B::SparseMatrixCSC) where {T} = ldiv!(L, Array(B))
(\)(U::UpperTriangular{T,<:SparseMatrixCSCUnion{T}}, B::SparseMatrixCSC) where {T} = ldiv!(U, Array(B))
\(A::Transpose{<:Real,<:Hermitian{<:Real,<:SparseMatrixCSC}}, B::Vector) = A.parent \ B
\(A::Transpose{<:Complex,<:Hermitian{<:Complex,<:SparseMatrixCSC}}, B::Vector) = copy(A) \ B
\(A::Transpose{<:Number,<:Symmetric{<:Number,<:SparseMatrixCSC}}, B::Vector) = A.parent \ B

function rdiv!(A::SparseMatrixCSC{T}, D::Diagonal{T}) where T
    dd = D.diag
    if (k = length(dd)) ≠ A.n
        throw(DimensionMismatch("size(A, 2)=$(A.n) should be size(D, 1)=$k"))
    end
    nonz = nonzeros(A)
    @inbounds for j in 1:k
        ddj = dd[j]
        if iszero(ddj)
            throw(LinearAlgebra.SingularException(j))
        end
        for i in nzrange(A, j)
            nonz[i] /= ddj
        end
    end
    A
end

rdiv!(A::SparseMatrixCSC{T}, adjD::Adjoint{<:Any,<:Diagonal{T}}) where {T} =
    (D = adjD.parent; rdiv!(A, conj(D)))
rdiv!(A::SparseMatrixCSC{T}, transD::Transpose{<:Any,<:Diagonal{T}}) where {T} =
    (D = transD.parent; rdiv!(A, D))

## triu, tril

function triu(S::SparseMatrixCSC{Tv,Ti}, k::Integer=0) where {Tv,Ti}
    m,n = size(S)
    if !(-m + 1 <= k <= n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-m + 1) and at most $(n + 1) in an $m-by-$n matrix")))
    end
    colptr = Vector{Ti}(undef, n+1)
    nnz = 0
    for col = 1 : min(max(k+1,1), n+1)
        colptr[col] = 1
    end
    for col = max(k+1,1) : n
        for c1 = S.colptr[col] : S.colptr[col+1]-1
            S.rowval[c1] > col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    A = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    for col = max(k+1,1) : n
        c1 = S.colptr[col]
        for c2 = A.colptr[col] : A.colptr[col+1]-1
            A.rowval[c2] = S.rowval[c1]
            A.nzval[c2] = S.nzval[c1]
            c1 += 1
        end
    end
    A
end

function tril(S::SparseMatrixCSC{Tv,Ti}, k::Integer=0) where {Tv,Ti}
    m,n = size(S)
    if !(-m - 1 <= k <= n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-m - 1) and at most $(n - 1) in an $m-by-$n matrix")))
    end
    colptr = Vector{Ti}(undef, n+1)
    nnz = 0
    colptr[1] = 1
    for col = 1 : min(n, m+k)
        l1 = S.colptr[col+1]-1
        for c1 = 0 : (l1 - S.colptr[col])
            S.rowval[l1 - c1] < col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    for col = max(min(n, m+k)+2,1) : n+1
        colptr[col] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    A = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    for col = 1 : min(n, m+k)
        c1 = S.colptr[col+1]-1
        l2 = A.colptr[col+1]-1
        for c2 = 0 : l2 - A.colptr[col]
            A.rowval[l2 - c2] = S.rowval[c1]
            A.nzval[l2 - c2] = S.nzval[c1]
            c1 -= 1
        end
    end
    A
end

## diff

function sparse_diff1(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m,n = size(S)
    m > 1 || return SparseMatrixCSC(0, n, fill(one(Ti),n+1), Ti[], Tv[])
    colptr = Vector{Ti}(undef, n+1)
    numnz = 2 * nnz(S) # upper bound; will shrink later
    rowval = Vector{Ti}(undef, numnz)
    nzval = Vector{Tv}(undef, numnz)
    numnz = 0
    colptr[1] = 1
    for col = 1 : n
        last_row = 0
        last_val = 0
        for k = S.colptr[col] : S.colptr[col+1]-1
            row = S.rowval[k]
            val = S.nzval[k]
            if row > 1
                if row == last_row + 1
                    nzval[numnz] += val
                    nzval[numnz]==zero(Tv) && (numnz -= 1)
                else
                    numnz += 1
                    rowval[numnz] = row - 1
                    nzval[numnz] = val
                end
            end
            if row < m
                numnz += 1
                rowval[numnz] = row
                nzval[numnz] = -val
            end
            last_row = row
            last_val = val
        end
        colptr[col+1] = numnz+1
    end
    deleteat!(rowval, numnz+1:length(rowval))
    deleteat!(nzval, numnz+1:length(nzval))
    return SparseMatrixCSC(m-1, n, colptr, rowval, nzval)
end

function sparse_diff2(a::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m,n = size(a)
    colptr = Vector{Ti}(undef, max(n,1))
    numnz = 2 * nnz(a) # upper bound; will shrink later
    rowval = Vector{Ti}(undef, numnz)
    nzval = Vector{Tv}(undef, numnz)

    z = zero(Tv)

    colptr_a = a.colptr
    rowval_a = a.rowval
    nzval_a = a.nzval

    ptrS = 1
    colptr[1] = 1

    n == 0 && return SparseMatrixCSC(m, n, colptr, rowval, nzval)

    startA = colptr_a[1]
    stopA = colptr_a[2]

    rA = startA : stopA - 1
    rowvalA = rowval_a[rA]
    nzvalA = nzval_a[rA]
    lA = stopA - startA

    for col = 1:n-1
        startB, stopB = startA, stopA
        startA = colptr_a[col+1]
        stopA = colptr_a[col+2]

        rowvalB = rowvalA
        nzvalB = nzvalA
        lB = lA

        rA = startA : stopA - 1
        rowvalA = rowval_a[rA]
        nzvalA = nzval_a[rA]
        lA = stopA - startA

        ptrB = 1
        ptrA = 1

        while ptrA <= lA && ptrB <= lB
            rowA = rowvalA[ptrA]
            rowB = rowvalB[ptrB]
            if rowA < rowB
                rowval[ptrS] = rowA
                nzval[ptrS] = nzvalA[ptrA]
                ptrS += 1
                ptrA += 1
            elseif rowB < rowA
                rowval[ptrS] = rowB
                nzval[ptrS] = -nzvalB[ptrB]
                ptrS += 1
                ptrB += 1
            else
                res = nzvalA[ptrA] - nzvalB[ptrB]
                if res != z
                    rowval[ptrS] = rowA
                    nzval[ptrS] = res
                    ptrS += 1
                end
                ptrA += 1
                ptrB += 1
            end
        end

        while ptrA <= lA
            rowval[ptrS] = rowvalA[ptrA]
            nzval[ptrS] = nzvalA[ptrA]
            ptrS += 1
            ptrA += 1
        end

        while ptrB <= lB
            rowval[ptrS] = rowvalB[ptrB]
            nzval[ptrS] = -nzvalB[ptrB]
            ptrS += 1
            ptrB += 1
        end

        colptr[col+1] = ptrS
    end
    deleteat!(rowval, ptrS:length(rowval))
    deleteat!(nzval, ptrS:length(nzval))
    return SparseMatrixCSC(m, n-1, colptr, rowval, nzval)
end

diff(a::SparseMatrixCSC; dims::Integer) = dims==1 ? sparse_diff1(a) : sparse_diff2(a)

## norm and rank
norm(A::SparseMatrixCSC, p::Real=2) = norm(view(A.nzval, 1:nnz(A)), p)

function opnorm(A::SparseMatrixCSC, p::Real=2)
    m, n = size(A)
    if m == 0 || n == 0 || isempty(A)
        return float(real(zero(eltype(A))))
    elseif m == 1 || n == 1
        # TODO: compute more efficiently using A.nzval directly
        return opnorm(Array(A), p)
    else
        Tnorm = typeof(float(real(zero(eltype(A)))))
        Tsum = promote_type(Float64,Tnorm)
        if p==1
            nA::Tsum = 0
            for j=1:n
                colSum::Tsum = 0
                for i = A.colptr[j]:A.colptr[j+1]-1
                    colSum += abs(A.nzval[i])
                end
                nA = max(nA, colSum)
            end
            return convert(Tnorm, nA)
        elseif p==2
            throw(ArgumentError("2-norm not yet implemented for sparse matrices. Try opnorm(Array(A)) or opnorm(A, p) where p=1 or Inf."))
        elseif p==Inf
            rowSum = zeros(Tsum,m)
            for i=1:length(A.nzval)
                rowSum[A.rowval[i]] += abs(A.nzval[i])
            end
            return convert(Tnorm, maximum(rowSum))
        end
    end
    throw(ArgumentError("invalid operator p-norm p=$p. Valid: 1, Inf"))
end

# TODO rank

# cond
function cond(A::SparseMatrixCSC, p::Real=2)
    if p == 1
        normAinv = opnormestinv(A)
        normA = opnorm(A, 1)
        return normA * normAinv
    elseif p == Inf
        normAinv = opnormestinv(copy(A'))
        normA = opnorm(A, Inf)
        return normA * normAinv
    elseif p == 2
        throw(ArgumentError("2-norm condition number is not implemented for sparse matrices, try cond(Array(A), 2) instead"))
    else
        throw(ArgumentError("second argument must be either 1 or Inf, got $p"))
    end
end

function opnormestinv(A::SparseMatrixCSC{T}, t::Integer = min(2,maximum(size(A)))) where T
    maxiter = 5
    # Check the input
    n = checksquare(A)
    F = factorize(A)
    if t <= 0
        throw(ArgumentError("number of blocks must be a positive integer"))
    end
    if t > n
        throw(ArgumentError("number of blocks must not be greater than $n"))
    end
    ind = Vector{Int64}(undef, n)
    ind_hist = Vector{Int64}(undef, maxiter * t)

    Ti = typeof(float(zero(T)))

    S = zeros(T <: Real ? Int : Ti, n, t)

    function _rand_pm1!(v)
        for i in eachindex(v)
            v[i] = rand()<0.5 ? 1 : -1
        end
    end

    function _any_abs_eq(v,n::Int)
        for vv in v
            if abs(vv)==n
                return true
            end
        end
        return false
    end

    # Generate the block matrix
    X = Matrix{Ti}(undef, n, t)
    X[1:n,1] .= 1
    for j = 2:t
        while true
            _rand_pm1!(view(X,1:n,j))
            yaux = X[1:n,j]' * X[1:n,1:j-1]
            if !_any_abs_eq(yaux,n)
                break
            end
        end
    end
    rmul!(X, inv(n))

    iter = 0
    local est
    local est_old
    est_ind = 0
    while iter < maxiter
        iter += 1
        Y = F \ X
        est = zero(real(eltype(Y)))
        est_ind = 0
        for i = 1:t
            y = norm(Y[1:n,i], 1)
            if y > est
                est = y
                est_ind = i
            end
        end
        if iter == 1
            est_old = est
        end
        if est > est_old || iter == 2
            ind_best = est_ind
        end
        if iter >= 2 && est <= est_old
            est = est_old
            break
        end
        est_old = est
        S_old = copy(S)
        for j = 1:t
            for i = 1:n
                S[i,j] = Y[i,j]==0 ? one(Y[i,j]) : sign(Y[i,j])
            end
        end

        if T <: Real
            # Check whether cols of S are parallel to cols of S or S_old
            for j = 1:t
                while true
                    repeated = false
                    if j > 1
                        saux = S[1:n,j]' * S[1:n,1:j-1]
                        if _any_abs_eq(saux,n)
                            repeated = true
                        end
                    end
                    if !repeated
                        saux2 = S[1:n,j]' * S_old[1:n,1:t]
                        if _any_abs_eq(saux2,n)
                            repeated = true
                        end
                    end
                    if repeated
                        _rand_pm1!(view(S,1:n,j))
                    else
                        break
                    end
                end
            end
        end

        # Use the conjugate transpose
        Z = F' \ S
        h_max = zero(real(eltype(Z)))
        h = zeros(real(eltype(Z)), n)
        h_ind = 0
        for i = 1:n
            h[i] = norm(Z[i,1:t], Inf)
            if h[i] > h_max
                h_max = h[i]
                h_ind = i
            end
            ind[i] = i
        end
        if iter >=2 && ind_best == h_ind
            break
        end
        p = sortperm(h, rev=true)
        h = h[p]
        permute!(ind, p)
        if t > 1
            addcounter = t
            elemcounter = 0
            while addcounter > 0 && elemcounter < n
                elemcounter = elemcounter + 1
                current_element = ind[elemcounter]
                found = false
                for i = 1:t * (iter - 1)
                    if current_element == ind_hist[i]
                        found = true
                        break
                    end
                end
                if !found
                    addcounter = addcounter - 1
                    for i = 1:current_element - 1
                        X[i,t-addcounter] = 0
                    end
                    X[current_element,t-addcounter] = 1
                    for i = current_element + 1:n
                        X[i,t-addcounter] = 0
                    end
                    ind_hist[iter * t - addcounter] = current_element
                else
                    if elemcounter == t && addcounter == t
                        break
                    end
                end
            end
        else
            ind_hist[1:t] = ind[1:t]
            for j = 1:t
                for i = 1:ind[j] - 1
                    X[i,j] = 0
                end
                X[ind[j],j] = 1
                for i = ind[j] + 1:n
                    X[i,j] = 0
                end
            end
        end
    end
    return est
end

## kron

# sparse matrix ⊗ sparse matrix
function kron(A::SparseMatrixCSC{T1,S1}, B::SparseMatrixCSC{T2,S2}) where {T1,S1,T2,S2}
    nnzC = nnz(A)*nnz(B)
    mA, nA = size(A); mB, nB = size(B)
    mC, nC = mA*mB, nA*nB
    colptrC = Vector{promote_type(S1,S2)}(undef, nC+1)
    rowvalC = Vector{promote_type(S1,S2)}(undef, nnzC)
    nzvalC  = Vector{typeof(one(T1)*one(T2))}(undef, nnzC)
    colptrC[1] = 1
    col = 1
    @inbounds for j = 1:nA
        startA = A.colptr[j]
        stopA = A.colptr[j+1] - 1
        lA = stopA - startA + 1
        for i = 1:nB
            startB = B.colptr[i]
            stopB = B.colptr[i+1] - 1
            lB = stopB - startB + 1
            ptr_range = (1:lB) .+ (colptrC[col]-1)
            colptrC[col+1] = colptrC[col] + lA*lB
            col += 1
            for ptrA = startA : stopA
                ptrB = startB
                for ptr = ptr_range
                    rowvalC[ptr] = (A.rowval[ptrA]-1)*mB + B.rowval[ptrB]
                    nzvalC[ptr] = A.nzval[ptrA] * B.nzval[ptrB]
                    ptrB += 1
                end
                ptr_range = ptr_range .+ lB
            end
        end
    end
    return SparseMatrixCSC(mC, nC, colptrC, rowvalC, nzvalC)
end

# sparse vector ⊗ sparse vector
function kron(x::SparseVector{T1,S1}, y::SparseVector{T2,S2}) where {T1,S1,T2,S2}
    nnzx = nnz(x); nnzy = nnz(y)
    nnzz = nnzx*nnzy # number of nonzeros in new vector
    nzind = Vector{promote_type(S1,S2)}(undef, nnzz) # the indices of nonzeros
    nzval = Vector{typeof(one(T1)*one(T2))}(undef, nnzz) # the values of nonzeros
    @inbounds for i = 1:nnzx, j = 1:nnzy
        this_ind = (i-1)*nnzy+j
        nzind[this_ind] = (x.nzind[i]-1)*y.n + y.nzind[j]
        nzval[this_ind] = x.nzval[i] * y.nzval[j]
    end
    return SparseVector(x.n*y.n, nzind, nzval)
end

# sparse matrix ⊗ sparse vector & vice versa
kron(A::SparseMatrixCSC, x::SparseVector) = kron(A, SparseMatrixCSC(x))
kron(x::SparseVector, A::SparseMatrixCSC) = kron(SparseMatrixCSC(x), A)

# sparse vec/mat ⊗ vec/mat and vice versa
kron(A::Union{SparseVector,SparseMatrixCSC}, B::VecOrMat) = kron(A, sparse(B))
kron(A::VecOrMat, B::Union{SparseVector,SparseMatrixCSC}) = kron(sparse(A), B)

## det, inv, cond

inv(A::SparseMatrixCSC) = error("The inverse of a sparse matrix can often be dense and can cause the computer to run out of memory. If you are sure you have enough memory, please convert your matrix to a dense matrix.")

# TODO

## scale methods

# Copy colptr and rowval from one sparse matrix to another
function copyinds!(C::SparseMatrixCSC, A::SparseMatrixCSC)
    if C.colptr !== A.colptr
        resize!(C.colptr, length(A.colptr))
        copyto!(C.colptr, A.colptr)
    end
    if C.rowval !== A.rowval
        resize!(C.rowval, length(A.rowval))
        copyto!(C.rowval, A.rowval)
    end
end

# multiply by diagonal matrix as vector
function mul!(C::SparseMatrixCSC, A::SparseMatrixCSC, D::Diagonal{<:Vector})
    m, n = size(A)
    b    = D.diag
    (n==length(b) && size(A)==size(C)) || throw(DimensionMismatch())
    copyinds!(C, A)
    Cnzval = C.nzval
    Anzval = A.nzval
    resize!(Cnzval, length(Anzval))
    for col = 1:n, p = A.colptr[col]:(A.colptr[col+1]-1)
        @inbounds Cnzval[p] = Anzval[p] * b[col]
    end
    C
end

function mul!(C::SparseMatrixCSC, D::Diagonal{<:Vector}, A::SparseMatrixCSC)
    m, n = size(A)
    b    = D.diag
    (m==length(b) && size(A)==size(C)) || throw(DimensionMismatch())
    copyinds!(C, A)
    Cnzval = C.nzval
    Anzval = A.nzval
    Arowval = A.rowval
    resize!(Cnzval, length(Anzval))
    for col = 1:n, p = A.colptr[col]:(A.colptr[col+1]-1)
        @inbounds Cnzval[p] = Anzval[p] * b[Arowval[p]]
    end
    C
end

function mul!(C::SparseMatrixCSC, A::SparseMatrixCSC, b::Number)
    size(A)==size(C) || throw(DimensionMismatch())
    copyinds!(C, A)
    resize!(C.nzval, length(A.nzval))
    mul!(C.nzval, A.nzval, b)
    C
end

function mul!(C::SparseMatrixCSC, b::Number, A::SparseMatrixCSC)
    size(A)==size(C) || throw(DimensionMismatch())
    copyinds!(C, A)
    resize!(C.nzval, length(A.nzval))
    mul!(C.nzval, b, A.nzval)
    C
end

function rmul!(A::SparseMatrixCSC, b::Number)
    rmul!(A.nzval, b)
    return A
end
function lmul!(b::Number, A::SparseMatrixCSC)
    lmul!(b, A.nzval)
    return A
end

function \(A::SparseMatrixCSC, B::AbstractVecOrMat)
    m, n = size(A)
    if m == n
        if istril(A)
            if istriu(A)
                return \(Diagonal(Vector(diag(A))), B)
            else
                return \(LowerTriangular(A), B)
            end
        elseif istriu(A)
            return \(UpperTriangular(A), B)
        end
        if ishermitian(A)
            return \(Hermitian(A), B)
        end
        return \(lu(A), B)
    else
        return \(qr(A), B)
    end
end
for (xformtype, xformop) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
    @eval begin
        function \(xformA::($xformtype){<:Any,<:SparseMatrixCSC}, B::AbstractVecOrMat)
            A = xformA.parent
            m, n = size(A)
            if m == n
                if istril(A)
                    if istriu(A)
                        return \($xformop(Diagonal(Vector(diag(A)))), B)
                    else
                        return \($xformop(LowerTriangular(A)), B)
                    end
                elseif istriu(A)
                    return \($xformop(UpperTriangular(A)), B)
                end
                if ishermitian(A)
                    return \($xformop(Hermitian(A)), B)
                end
                return \($xformop(lu(A)), B)
            else
                return \($xformop(qr(A)), B)
            end
        end
    end
end

function factorize(A::SparseMatrixCSC)
    m, n = size(A)
    if m == n
        if istril(A)
            if istriu(A)
                return Diagonal(A)
            else
                return LowerTriangular(A)
            end
        elseif istriu(A)
            return UpperTriangular(A)
        end
        if ishermitian(A)
            return factorize(Hermitian(A))
        end
        return lu(A)
    else
        return qr(A)
    end
end

function factorize(A::LinearAlgebra.RealHermSymComplexHerm{Float64,<:SparseMatrixCSC})
    F = cholesky(A; check = false)
    if LinearAlgebra.issuccess(F)
        return F
    else
        ldlt!(F, A)
        return F
    end
end

eigen(A::SparseMatrixCSC) = error("Use IterativeEigensolvers.eigs() instead of eigen() for sparse matrices.")

include("deprecated.jl")

end
