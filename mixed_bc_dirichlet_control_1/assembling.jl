function doassemble_K!(K::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(K)
    for cell in CellIterator(dh)

        fill!(Ke, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end

function doassemble_M!(M::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(M)
    for cell in CellIterator(dh)

        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                v = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues,q_point, j)
                    Me[i, j] += (v*u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Me)
    end
    return M
end

function doassemble_∂M!(∂M::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, facevalues::FaceScalarValues{dim}, dh::DofHandler) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    ∂Me = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(∂M)
    for cell in CellIterator(dh)

        fill!(∂Me, 0)
        for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(dh.grid, "Control")
                reinit!(facevalues, cell, face)
                for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
                    for i in 1:getnbasefunctions(facevalues)
                        u = shape_value(facevalues, q_point, i)
                        for j in 1:getnbasefunctions(facevalues)
                            v = shape_value(facevalues, q_point, j)
                            ∂Me[i,j] += u * v * dΓ
                        end
                    end
                end
            end
        end
        assemble!(assembler, celldofs(cell),  ∂Me)
    end
    return ∂M
end

# the code as is only works for boundary x = 0
# needs to be generalized
function doassemble_∇Γ!(∇Γ::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, facevalues::FaceScalarValues{dim}, dh::DofHandler) where {dim}
    
    n_basefuncs = getnbasefunctions(cellvalues)
    ∇Γe = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(∇Γ)

    for cell in CellIterator(dh)
        do_print = true;
        fill!(∇Γe, 0)
        for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(dh.grid, "Control")
                reinit!(facevalues, cell, face)
                for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
                    for i in 1:getnbasefunctions(facevalues)
                        ∇u = shape_gradient(facevalues, q_point, i)
                        for j in 1:getnbasefunctions(facevalues)
                            ∇v = shape_gradient(facevalues, q_point, j)
                            ∇Γe[i,j] += ∇u[1] * ∇v[1] * dΓ
                        end
                    end
                end
            end
        end
        assemble!(assembler, celldofs(cell),  ∇Γe)
    end
    return ∇Γ
end

