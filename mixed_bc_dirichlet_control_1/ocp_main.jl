using Ferrite, FerriteGmsh, FerriteViz, SparseArrays, WGLMakie

grid = togrid("newmesh.msh")

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}()
qr = QuadratureRule{dim, RefTetrahedron}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, 1)
close!(dh);

K = create_sparsity_pattern(dh);
M = create_sparsity_pattern(dh);

ch = ConstraintHandler(dh);


Γ = getfaceset(dh.grid, "Boundary");
ΓH = getfaceset(dh.grid, "Homogeneous");
ΓC = setdiff(getfaceset(dh.grid, "Control"), ΓH);
ΓD = union(ΓH, ΓC);
dbc = Dirichlet(:u, ΓD , (x, t) -> 0);
add!(ch, dbc);
close!(ch);

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

function doassemble_∂M(∂M::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    assembler  = start_assemble(∂M)

    for cell in CellIterator(dh)

        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)

            for i in 1:n_basefuncs
                v = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    Me[i,j] += (v*u) * dΩ
                end
            end
        end
    end
end

doassemble_K!(K, cellvalues, dh);
doassemble_M!(M, cellvalues, dh);
f_data = ones(ndofs(dh));

#idxs
free_idx = setdiff(1:ndofs(dh), ch.prescribed_dofs)

# checking method works

temp = M*f_data;
sol = K[free_idx, free_idx] \ temp[free_idx];

n_dofs = ndofs(dh);

# solving biharmonic 
zeros = spzeros()
bi_matrix = [K 0;M K];  