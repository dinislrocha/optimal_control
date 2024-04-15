using Ferrite, FerriteGmsh, FerriteViz, SparseArrays, WGLMakie

dim = 2

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
            if (cellid(cell), face) ∈ getfaceset(grid, "Control")
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


function solve_ocp(grid::Grid)

    ip = Lagrange{dim, RefTetrahedron, 1}()
    qr = QuadratureRule{dim, RefTetrahedron}(2)
    qr_face = QuadratureRule{dim-1, RefTetrahedron}(2)
    cellvalues = CellScalarValues(qr, ip);
    facevalues = FaceScalarValues(qr_face,ip);

    dh = DofHandler(grid)
    add!(dh, :u, 1)
    close!(dh);

    K = create_sparsity_pattern(dh);
    M = create_sparsity_pattern(dh);
    ∂M = create_sparsity_pattern(dh);

    ΓH = getfaceset(dh.grid, "Homogeneous");
    ΓC = setdiff(getfaceset(dh.grid, "Control"), ΓH);
    ΓD = union(ΓH, ΓC);
    
    dch = ConstraintHandler(dh);
    dbc = Dirichlet(:u, ΓH , (x, t) -> 0);
    add!(dch, dbc);
    close!(dch);

    cch = ConstraintHandler(dh);
    cbc = Dirichlet(:u, ΓD, (x,t) -> 0);
    add!(cch, cbc);
    close!(cch);

    doassemble_K!(K, cellvalues, dh);
    doassemble_M!(M, cellvalues, dh);
    doassemble_∂M!(∂M, cellvalues, facevalues, dh);
    
    
    hom_idx = dch.prescribed_dofs;
    ❌hom_idx = setdiff(1:ndofs(dh), hom_idx);
    dir_idx = cch.prescribed_dofs;
    ❌dir_idx = setdiff(1:ndofs(dh), dir_idx);
    
    
    f_data = ones(ndofs(dh));
    
    fun_aux(a,b,c,d) = sparse_vcat(sparse_hcat(a, b), sparse_hcat(c,d));
    lhs_matrix = fun_aux(K[❌dir_idx,❌hom_idx], zeros(❌dir_idx |> length,❌dir_idx |> length), 
    (M+∂M)[❌hom_idx,❌hom_idx], K[❌hom_idx, ❌dir_idx]);
    
    rhs = vcat( (M*f_data)[❌dir_idx], zeros( ❌hom_idx |> length) );
    
    sol = lhs_matrix \ rhs
    
    y = zeros(ndofs(dh));
    y[❌hom_idx] = sol[1:length(❌hom_idx) ];
    
    z = zeros(ndofs(dh));
    z[❌dir_idx] = sol[length(❌hom_idx)+1:length(sol)];
    
    u = y;
    return grid, u
end

filenames = ["1em1.msh", "8em2.msh", "5em2.msh"]


fine_mesh_name = "2em2.msh";

fine_grid = togrid("meshes/" * fine_mesh_name)
fine_grid, fine_u = solve_ocp(fine_grid)



for filename in filenames
    joined_filename  = "meshes/" * filename
    grid = togrid(joined_filename)
    print("clrd")
    solve_ocp(grid)
end



