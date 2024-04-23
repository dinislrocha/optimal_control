using Ferrite, FerriteGmsh, FerriteViz, SparseArrays, WGLMakie, LinearAlgebra

dim = 2

include("assembling.jl")

function compute_control_error(dh1 :: DofHandler, u1, dh2::DofHandler, u2)
    N = 1000
    points_ΓC = [Tensor{1,2}([i/N 0]) for i in 0:N];
    ph1 = PointEvalHandler(dh1.grid, points_ΓC);
    ph2 = PointEvalHandler(dh2.grid, points_ΓC);
    u1_eval = Ferrite.get_point_values(ph1, dh1, u1, :u);
    u2_eval = Ferrite.get_point_values(ph2, dh2, u2, :u);
    diff = u1_eval - u2_eval
    return sqrt(( diff'diff)/(u2_eval'u2_eval))
end

function solve_ocp(grid::Grid, f_fun::Function, yd_fun::Function)

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
    

    fun_aux(a,b,c,d) = sparse_vcat(sparse_hcat(a, b), sparse_hcat(c,d));
    lhs_matrix = fun_aux(K[❌dir_idx,❌hom_idx], zeros(❌dir_idx |> length,❌dir_idx |> length), 
    (M+∂M)[❌hom_idx,❌hom_idx], K[❌hom_idx, ❌dir_idx]);

    f = zeros(ndofs(dh));
    yd = zeros(ndofs(dh));
    f = apply_analytical!(f, dh, :u, f_fun);
    yd = apply_analytical!(yd, dh, :u, yd_fun);

    rhs = vcat( (M*f)[❌dir_idx], (M*yd)[❌hom_idx] );
    
    sol = lhs_matrix \ rhs
    
    y = zeros(ndofs(dh));
    y[❌hom_idx] = sol[1:length(❌hom_idx) ];
    
    z = zeros(ndofs(dh));
    z[❌dir_idx] = sol[length(❌hom_idx)+1:length(sol)];
    
    return dh, grid, y
end

function get_rel_error(approx, exact, M)
    diff = approx - exact;
    return sqrt( (diff'*M*diff) / (exact'*M*exact) )
end

function lsq_fit(h, err)
    x, y = log.(h), log.(err)
    A = hcat( ones(length(x)), x')
    par = qr(A) \ y
    return par
end


α = 0.1;

filenames = ["1em1.msh", "8em2.msh", "5em2.msh"];
h_arr = [1e-1 8e-2  5e-2];
fine_mesh_filename = "2em2.msh";

grid_fine = togrid("mixed_bc_dirichlet_control_1/meshes/" * fine_mesh_filename);

#
f_fun = x -> 1;
yd_fun = x ->  1;

dh_fine, grid_fine, u_fine  = solve_ocp(grid_fine, f_fun, yd_fun);

p_idx = reshape_to_nodes(dh_fine, collect(1:ndofs(dh_fine)), :u);
p_idx = p_idx[:] |> invperm;

#generate mass matrix for l2 norm computation
M = create_sparsity_pattern(dh_fine);
ip = Lagrange{dim, RefTetrahedron, 1}();
qrule = QuadratureRule{dim, RefTetrahedron}(2);
cellvalues = CellScalarValues(qrule, ip);
doassemble_M!(M, cellvalues, dh_fine);

# evaluation points for projection
points = [node.x for node in grid_fine.nodes]
rel_err_arr = zeros(0);
rel_err_control_arr = zeros(0);

for filename in filenames
    joined_filename  = "mixed_bc_dirichlet_control_1/meshes/" * filename;
    grid = togrid(joined_filename);
    dh_coarse, grid_coarse, u_coarse = solve_ocp(grid, f_fun, yd_fun);
    #project u_coarse onto fine grid
    ph = PointEvalHandler(grid_coarse, points);
    u_coarse_proj = Ferrite.get_point_values(ph, dh_coarse, u_coarse, :u);
    u_coarse_proj = u_coarse_proj[p_idx];
    rel_err = get_rel_error(u_coarse_proj, u_fine, M)
    rel_err_control = compute_control_error(dh_coarse, u_coarse, dh_fine, u_fine);
    append!(rel_err_arr, rel_err);
    append!(rel_err_control_arr, rel_err_control)
    global plotter = FerriteViz.MakiePlotter(dh_fine,u_coarse_proj-u_fine)
end

plotter = FerriteViz.MakiePlotter(dh_fine, u_fine)
FerriteViz.surface(plotter, field=:u)

# least squares fit
lsq_fit(h_arr, rel_err_control_arr)