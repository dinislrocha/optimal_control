import Gmsh:gmsh

gmsh.initialize()
gmsh.model.add("bif")

lc = 1e-3
gmsh.model.geo.addPoint(0., 0., 0, lc, 1);
gmsh.model.geo.addPoint(1., 0., 0, lc, 2);
gmsh.model.geo.addPoint(1., 1., 0, lc, 3);
gmsh.model.geo.addPoint(2., 1., 0, lc, 4);
gmsh.model.geo.addPoint(2., 2., 0, lc, 5);
gmsh.model.geo.addPoint(0., 2., 0, lc, 6);

gmsh.model.geo.addLine(1, 2, 1);
gmsh.model.geo.addLine(2, 3, 2);
gmsh.model.geo.addLine(3, 4, 3);
gmsh.model.geo.addLine(4, 5, 4);
gmsh.model.geo.addLine(5, 6, 5);
gmsh.model.geo.addLine(6, 1, 6);


gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1, 2, 3,4,5,6],-1, "Boundary");
gmsh.model.addPhysicalGroup(1, [2,3,5,6], -1, "Homogeneous");
gmsh.model.addPhysicalGroup(1, [1],-1, "Control");
gmsh.model.addPhysicalGroup(1, [4], -1, "Neumann");
gmsh.model.addPhysicalGroup(2, [1], -1, "My surface")

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)


gmsh.write("meshes/1em2.msh")


if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.finalize()
