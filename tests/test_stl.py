"""
Check things related to STL files
"""

try:
    from . import generic as g
except BaseException:
    import generic as g


class STLTests(g.unittest.TestCase):
    def test_header(self):
        m = g.get_mesh("featuretype.STL")
        # make sure we have the right mesh
        assert g.np.isclose(m.volume, 11.627733431196749, atol=1e-6)

        # should have saved the header from the STL file
        assert len(m.metadata["header"]) > 0

        # should have saved the STL face attributes
        assert len(m.face_attributes["stl"]) == len(m.faces)
        assert len(m.faces) > 1000
        # add a non-correlated face attribute, which should be ignored
        m.face_attributes["nah"] = 10

        # remove all faces except three random ones
        m.update_faces([1, 3, 4])
        # faces and face attributes should be untouched
        assert len(m.faces) == 3
        assert len(m.face_attributes["stl"]) == 3
        # attribute that wasn't len(m.faces) shouldn't have been touched
        assert m.face_attributes["nah"] == 10

    def test_header_overflow(self):
        from trimesh.exchange.stl import HeaderError, load_stl_binary

        blob = g.trimesh.creation.box().export(file_type="stl")
        assert len(load_stl_binary(g.io_wrap(blob))["faces"]) == 12

        # try a file with a wrapped header
        wrap = blob[:80] + g.np.uint32(2**31 + 12).tobytes() + blob[84:]
        assert len(wrap) == len(blob)
        try:
            load_stl_binary(g.io_wrap(wrap))
        except HeaderError:
            return
        raise ValueError("overflow face count passed")

    def test_attrib(self):
        m = g.get_mesh("featuretype.STL")

        len_vertices = len(m.vertices)
        # assign some random vertex attributes
        random = g.random(len(m.vertices))
        m.vertex_attributes["random"] = random
        m.vertex_attributes["nah"] = 20

        # should have saved the STL face attributes
        assert len(m.face_attributes["stl"]) == len(m.faces)
        assert len(m.faces) > 1000
        # add a non-correlated face attribute, which should be ignored
        m.face_attributes["nah"] = 10

        # remove all faces except three random ones
        m.update_faces([1, 3, 4])
        # faces and face attributes should be untouched
        assert len(m.faces) == 3
        assert len(m.face_attributes["stl"]) == 3
        # attribute that wasn't len(m.faces) shouldn't have been touched
        assert m.face_attributes["nah"] == 10

        # check all vertices are still in place
        assert m.vertex_attributes["nah"] == 20
        assert g.np.allclose(random, m.vertex_attributes["random"])
        assert len(m.vertices) == len_vertices

        # remove all vertices except four
        v_mask = [0, 1, 2, 3]
        m.update_vertices(v_mask)
        # make sure things are still correct
        assert m.vertex_attributes["nah"] == 20
        assert g.np.allclose(m.vertex_attributes["random"], random[v_mask])
        assert len(m.vertices) == len(v_mask)

    def test_ascii_multibody(self):
        s = g.get_mesh("multibody.stl")
        assert len(s.geometry) == 2
        assert set(s.geometry.keys()) == {"bodyA", "bodyB"}

    def test_ascii_solid_name(self):
        mesh = g.trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        mesh.metadata = {"name": "solid_A"}
        assert (
            g.trimesh.exchange.stl.export_stl_ascii(mesh).splitlines()[0]
            == "solid solid_A"
        )

    def test_empty(self):
        # demo files to check
        empty_files = ["stl_empty_ascii.stl", "stl_empty_bin.stl"]

        for empty_file in empty_files:
            e = g.get_mesh("emptyIO/" + empty_file)

            # result should be an empty scene without vertices
            assert isinstance(e, g.trimesh.Scene)
            assert not hasattr(e, "vertices")

            # create export
            try:
                e.export(file_type="ply")
            except BaseException:
                return
            raise ValueError("Shouldn't export empty scenes!")

    def test_vertex_order(self):
        for stl in ["featuretype.STL", "ADIS16480.STL", "1002_tray_bottom.STL"]:
            # removing doubles should respect the vertex order
            m_raw = g.get_mesh(stl, process=False)
            m_proc = g.get_mesh(stl, process=True, keep_vertex_order=True)

            verts_raw = g.trimesh.grouping.hashable_rows(m_raw.vertices)
            verts_proc = g.trimesh.grouping.hashable_rows(m_proc.vertices)

            # go through all processed verts
            # find index in unprocessed mesh
            idxs = []
            for vert in verts_proc:
                idxs.append(g.np.where(verts_raw == vert)[0][0])

            # indices should be increasing
            assert (g.np.diff(idxs) > 0).all()

            tris_raw = m_raw.triangles
            tris_proc = m_proc.triangles

            # of course mesh needs to have same faces as before
            assert g.np.allclose(
                g.np.sort(tris_raw, axis=0), g.np.sort(tris_proc, axis=0)
            )

    def test_ascii_keyword(self):
        m = g.get_mesh("ascii.stl.zip", force="mesh")
        assert m.is_watertight

    @staticmethod
    def _winding_normals(mesh):
        # unit normals implied purely by vertex winding
        normals, valid = g.trimesh.triangles.normals(mesh.triangles)
        padded = g.np.zeros((len(mesh.faces), 3))
        padded[valid] = normals
        return padded

    def _corrupted_binary(self):
        """
        Export the 20mm cube to binary STL and corrupt some of the
        stored face normals: flip one, zero one, and make one non-unit.
        """
        from trimesh.exchange.stl import _stl_dtype, _stl_dtype_header

        mesh = g.get_mesh("20mm-xyz-cube.stl", process=False)
        blob = bytearray(mesh.export(file_type="stl"))

        header = _stl_dtype_header.itemsize
        data = g.np.frombuffer(bytes(blob[header:]), dtype=_stl_dtype).copy()
        # flip a normal 180 degrees
        data["normals"][2] *= -1.0
        # zero out a normal
        data["normals"][4] = 0.0
        # correct direction but wildly non-unit magnitude
        data["normals"][6] *= 100.0
        blob[header:] = data.tobytes()
        return bytes(blob), mesh

    def test_binary_bad_normals(self):
        # bad stored normals should be recomputed per-face from the
        # vertex winding rather than trusted or discarded wholesale
        blob, source = self._corrupted_binary()
        m = g.trimesh.load(
            g.io_wrap(blob), file_type="stl", process=False, force="mesh"
        )
        assert len(m.faces) == len(source.faces)

        check = self._winding_normals(m)
        # every normal should now match the winding-order normal
        assert g.np.allclose(m.face_normals, check)
        # and be unit magnitude
        assert g.np.allclose(g.np.linalg.norm(m.face_normals, axis=1), 1.0)
        # vertex normals and dihedral angles derive from face normals
        assert g.np.isfinite(m.vertex_normals).all()
        assert g.np.allclose(
            g.np.linalg.norm(m.vertex_normals, axis=1), 1.0, atol=1e-8
        )
        assert g.np.allclose(
            g.np.abs(m.face_adjacency_angles), g.np.pi / 2, atol=1e-8
        )

    def test_binary_bad_normals_roundtrip(self):
        # exporting a mesh loaded from a corrupt file should write
        # clean normals back out
        blob, _ = self._corrupted_binary()
        m = g.trimesh.load(
            g.io_wrap(blob), file_type="stl", process=False, force="mesh"
        )
        reloaded = g.trimesh.load(
            g.io_wrap(m.export(file_type="stl")),
            file_type="stl",
            process=False,
            force="mesh",
        )
        check = self._winding_normals(reloaded)
        assert g.np.allclose(reloaded.face_normals, check)

    def test_fix_normals_cached(self):
        # fix_normals should repair corrupt normals sitting in the cache
        m = g.get_mesh("20mm-xyz-cube.stl", process=False)
        bad = m.face_normals.copy()
        bad[::2] *= -1.0
        bad[1::2] = 0.0
        # bypass the setter and stuff bad values into the cache
        m._cache["face_normals"] = bad
        assert g.np.allclose(m.face_normals, bad)

        m.fix_normals()
        check = self._winding_normals(m)
        assert g.np.allclose(m.face_normals, check)

    def test_ascii_bad_normals(self):
        # a solid name containing "normal" and facets with missing
        # or corrupt normal lines shouldn't discard every normal
        mesh = g.get_mesh("20mm-xyz-cube.stl", process=False)
        mesh.metadata["name"] = "normal_cover_plate"
        text = mesh.export(file_type="stl_ascii")
        assert text.splitlines()[0] == "solid normal_cover_plate"

        lines = text.splitlines()
        normal_idx = [
            i for i, line in enumerate(lines) if line.startswith("facet normal")
        ]
        assert len(normal_idx) == len(mesh.faces)

        # remove the normal line from one facet entirely
        del lines[normal_idx[3]]
        # flip the normal of another facet
        i = normal_idx[1]
        parts = lines[i].split()
        flipped = [-float(v) for v in parts[2:5]]
        lines[i] = "facet normal {} {} {}".format(*flipped)
        # zero out the normal of a third facet
        i = normal_idx[5]
        lines[i] = "facet normal 0.0 0.0 0.0"

        m = g.trimesh.load(
            g.io_wrap("\n".join(lines).encode("utf-8")),
            file_type="stl",
            process=False,
            force="mesh",
        )
        assert len(m.faces) == len(mesh.faces)
        check = self._winding_normals(m)
        assert g.np.allclose(m.face_normals, check)
        assert g.np.allclose(g.np.linalg.norm(m.face_normals, axis=1), 1.0)

    def test_ascii_slight_normal_kept(self):
        # normals within one degree of the winding normal should be
        # kept (normalized) rather than recomputed
        mesh = g.get_mesh("20mm-xyz-cube.stl", process=False)
        normals = mesh.face_normals.copy()
        # rotate the first normal by a fraction of a degree
        n = normals[0].copy()
        axis = g.np.cross(n, [1.0, 2.0, 3.0])
        axis /= g.np.linalg.norm(axis)
        slight = g.trimesh.transformations.rotation_matrix(
            g.np.radians(0.5), axis
        )[:3, :3].dot(n)
        normals[0] = slight * 7.0

        m = g.trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            face_normals=normals,
            process=False,
        )
        # the slightly-off normal should be preserved but normalized
        assert g.np.allclose(m.face_normals[0], slight)
        assert g.np.allclose(m.face_normals[1:], normals[1:])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
