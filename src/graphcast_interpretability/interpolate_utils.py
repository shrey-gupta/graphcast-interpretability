import os
import hashlib
import numpy as np
from graphcast import icosahedral_mesh, grid_mesh_connectivity


class GridMeshMapper:
    """
    Precompute, cache, and apply Grid↔Mesh mappings for GraphCast interpolation.
    """

    def __init__(self, grid_lat, grid_lon, splits=6, cache_dir="./cache", radius=None):
        self.grid_lat = np.asarray(grid_lat)
        self.grid_lon = np.asarray(grid_lon)
        self.splits = splits
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=splits)
        self.radius = radius if radius is not None else self._default_radius(self.mesh)
        self.grid_hash = self._hash_grid(self.grid_lat, self.grid_lon)

    # ------------------------------------------------------------------
    # 🔸 Utility functions
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_grid(lat, lon):
        h = hashlib.sha1()
        h.update(np.asarray(lat).tobytes())
        h.update(np.asarray(lon).tobytes())
        return h.hexdigest()[:8]

    @staticmethod
    def _default_radius(mesh):
        faces = mesh.faces
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        v = mesh.vertices
        edge_vecs = v[edges[:, 0]] - v[edges[:, 1]]
        edge_len = np.linalg.norm(edge_vecs, axis=1)
        return 0.6 * edge_len.max()

    # ------------------------------------------------------------------
    # 🔹 Grid → Mesh mapping
    # ------------------------------------------------------------------
    def precompute_grid_to_mesh(self, force=False):
        fname = os.path.join(
            self.cache_dir,
            f"grid2mesh_s{self.splits}_r{self.radius:.5f}_g{self.grid_hash}.npz"
        )
        if os.path.exists(fname) and not force:
            print(f"✅ Using cached G→M mapping: {fname}")
            data = np.load(fname)
            return dict(grid_idx=data["grid_idx"], mesh_idx=data["mesh_idx"], dist=data.get("dist", None))

        print("⏳ Building Grid→Mesh radius query mapping …")
        g_idx, m_idx = grid_mesh_connectivity.radius_query_indices(
            grid_latitude=self.grid_lat, grid_longitude=self.grid_lon,
            mesh=self.mesh, radius=self.radius
        )
        grid_xyz = grid_mesh_connectivity._grid_lat_lon_to_coordinates(
            self.grid_lat, self.grid_lon
        ).reshape(-1, 3)
        d = np.linalg.norm(grid_xyz[g_idx] - self.mesh.vertices[m_idx], axis=1)

        np.savez(fname, grid_idx=g_idx, mesh_idx=m_idx, dist=d)
        print(f"💾 Saved G→M mapping to {fname}")
        return dict(grid_idx=g_idx, mesh_idx=m_idx, dist=d)

    # ------------------------------------------------------------------
    # 🔹 Mesh → Grid mapping
    # ------------------------------------------------------------------
    def precompute_mesh_to_grid(self, force=False):
        fname = os.path.join(
            self.cache_dir,
            f"mesh2grid_s{self.splits}_g{self.grid_hash}.npz"
        )
        if os.path.exists(fname) and not force:
            print(f"✅ Using cached M→G mapping: {fname}")
            data = np.load(fname)
            return dict(grid_idx=data["grid_idx"], mesh_idx=data["mesh_idx"])

        print("⏳ Building Mesh→Grid triangle-containment mapping …")
        g_idx, m_idx = grid_mesh_connectivity.in_mesh_triangle_indices(
            grid_latitude=self.grid_lat, grid_longitude=self.grid_lon, mesh=self.mesh
        )
        np.savez(fname, grid_idx=g_idx, mesh_idx=m_idx)
        print(f"💾 Saved M→G mapping to {fname}")
        return dict(grid_idx=g_idx, mesh_idx=m_idx)

    # ------------------------------------------------------------------
    # 🔹 Quick loader
    # ------------------------------------------------------------------
    def load_mapping(self, kind="grid2mesh"):
        if kind == "grid2mesh":
            fname = os.path.join(
                self.cache_dir,
                f"grid2mesh_s{self.splits}_r{self.radius:.5f}_g{self.grid_hash}.npz"
            )
        elif kind == "mesh2grid":
            fname = os.path.join(
                self.cache_dir,
                f"mesh2grid_s{self.splits}_g{self.grid_hash}.npz"
            )
        else:
            raise ValueError("kind must be 'grid2mesh' or 'mesh2grid'")
        return np.load(fname)

    # ------------------------------------------------------------------
    # 🔹 Fast application helpers
    # ------------------------------------------------------------------
    def apply_grid_to_mesh(self, grid_field, method="mean"):
        """
        Interpolate a lat/lon field onto mesh vertices using the cached mapping.
        """
        mapping = self.load_mapping("grid2mesh")
        g_idx, m_idx = mapping["grid_idx"], mapping["mesh_idx"]
        grid_flat = grid_field.ravel()
        mesh_vals = np.zeros(self.mesh.vertices.shape[0])
        counts = np.zeros_like(mesh_vals)

        if method == "mean":
            np.add.at(mesh_vals, m_idx, grid_flat[g_idx])
            np.add.at(counts, m_idx, 1)
            mesh_vals /= np.maximum(counts, 1)
        elif method == "distance":
            d = mapping.get("dist")
            if d is None:
                raise ValueError("distance weights not precomputed; run precompute_grid_to_mesh(force=True)")
            w = np.exp(-(d / (self.radius + 1e-9)) ** 2)
            np.add.at(mesh_vals, m_idx, w * grid_flat[g_idx])
            np.add.at(counts, m_idx, w)
            mesh_vals /= np.maximum(counts, 1)
        else:
            raise ValueError("method must be 'mean' or 'distance'")
        return mesh_vals

    def apply_mesh_to_grid(self, mesh_field, method="mean"):
        """
        Interpolate a field defined on mesh vertices back to a regular lat/lon grid.
        """
        mapping = self.load_mapping("mesh2grid")
        g_idx, m_idx = mapping["grid_idx"], mapping["mesh_idx"]
        n_grid = self.grid_lat.size * self.grid_lon.size

        # Each grid point is linked to 3 mesh nodes (triangle)
        mesh_triplets = m_idx.reshape(n_grid, 3)
        if method == "mean":
            grid_vals = mesh_field[mesh_triplets].mean(axis=1)
        else:
            raise ValueError("only 'mean' interpolation supported for mesh→grid currently")
        return grid_vals.reshape(len(self.grid_lat), len(self.grid_lon))
