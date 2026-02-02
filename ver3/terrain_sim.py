import numpy as np
import pyvista as pv

from geom import unit

class TerrainSimulator:
    """
    (2) Terrain roughness map 포함:
      - 지형 생성(또는 향후 DEM 입력으로 확장 가능)
      - facet centers/normals/areas + facet별 roughness 추출
    """
    def __init__(self, xlim=(-200, 200), ylim=(-200, 200), nx=201, ny=201, kind="hill"):
        self.xlim, self.ylim = xlim, ylim
        self.nx, self.ny = nx, ny
        self.kind = kind

        self.terrain_mesh = None
        self.X = self.Y = self.Z = None
        self.x = self.y = None

    def build(self):
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        if self.kind == "flat":
            Z = np.zeros_like(X)
        elif self.kind == "hill":
            Z = 15.0 * np.exp(-((X/120.0)**2 + (Y/120.0)**2))
        elif self.kind == "sine":
            Z = 8.0*np.sin(2*np.pi*X/250.0) * np.cos(2*np.pi*Y/250.0)
        else:
            raise ValueError("unknown terrain kind")

        grid = pv.StructuredGrid(X, Y, Z)
        self.terrain_mesh = grid.extract_surface().triangulate()

        self.x, self.y = x, y
        self.X, self.Y, self.Z = X, Y, Z
        return self.terrain_mesh

    def height_fn(self, xq, yq):
        """
        간단 bilinear interpolation.
        DEM이면 RegularGridInterpolator로 교체 추천.
        """
        x, y, Z = self.x, self.y, self.Z
        ix = np.clip(np.searchsorted(x, xq) - 1, 0, len(x)-2)
        iy = np.clip(np.searchsorted(y, yq) - 1, 0, len(y)-2)

        x0, x1 = x[ix], x[ix+1]
        y0, y1 = y[iy], y[iy+1]

        tx = (xq - x0) / (x1 - x0 + 1e-12)
        ty = (yq - y0) / (y1 - y0 + 1e-12)

        z00 = Z[iy, ix]
        z10 = Z[iy, ix+1]
        z01 = Z[iy+1, ix]
        z11 = Z[iy+1, ix+1]
        return (1-tx)*(1-ty)*z00 + tx*(1-ty)*z10 + (1-tx)*ty*z01 + tx*ty*z11

    def roughness_map(self, X, Y):
        """
        (2) roughness map 예시:
        - 지형 기울기/곡률 기반으로 roughness를 줘서 클러터 강화
        - 값 범위: [0, ~0.2] 정도가 무난(모델에서 specular 퍼짐 조절)
        """
        # 기울기 기반 roughness
        dZdx = np.gradient(self.Z, self.x, axis=1)
        dZdy = np.gradient(self.Z, self.y, axis=0)
        slope = np.sqrt(dZdx**2 + dZdy**2)

        # 정규화 후 스케일링
        slope_n = slope / (np.max(slope) + 1e-12)
        rough = 0.02 + 0.15 * slope_n  # base + slope-based
        return rough

    def facets_with_roughness(self):
        """
        terrain facet: centers/normals/areas + facet별 roughness 반환
        """
        if self.terrain_mesh is None:
            self.build()

        m = self.terrain_mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
        centers = np.asarray(m.cell_centers().points)
        normals = unit(np.asarray(m.cell_normals))

        sizes = m.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sizes.cell_data["Area"])

        # facet center (x,y)에 대해 roughness 값을 계산
        # roughness_map은 grid 기반이지만 여기서는 height_fn처럼 bilinear 샘플링으로 충분
        rough_grid = self.roughness_map(self.X, self.Y)

        x = self.x
        y = self.y

        def sample_rough(xq, yq):
            ix = np.clip(np.searchsorted(x, xq) - 1, 0, len(x)-2)
            iy = np.clip(np.searchsorted(y, yq) - 1, 0, len(y)-2)

            x0, x1 = x[ix], x[ix+1]
            y0, y1 = y[iy], y[iy+1]
            tx = (xq - x0) / (x1 - x0 + 1e-12)
            ty = (yq - y0) / (y1 - y0 + 1e-12)

            r00 = rough_grid[iy, ix]
            r10 = rough_grid[iy, ix+1]
            r01 = rough_grid[iy+1, ix]
            r11 = rough_grid[iy+1, ix+1]
            return (1-tx)*(1-ty)*r00 + tx*(1-ty)*r10 + (1-tx)*ty*r01 + tx*ty*r11

        roughness = np.array([sample_rough(p[0], p[1]) for p in centers], dtype=np.float64)
        return m, centers, normals, areas, roughness
