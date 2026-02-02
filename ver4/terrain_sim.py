import numpy as np
import pyvista as pv
import rasterio

from geom import unit


class TerrainSimulator:
    """
    TerrainSimulator
    - tif_path (GeoTIFF DEM)이 있으면: DEM 로드 -> PyVista mesh 생성
    - 없으면: kind( flat/hill/sine )로 synthetic 지형 생성
    - facets_with_roughness: facet centers/normals/areas + roughness 샘플
    """

    def __init__(
        self,
        tif_path=None,
        # synthetic 옵션
        xlim=(-200, 200),
        ylim=(-200, 200),
        nx=201,
        ny=201,
        kind="hill",
        # DEM 옵션
        band=1,
        max_size=2048,          # DEM이 너무 크면 downsample
        z_scale=1.0,            # 고도 스케일(예: 과장하려면 2~5)
        xy_scale=1.0,           # 좌표 스케일(UTM meter 아닌 경우 보정)
        fill_nodata="min",      # "min" | "zero" | float
    ):
        self.tif_path = tif_path

        self.xlim, self.ylim = xlim, ylim
        self.nx, self.ny = nx, ny
        self.kind = kind

        self.band = band
        self.max_size = max_size
        self.z_scale = float(z_scale)
        self.xy_scale = float(xy_scale)
        self.fill_nodata = fill_nodata

        self.terrain_mesh = None
        self.C = None
        self.X = self.Y = self.Z = None
        self.x = self.y = None

        # 참고용 메타
        self.crs = None
        self.transform = None
        self.nodata = None

    # -------------------------
    # 내부 유틸
    # -------------------------
    def _downsample_dem(self, Z):
        """max_size 기준으로 단순 stride 다운샘플 (빠르고 의존성 없음)."""
        H, W = Z.shape
        s = int(np.ceil(max(H, W) / self.max_size))
        s = max(s, 1)
        if s == 1:
            return Z, 1
        return Z[::s, ::s], s

    def _fill_nodata(self, Z, nodata):
        if nodata is not None:
            Z = np.where(Z == nodata, np.nan, Z)

        if np.isnan(Z).any():
            if self.fill_nodata == "min":
                v = np.nanmin(Z)
            elif self.fill_nodata == "zero":
                v = 0.0
            elif isinstance(self.fill_nodata, (int, float)):
                v = float(self.fill_nodata)
            else:
                raise ValueError("fill_nodata must be 'min', 'zero', or a number")
            Z = np.nan_to_num(Z, nan=v)
        return Z

    # -------------------------
    # DEM 로드 -> grid 구성
    # -------------------------
    def _build_from_tif(self, tif_path):
        with rasterio.open(tif_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata

            Z = src.read(self.band).astype(np.float32)  # (H,W)
            Z = self._fill_nodata(Z, src.nodata)

            # downsample if needed
            Z, stride = self._downsample_dem(Z)

            # transform 업데이트(스트라이드 반영)
            t = src.transform
            # 픽셀 크기 a,e를 stride만큼 키워줌
            # (Affine: c + col*a, f + row*e)
            # stride 샘플링이면 col, row가 stride 간격이므로 a,e를 stride배
            a = t.a * stride
            e = t.e * stride
            c = t.c
            f = t.f

        H, W = Z.shape

        # 픽셀 중심 좌표
        xs = c + (np.arange(W) + 0.5) * a
        ys = f + (np.arange(H) + 0.5) * e  # e는 보통 음수

        # 사용자 보정 스케일 (UTM 아닌 degree 좌표면 여기에 임의 스케일을 줄 수 있음)
        xs = xs * self.xy_scale
        ys = ys * self.xy_scale

        # (0, 0) 정렬
        xs -= np.mean(xs)
        ys -= np.mean(ys)
        
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = Z * self.z_scale

        grid = pv.StructuredGrid(X, Y, Z)
        mesh = grid.extract_surface().triangulate()

        self.x, self.y = xs, ys
        self.X, self.Y, self.Z = X, Y, Z
        
        self.terrain_mesh = mesh
        return mesh

    # -------------------------
    # synthetic 지형 생성(기존)
    # -------------------------
    def _build_synthetic(self):
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        if self.kind == "flat":
            Z = np.zeros_like(X)
        elif self.kind == "hill":
            Z = 15.0 * np.exp(-((X / 120.0) ** 2 + (Y / 120.0) ** 2))
        elif self.kind == "sine":
            Z = 8.0 * np.sin(2 * np.pi * X / 250.0) * np.cos(2 * np.pi * Y / 250.0)
        else:
            raise ValueError("unknown terrain kind")

        grid = pv.StructuredGrid(X, Y, Z)
        self.terrain_mesh = grid.extract_surface().triangulate()

        self.x, self.y = x, y
        self.X, self.Y, self.Z = X, Y, Z
        return self.terrain_mesh

    # -------------------------
    # public API
    # -------------------------
    def build(self):
        """tif_path가 있으면 DEM, 없으면 synthetic."""
        if self.tif_path:
            return self._build_from_tif(self.tif_path)
        return self._build_synthetic()
    
    def height_fn(self, xq, yq):
        """
        간단 bilinear interpolation.
        DEM이든 synthetic이든 grid 기반일 때 사용.
        """
        if self.Z is None:
            self.build()

        x, y, Z = self.x, self.y[::-1], self.Z

        # scalar 또는 ndarray 모두 지원
        xq = np.asarray(xq)
        yq = np.asarray(yq)

        ix = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
        iy = np.clip(np.searchsorted(y, yq) - 1, 0, len(y) - 2)

        x0, x1 = x[ix], x[ix + 1]
        y0, y1 = y[iy], y[iy + 1]

        tx = (xq - x0) / (x1 - x0 + 1e-12)
        ty = (yq - y0) / (y1 - y0 + 1e-12)

        z00 = Z[iy, ix]
        z10 = Z[iy, ix + 1]
        z01 = Z[iy + 1, ix]
        z11 = Z[iy + 1, ix + 1]
        return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11
    
    def height_fn_idx(self, ix, iy):
        """
        입력: (ix, iy) = 중심(C)을 기준으로 한 인덱스 오프셋
        반환: Z[ iy + Cy, ix + Cx ] 값

        맵 크기: (NY, NX) = self.Z.shape  (주의: Z는 [row(y), col(x)] = [iy, ix])
        중심 인덱스: Cx = NX//2 - 1,  Cy = NY//2 - 1
        """
        if self.Z is None:
            self.build()

        Z = self.Z
        NY, NX = Z.shape  # row-major: (y, x)

        Cx = NX // 2 - 1
        Cy = NY // 2 - 1

        ix = np.asarray(ix)
        iy = np.asarray(iy)

        # 중심 기준 오프셋 -> 절대 인덱스
        jx = ix + Cx
        jy = iy + Cy

        # 경계 클램프 (원하면 wrap/에러로 바꿀 수 있음)
        jx = int(np.clip(jx, 0, NX - 1))
        jy = int(np.clip(jy, 0, NY - 1))

        return Z[jy, jx]

    def roughness_map(self, X, Y):
        """
        roughness map 예시 (기울기 기반)
        """
        if self.Z is None:
            self.build()

        dZdx = np.gradient(self.Z, self.x, axis=1)
        dZdy = np.gradient(self.Z, self.y, axis=0)
        slope = np.sqrt(dZdx**2 + dZdy**2)

        slope_n = slope / (np.max(slope) + 1e-12)
        rough = 0.02 + 0.15 * slope_n
        return rough

    def facets_with_roughness(self):
        """
        terrain facet: centers/normals/areas + facet별 roughness 반환
        """
        if self.terrain_mesh is None:
            self.build()

        m = self.terrain_mesh.compute_normals(
            cell_normals=True,
            point_normals=False,
            auto_orient_normals=True
        )

        centers = np.asarray(m.cell_centers().points)
        normals = unit(np.asarray(m.cell_normals))

        sizes = m.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sizes.cell_data["Area"])

        rough_grid = self.roughness_map(self.X, self.Y)

        x = self.x
        y = self.y

        def sample_rough(xq, yq):
            ix = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
            iy = np.clip(np.searchsorted(y, yq) - 1, 0, len(y) - 2)

            x0, x1 = x[ix], x[ix + 1]
            y0, y1 = y[iy], y[iy + 1]
            tx = (xq - x0) / (x1 - x0 + 1e-12)
            ty = (yq - y0) / (y1 - y0 + 1e-12)

            r00 = rough_grid[iy, ix]
            r10 = rough_grid[iy, ix + 1]
            r01 = rough_grid[iy + 1, ix]
            r11 = rough_grid[iy + 1, ix + 1]
            return (1 - tx) * (1 - ty) * r00 + tx * (1 - ty) * r10 + (1 - tx) * ty * r01 + tx * ty * r11

        roughness = np.array([sample_rough(p[0], p[1]) for p in centers], dtype=np.float64)
        return m, centers, normals, areas, roughness
