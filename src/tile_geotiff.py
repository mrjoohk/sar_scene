import os
import math
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

def tile_geotiff_utm(
    in_tif: str,
    out_dir: str,
    tile_size_m: float = 2000.0,   # 2 km
    prefix: str = "tile",
    pad_partial: bool = False      # True면 가장자리 부족분을 빈 값으로 패딩해서 항상 2km로 저장
):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(in_tif) as src:
        # --- 기본 정보 ---
        if src.crs is None:
            raise ValueError("Input TIFF has no CRS. Must be UTM (meters).")
        if not (src.crs.is_projected):
            raise ValueError(f"Input CRS is not projected: {src.crs}. Need UTM/meter CRS.")

        # 픽셀 해상도 (m/pixel)
        # transform.a: pixel width, transform.e: pixel height(보통 음수)
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)

        # 타일 픽셀 크기
        tile_w_px = int(round(tile_size_m / res_x))
        tile_h_px = int(round(tile_size_m / res_y))
        if tile_w_px <= 0 or tile_h_px <= 0:
            raise ValueError("Computed tile pixel size <= 0. Check raster resolution.")

        width = src.width
        height = src.height

        # 타일 개수
        n_tiles_x = math.ceil(width / tile_w_px)
        n_tiles_y = math.ceil(height / tile_h_px)

        print(f"CRS: {src.crs}")
        print(f"Resolution: {res_x:.3f}m x {res_y:.3f}m")
        print(f"Tile size: {tile_size_m}m -> {tile_w_px}px x {tile_h_px}px")
        print(f"Raster size: {width}px x {height}px -> tiles: {n_tiles_x} x {n_tiles_y} = {n_tiles_x*n_tiles_y}")

        profile = src.profile.copy()

        count_written = 0
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                col_off = tx * tile_w_px
                row_off = ty * tile_h_px

                w = min(tile_w_px, width - col_off)
                h = min(tile_h_px, height - row_off)
                if w <= 0 or h <= 0:
                    continue

                window = Window(col_off=col_off, row_off=row_off, width=w, height=h)

                # 데이터 읽기
                data = src.read(window=window)

                # 가장자리 타일을 항상 동일 크기로 저장하고 싶으면 패딩
                if pad_partial and (w != tile_w_px or h != tile_h_px):
                    import numpy as np
                    nodata = profile.get("nodata", None)
                    if nodata is None:
                        # nodata 없으면 0으로 패딩(필요하면 바꾸세요)
                        nodata = 0
                        profile["nodata"] = nodata

                    padded = np.full((src.count, tile_h_px, tile_w_px), nodata, dtype=data.dtype)
                    padded[:, :h, :w] = data
                    data = padded
                    w, h = tile_w_px, tile_h_px
                    window = Window(col_off=col_off, row_off=row_off, width=w, height=h)

                # 타일 transform 계산
                tile_transform = rasterio.windows.transform(window, src.transform)

                # 저장 프로파일 업데이트
                profile.update({
                    "height": int(h),
                    "width": int(w),
                    "transform": tile_transform
                })

                # 파일명: 타일 인덱스 + UTM 좌표 범위도 같이 넣기(추적 편함)
                # tile bounds
                left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
                out_name = f"{prefix}_r{ty:02d}_c{tx:02d}_E{int(left)}_N{int(bottom)}.tif"
                out_path = os.path.join(out_dir, out_name)

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)

                count_written += 1

        print(f"Done. Wrote {count_written} tiles to: {out_dir}")


if __name__ == "__main__":
    # 예시 실행
    tile_geotiff_utm(
        in_tif="Inje_48km_utm.tif",
        out_dir="tiles_2km",
        tile_size_m=2000.0,
        prefix="utm2km",
        pad_partial=True
    )
