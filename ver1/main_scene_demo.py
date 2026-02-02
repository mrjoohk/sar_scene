import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from geom import make_tx_rx_trajectory
from terrain_sim import TerrainSimulator
from sar_sim import SARSimulator
from bp_gpu import BackprojectionGPU
# from rcs_weight import make_W_of_theta  # 필요 시

def hann_window(n):
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n) / (n-1))

def range_compress_from_freq(s_freq, apply_window=True):
    x = s_freq.copy()
    if apply_window:
        w = hann_window(x.shape[1])
        x *= w[None, :]
    rc = np.fft.ifft(np.fft.ifftshift(x, axes=1), axis=1)
    return rc

def db10(x, eps=1e-30):
    return 10.0 * np.log10(np.maximum(x, eps))

def main():
    stl_path = "F16.stl"

    # 1) Terrain 생성 + roughness map
    terrain = TerrainSimulator(xlim=(-300, 300), ylim=(-300, 300), nx=251, ny=251, kind="hill")
    terrain_mesh = terrain.build()

    # 2) Object 로드 + terrain 위 배치
    obj_raw = pv.read(stl_path).triangulate()
    # 성능 필요 시: obj_raw = obj_raw.decimate(0.7)

    sar = SARSimulator(
        f0=10e9, bandwidth=300e6, n_freq=256,
        roughness_obj=0.04,
        k_spec_obj=1.0, k_diff_obj=0.05,
        k_spec_ter=0.08, k_diff_ter=0.20,   # terrain을 더 diffuse하게 -> clutter 강화
        chunk_facets=50000,
        use_cupy=True
    )

    obj_mesh = sar.place_object_on_terrain(
        obj_mesh=obj_raw,
        height_fn=terrain.height_fn,
        xy=(0.0, 0.0),
        yaw_deg=25.0,
        z_offset=0.0
    )

    # 3) scene mesh merge (object hybrid shadow용)
    scene_mesh = sar.merge_scene_meshes([terrain_mesh, obj_mesh])

    # 4) facets: object / terrain 분리 추출
    _, c_obj, n_obj, a_obj = sar.mesh_to_facets(obj_mesh)
    _, c_ter, n_ter, a_ter, r_ter = terrain.facets_with_roughness()

    print(f"object facets={len(c_obj):,}, terrain facets={len(c_ter):,}")

    # 5) 궤적
    n_pulses = 120
    tx_traj, rx_traj = make_tx_rx_trajectory(
        n_pulses=n_pulses,
        radius=2500.0, z=2500.0,
        start_deg=-30.0, end_deg=30.0,
        bistatic=False
    )

    # (선택) W(theta)
    W = None
    # theta_samples = np.arange(0, 361, 1)
    # rcs_dbsm = ...  # 0~360 배열
    # W = make_W_of_theta(theta_samples, rcs_dbsm)

    # 6) raw data 취득 (object hybrid, terrain fast)
    s_freq, freqs = sar.generate_phase_history_scene(
        centers_obj=c_obj, normals_obj=n_obj, areas_obj=a_obj,
        centers_ter=c_ter, normals_ter=n_ter, areas_ter=a_ter, roughness_ter=r_ter,
        scene_mesh=scene_mesh,
        tx_traj=tx_traj, rx_traj=rx_traj,
        W_of_theta=W,
        hybrid_top_k=8000
    )

    # 7) range compression
    rc = range_compress_from_freq(s_freq, apply_window=True)
    plt.figure(figsize=(7,4))
    plt.plot(db10(np.abs(rc[n_pulses//2])**2))
    plt.title("Range profile (mid pulse)")
    plt.xlabel("Range bin")
    plt.ylabel("Power (dB)")
    plt.tight_layout()
    plt.show()

    # 8) BP (GPU streaming)
    bp = BackprojectionGPU(use_cupy=True, chunk_pixels=250000)
    x_grid = np.linspace(-120, 120, 601)
    y_grid = np.linspace(-120, 120, 601)

    img = bp.bp_2d_streaming(
        s_freq=s_freq, freqs=freqs,
        tx_traj=tx_traj, rx_traj=rx_traj,
        x_grid=x_grid, y_grid=y_grid, z0=0.0
    )

    img_db = db10(np.abs(img)**2)
    plt.figure(figsize=(7,6))
    plt.imshow(img_db, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
               origin="lower", aspect="equal")
    plt.colorbar(label="Intensity (dB)")
    plt.title("SAR BP (Scene) - object hybrid shadow / terrain fast + roughness")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.show()

    # (옵션) 씬 확인
    # plotter = pv.Plotter()
    # plotter.add_mesh(terrain_mesh, color="tan", opacity=0.9, show_edges=False)
    # plotter.add_mesh(obj_mesh, color="lightgray", show_edges=False)
    # plotter.add_axes()
    # plotter.show()

if __name__ == "__main__":
    main()
