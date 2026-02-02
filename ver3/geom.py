import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# gazebo에서 px4 특정 모델 고정익 기체를 모의하는 환경을 만들고 동일한 px4 모델을 탑재한 실기체와 실시간 연동해서 HITL 환경을 만들고 싶어, 다음은 내가 생각하는 1) gazebo에서 실기체로 보내는 정보들과  2) 실기체에서 gazebo로 보내는 정보야. 적합한지 확인해줘

# 1) ㅁ

C0 = 299792458.0

def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def plot_radar_trajectory_and_object(
    object_xyz=None,
    tx_traj=None, rx_traj=None,
    bistatic=False,
):
        
    # 2) Top-down(XY) 플롯
    plt.figure(figsize=(7, 7))

    # Radar trajectory (dashed)
    plt.plot(tx_traj[:, 0], tx_traj[:, 1], "--", label="Tx trajectory")
    if bistatic:
        plt.plot(rx_traj[:, 0], rx_traj[:, 1], "--", label="Rx trajectory")

    # Object position (X marker)
    for (x, y, z) in object_xyz:
        plt.scatter([x], [y], marker="o", s=150, linewidths=1)

    # 시작/끝 표시(선택)
    plt.scatter([tx_traj[0, 0]],  [tx_traj[0, 1]],  s=60, label="Tx start")
    plt.scatter([tx_traj[-1, 0]], [tx_traj[-1, 1]], s=60, label="Tx end")
    if bistatic:
        plt.scatter([rx_traj[0, 0]],  [rx_traj[0, 1]],  s=60, label="Rx start")
        plt.scatter([rx_traj[-1, 0]], [rx_traj[-1, 1]], s=60, label="Rx end")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Radar Trajectory (dashed) & Object Position (X)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def place_object_on_terrain(
    obj_mesh: pv.PolyData,
    height_fn,
    # 배치 위치(지형 기준 좌표)
    xyz=(0.0, 0.0, 0.0),
    # 자세(deg)
    rpy_deg=(0.0, 0.0, 0.0),  # (roll, pitch, yaw)
    # 회전 중심: "origin" | "center" | (x,y,z)
    rotate_about="center",
    # 지형면에서 추가로 띄우는 값
    z_offset=0.0
):
    """
    STL 오브젝트에 RPY 자세 + 위치 변환을 적용한 뒤,
    (옵션) 지형 높이에 맞춰 바닥(min z)을 올려 배치한다.

    좌표계 가정:
        - x: 전방, y: 좌/우, z: 상방 (일반적인 ENU처럼 사용)
        - roll  : x축 회전
        - pitch : y축 회전
        - yaw   : z축 회전

    적용 순서(중요):
        1) 회전(yaw -> pitch -> roll) : 바디 고정축 기준의 Tait-Bryan(내재 회전) 느낌으로 구현
            - PyVista는 회전을 누적 적용하므로 호출 순서가 의미 있음
        2) 평면 이동(x,y)
        3) 지형 스냅(ground) : obj의 min z를 지형 높이로 올림
        4) z 추가 이동(xyz[2] + z_offset)
    """
    snap_to_ground = False
    m = obj_mesh.copy(deep=True).triangulate()

    roll, pitch, yaw = rpy_deg
    x0, y0, z0 = xyz

    # 회전 중심점 결정
    if rotate_about == "origin":
        pivot = (0.0, 0.0, 0.0)
    elif rotate_about == "center":
        pivot = tuple(m.center)
    elif isinstance(rotate_about, (tuple, list)) and len(rotate_about) == 3:
        pivot = tuple(rotate_about)
    else:
        raise ValueError("rotate_about must be 'origin', 'center', or (x,y,z)")

    # 1) RPY 회전 적용 (순서 중요)
    # roll: x축, pitch: y축, yaw: z축
    # PyVista의 rotate_x/y/z는 'point=pivot' 기준 회전
    if abs(yaw)   > 1e-12:
        m = m.rotate_z(yaw,   point=pivot, inplace=False)
    if abs(pitch) > 1e-12:
        m = m.rotate_y(pitch, point=pivot, inplace=False)
    if abs(roll)  > 1e-12:
        m = m.rotate_x(roll,  point=pivot, inplace=False)

    # 2) x,y 이동 (z는 스냅에서 처리)
    m = m.translate([x0, y0, 0.0], inplace=False)

    if height_fn is not None:
        snap_to_ground = True
    # 3) 지형 스냅: 오브젝트 바닥(min z)을 지형 높이에 맞춤
    if snap_to_ground:
        z_ground = float(height_fn(x0, y0))
        z_min = m.bounds[4]  # min z after rotation+xy translation
        dz_snap = (z_ground - z_min)
    else:
        dz_snap = 0.0

    # 4) z 이동: 입력 z0 + z_offset + snap
    m = m.translate([0.0, 0.0, dz_snap + z0 + z_offset], inplace=False)

    return m


def mesh_to_facets(mesh: pv.PolyData):
    m = mesh.triangulate()
    m = m.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    centers = np.asarray(m.cell_centers().points)
    normals = unit(np.asarray(m.cell_normals))
    sizes = m.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sizes.cell_data["Area"])
    return m, centers, normals, areas