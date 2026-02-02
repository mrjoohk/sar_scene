import numpy as np
import pyvista as pv

from geom import C0, unit

class SARSimulator:
    """
    SAR raw data 생성기:
      - object facets: hybrid shadow (fast + topK raytrace)
      - terrain facets: fast shadow만
      - pulse마다 sigma 재계산
      - (C) CuPy chunk streaming으로 phase history 생성
    """
    def __init__(
        self,
        f0=10e9,
        bandwidth=300e6,
        n_freq=256,
        roughness_obj=0.05,
        k_spec_obj=1.0,
        k_diff_obj=0.05,
        k_spec_ter=0.1,
        k_diff_ter=0.15,
        chunk_facets=50000,
        use_baseband=True,
        use_cupy=True,
    ):
        self.f0 = f0
        self.bandwidth = bandwidth
        self.n_freq = n_freq

        self.roughness_obj = roughness_obj
        self.k_spec_obj = k_spec_obj
        self.k_diff_obj = k_diff_obj

        self.k_spec_ter = k_spec_ter
        self.k_diff_ter = k_diff_ter

        self.chunk_facets = chunk_facets
        self.use_cupy = use_cupy

        if (use_baseband):
            self.freqs = np.linspace(-self.bandwidth/2, self.bandwidth/2, self.n_freq)
        else:
            self.freqs = self.f0 + np.linspace(-self.bandwidth/2, self.bandwidth/2, self.n_freq)
        self.wavelength = C0 / self.f0

    # -------------------------
    # facet utilities
    # -------------------------

    @staticmethod
    def mesh_to_facets(mesh: pv.PolyData):
        m = mesh.triangulate()
        m = m.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
        centers = np.asarray(m.cell_centers().points)
        normals = unit(np.asarray(m.cell_normals))
        sizes = m.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sizes.cell_data["Area"])
        return m, centers, normals, areas

    @staticmethod
    def place_object_on_terrain(
        obj_mesh: pv.PolyData,
        height_fn,
        # 배치 위치(지형 기준 좌표)
        xyz=(0.0, 0.0, 0.0),
        # 자세(deg)
        rpy_deg=(0.0, 0.0, 0.0),  # (roll, pitch, yaw)
        # 회전 중심: "origin" | "center" | (x,y,z)
        rotate_about="center",
        # 지형에 "바닥 맞춤" 여부
        snap_to_ground=True,
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
          1) 회전(roll -> pitch -> yaw) : 바디 고정축 기준의 Tait-Bryan(내재 회전) 느낌으로 구현
             - PyVista는 회전을 누적 적용하므로 호출 순서가 의미 있음
          2) 평면 이동(x,y)
          3) 지형 스냅(ground) : obj의 min z를 지형 높이로 올림
          4) z 추가 이동(xyz[2] + z_offset)
        """
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

    @staticmethod
    def merge_scene_meshes(mesh_list):
        merged = mesh_list[0].copy(deep=True)
        for m in mesh_list[1:]:
            merged = merged.merge(m, merge_points=False)
        return merged.triangulate()

    # -------------------------
    # scattering model
    # -------------------------

    def facet_sigma_po_lite(
        self, centers, normals, areas,
        tx_pos, rx_pos,
        roughness, k_spec, k_diff
    ):
        vin  = tx_pos[None, :] - centers
        vout = rx_pos[None, :] - centers
        kin  = unit(vin)
        kout = unit(vout)
        n    = normals

        mu_in = np.sum(n * kin, axis=1)
        mu_in_pos = np.clip(mu_in, 0.0, 1.0)

        k_ref = kin - 2.0 * (np.sum(kin*n, axis=1, keepdims=True)) * n
        align = np.sum(unit(k_ref) * kout, axis=1)
        align_pos = np.clip(align, 0.0, 1.0)

        # roughness↑ -> specular 퍼짐 -> align exponent 감소
        p_align = 30.0 / (1.0 + 20.0*roughness)
        p_mu    = 4.0

        spec = (mu_in_pos**p_mu) * (align_pos**p_align) * (areas**2) * k_spec
        diff = (mu_in_pos) * (areas) * k_diff

        sigma = (spec + diff) / (self.wavelength**2)
        return sigma, mu_in_pos, align_pos

    # -------------------------
    # shadowing
    # -------------------------

    @staticmethod
    def shadow_fast_backface(centers, normals, tx_pos):
        kin = unit(tx_pos[None, :] - centers)
        mu = np.sum(normals * kin, axis=1)
        return (mu > 0.0)

    @staticmethod
    def _build_obb(scene_mesh):
        obb = pv._vtk.vtkOBBTree()
        obb.SetDataSet(scene_mesh)
        obb.BuildLocator()
        return obb

    @staticmethod
    def raytrace_visible_for_indices(obb, tx_pos, centers, idx_list, eps=1e-3):
        tx = tx_pos.astype(float)
        visible = np.ones((len(idx_list),), dtype=bool)

        for k, idx in enumerate(idx_list):
            p = centers[idx].astype(float)
            d = p - tx
            L = np.linalg.norm(d)
            if L < 1e-9:
                visible[k] = False
                continue

            points = pv._vtk.vtkPoints()
            ids = pv._vtk.vtkIdList()
            obb.IntersectWithLine(tx, p, points, ids)

            npts = points.GetNumberOfPoints()
            if npts <= 0:
                continue

            min_dist = np.inf
            for i in range(npts):
                q = np.array(points.GetPoint(i), dtype=float)
                dist = np.linalg.norm(q - tx)
                if dist < min_dist:
                    min_dist = dist

            if min_dist < (L - eps):
                visible[k] = False

        return visible

    def hybrid_shadow_object(
        self,
        scene_mesh,
        centers_obj, normals_obj,
        tx_pos,
        bright_score,
        top_k=8000
    ):
        """
        (1) object만 hybrid:
          - fast(backface)로 후보 거르고
          - 밝기 상위 top_k만 raytrace로 occlusion 정밀화
        """
        vis_fast = self.shadow_fast_backface(centers_obj, normals_obj, tx_pos)
        cand = np.where(vis_fast)[0]
        if cand.size == 0:
            return vis_fast

        scores = bright_score[cand]
        if cand.size > top_k:
            sel = cand[np.argpartition(scores, -top_k)[-top_k:]]
        else:
            sel = cand

        obb = self._build_obb(scene_mesh)
        vis_ray = self.raytrace_visible_for_indices(obb, tx_pos, centers_obj, sel)

        vis = vis_fast.copy()
        vis[sel] = vis[sel] & vis_ray
        return vis
    
    def range_compress(self, s_freq, frequency, fft_length=None, window="Hanning"):
        """
        참고 코드와 동일한 range compression:
        range_profile[p, :] = fftshift(ifft( signal_f[p, :], fft_length ))

        그리고 range_window도 동일:
        df = f[1]-f[0]
        range_extent = c/(2*df)
        range_window = linspace(-0.5*range_extent, +0.5*range_extent, fft_length)

        반환:
        rc: (n_pulses, fft_length) complex
        range_window: (fft_length,) meters
        """
        from numpy.fft import ifft, fftshift

        f = np.asarray(frequency, dtype=np.float64)
        df = float(f[1] - f[0])

        if fft_length is None:
            # 참고 코드처럼 넉넉한 zero-padding(최소 8x, power-of-2)
            n = len(f)
            fft_length = int(8 * 2 ** np.ceil(np.log2(n)))

        # range window (m)
        range_extent = C0 / (2.0 * df)
        range_window = np.linspace(-0.5 * range_extent, 0.5 * range_extent, fft_length)

        x = np.asarray(s_freq, dtype=np.complex128)

        # window 적용(참고 코드의 coefficients와 동일한 취지)
        if window is None or window.lower() == "none":
            w = 1.0
        elif window.lower() == "hanning" or window.lower() == "hann":
            w = np.hanning(x.shape[1])[None, :]
        elif window.lower() == "hamming":
            w = np.hamming(x.shape[1])[None, :]
        else:
            raise ValueError("window must be None/'Hanning'/'Hamming'")

        xw = x * w

        # ifft along frequency axis with zero-padding + fftshift
        rc = fftshift(ifft(xw, n=fft_length, axis=1), axes=1)

        return rc, range_window
    
    # -------------------------
    # phase history (GPU streaming)
    # -------------------------
    def generate_phase_history_points_test(
        self,
        centers,                 # (N,3)
        rcs=None,                # (N,)  (참고 코드의 rt 역할)  없으면 amp/sigma로부터 생성
        amp=None,                # (N,)  (복소/실수 진폭)  rcs와 동시 사용 금지 권장
        sigma=None,              # (N,)  sigma -> amp = sqrt(sigma)
        tx_traj=None, rx_traj=None,
        use_farfield=False,      # True면 far-field 근사(LOS dot)로 빠르게(옵션),
        use_bistatic=False,
        ref_point=None,  # far-field 기준점
    ):
        """
        참고 코드 스타일(deramp / range_center 기반)의 phase history 생성.

        - 주파수축: frequency = linspace(f0, f0+bandwidth, n_freq)  (참고 코드와 동일)
        - 신호:
            S(f,p) = sum_i rcs_i * exp( j * 4π f/c * (R_oneway(i,p) - range_center(p)) )

        여기서 R_oneway = (|Tx-xi| + |Rx-xi|)/2  (monostatic면 그냥 |Tx-xi|)

        use_farfield=True 옵션:
        - 센서가 멀리 있고 scene이 작을 때 LOS 투영(dot)으로 빠르게 생성 가능
        """
        if tx_traj is None or rx_traj is None:
            raise ValueError("tx_traj and rx_traj required")

        centers = np.asarray(centers, dtype=np.float64)
        n_scatter = centers.shape[0]
        n_pulses = tx_traj.shape[0]

        # --- amplitude/rcs 결정 ---
        if rcs is not None:
            a = np.asarray(rcs, dtype=np.complex128)
        elif amp is not None:
            a = np.asarray(amp, dtype=np.complex128)
        elif sigma is not None:
            a = np.sqrt(np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)).astype(np.complex128)
        else:
            a = np.ones((n_scatter,), dtype=np.complex128)

        # --- 참고 코드와 동일한 주파수축(절대 주파수) ---
        f = np.linspace(self.f0, self.f0 + self.bandwidth, self.n_freq).astype(np.float64)
        
        if ref_point is None:
            ref_point = np.mean(centers, axis=0)            

        # --- r0 처리: scalar 또는 per-pulse ---
        # 기본: scene 중심(ref_point)까지의 one-way 거리 기준
        # monostatic이면 |Tx-ref|, bistatic이면 0.5*(|Tx-ref|+|Rx-ref|)
        ref = np.asarray(ref_point, dtype=np.float64)
        
        if use_bistatic:
            Rref = 0.5 * (np.linalg.norm(tx_traj - ref[None, :], axis=1) +
                        np.linalg.norm(rx_traj - ref[None, :], axis=1))
        else:
            Rref = np.linalg.norm(tx_traj - ref[None, :], axis=1)

        use_cupy = self.use_cupy
        if use_cupy:
            try:
                import cupy as cp
            except Exception:
                use_cupy = False

        s_freq = np.zeros((n_pulses, self.n_freq), dtype=np.complex128)

        if use_cupy:
            import cupy as cp
            centers_gpu = cp.asarray(centers, dtype=cp.float64)
            a_gpu = cp.asarray(a, dtype=cp.complex128)
            f_gpu = cp.asarray(f, dtype=cp.float64)
            Rref_gpu = cp.asarray(Rref, dtype=cp.float64)

            # j * 4π f/c
            const = (1j * 4.0 * cp.pi / C0)

            for p in range(n_pulses):
                tx = cp.asarray(tx_traj[p], dtype=cp.float64)
                rx = cp.asarray(rx_traj[p], dtype=cp.float64)

                if use_farfield:
                    # far-field LOS dot approximation (옵션)
                    # LOS는 ref_point 기준으로 정의
                    ref = cp.asarray(ref_point, dtype=cp.float64)
                    los = tx - ref
                    los = los / (cp.linalg.norm(los) + 1e-12)
                    # one-way delta range ≈ dot(los, (xi-ref))
                    d = centers_gpu - ref[None, :]
                    dR = cp.sum(d * los[None, :], axis=1)  # (N,)
                    # range_center는 이미 Rref로 처리하므로, 여기선 dR만 사용
                    phase = cp.exp(const * (f_gpu[None, :] * (dR[:, None])))
                else:
                    Rtx = cp.linalg.norm(centers_gpu - tx[None, :], axis=1)
                    Rrx = cp.linalg.norm(centers_gpu - rx[None, :], axis=1)
                    R_oneway = 0.5 * (Rtx + Rrx)
                    dR = R_oneway - Rref_gpu[p]
                    phase = cp.exp(const * (f_gpu[None, :] * (dR[:, None])))

                s_pf = cp.sum(a_gpu[:, None] * phase, axis=0)
                s_freq[p, :] = cp.asnumpy(s_pf)

        else:
            const = (1j * 4.0 * np.pi / C0)
            for p in range(n_pulses):
                tx = tx_traj[p]
                rx = rx_traj[p]

                if use_farfield:
                    ref = np.asarray(ref_point, dtype=np.float64)
                    los = tx - ref
                    los = los / (np.linalg.norm(los) + 1e-12)
                    d = centers - ref[None, :]
                    dR = np.sum(d * los[None, :], axis=1)
                    phase = np.exp(const * (f[None, :] * (dR[:, None])))
                else:
                    Rtx = np.linalg.norm(centers - tx[None, :], axis=1)
                    Rrx = np.linalg.norm(centers - rx[None, :], axis=1)
                    R_oneway = 0.5 * (Rtx + Rrx)
                    dR = R_oneway - Rref[p]
                    phase = np.exp(const * (f[None, :] * (dR[:, None])))

                s_freq[p, :] = np.sum(a[:, None] * phase, axis=0)

        return s_freq, f, Rref

    # -------------------------
    # phase history (GPU streaming)
    # -------------------------
    def generate_phase_history_points(
        self,
        centers,                 # (N,3)
        rcs=None,                # (N,)  (참고 코드의 rt 역할)  없으면 amp/sigma로부터 생성
        amp=None,                # (N,)  (복소/실수 진폭)  rcs와 동시 사용 금지 권장
        sigma=None,              # (N,)  sigma -> amp = sqrt(sigma)
        tx_traj=None, rx_traj=None,
        range_center=None,       # scalar 또는 (n_pulses,)  [m]  ★ 중요: 기준거리
        use_farfield=False,      # True면 far-field 근사(LOS dot)로 빠르게(옵션)
        ref_point=np.array([0.0, 0.0, 0.0]),  # far-field 기준점
    ):
        """
        참고 코드 스타일(deramp / range_center 기반)의 phase history 생성.

        - 주파수축: frequency = linspace(f0, f0+bandwidth, n_freq)  (참고 코드와 동일)
        - 신호:
            S(f,p) = sum_i rcs_i * exp( j * 4π f/c * (R_oneway(i,p) - range_center(p)) )

        여기서 R_oneway = (|Tx-xi| + |Rx-xi|)/2  (monostatic면 그냥 |Tx-xi|)

        use_farfield=True 옵션:
        - 센서가 멀리 있고 scene이 작을 때 LOS 투영(dot)으로 빠르게 생성 가능
        """
        if tx_traj is None or rx_traj is None:
            raise ValueError("tx_traj and rx_traj required")

        centers = np.asarray(centers, dtype=np.float64)
        n_scatter = centers.shape[0]
        n_pulses = tx_traj.shape[0]

        # --- amplitude/rcs 결정 ---
        if rcs is not None:
            a = np.asarray(rcs, dtype=np.complex128)
        elif amp is not None:
            a = np.asarray(amp, dtype=np.complex128)
        elif sigma is not None:
            a = np.sqrt(np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)).astype(np.complex128)
        else:
            a = np.ones((n_scatter,), dtype=np.complex128)

        # --- 참고 코드와 동일한 주파수축(절대 주파수) ---
        f = np.linspace(self.f0, self.f0 + self.bandwidth, self.n_freq).astype(np.float64)

        # --- range_center 처리: scalar 또는 per-pulse ---
        if range_center is None:
            # 기본: scene 중심(ref_point)까지의 one-way 거리 기준
            # monostatic이면 |Tx-ref|, bistatic이면 0.5*(|Tx-ref|+|Rx-ref|)
            ref = np.asarray(ref_point, dtype=np.float64)
            Rref = 0.5 * (np.linalg.norm(tx_traj - ref[None, :], axis=1) +
                        np.linalg.norm(rx_traj - ref[None, :], axis=1))
        else:
            if np.isscalar(range_center):
                Rref = np.full((n_pulses,), float(range_center), dtype=np.float64)
            else:
                Rref = np.asarray(range_center, dtype=np.float64).reshape(-1)
                if Rref.shape[0] != n_pulses:
                    raise ValueError("range_center length must match number of pulses")

        use_cupy = self.use_cupy
        if use_cupy:
            try:
                import cupy as cp
            except Exception:
                use_cupy = False

        s_freq = np.zeros((n_pulses, self.n_freq), dtype=np.complex128)

        if use_cupy:
            import cupy as cp
            centers_gpu = cp.asarray(centers, dtype=cp.float64)
            a_gpu = cp.asarray(a, dtype=cp.complex128)
            f_gpu = cp.asarray(f, dtype=cp.float64)
            Rref_gpu = cp.asarray(Rref, dtype=cp.float64)

            # j * 4π f/c
            const = (1j * 4.0 * cp.pi / C0)

            for p in range(n_pulses):
                tx = cp.asarray(tx_traj[p], dtype=cp.float64)
                rx = cp.asarray(rx_traj[p], dtype=cp.float64)

                if use_farfield:
                    # far-field LOS dot approximation (옵션)
                    # LOS는 ref_point 기준으로 정의
                    ref = cp.asarray(ref_point, dtype=cp.float64)
                    los = tx - ref
                    los = los / (cp.linalg.norm(los) + 1e-12)
                    # one-way delta range ≈ dot(los, (xi-ref))
                    d = centers_gpu - ref[None, :]
                    dR = cp.sum(d * los[None, :], axis=1)  # (N,)
                    # range_center는 이미 Rref로 처리하므로, 여기선 dR만 사용
                    phase = cp.exp(const * (f_gpu[None, :] * (dR[:, None])))
                else:
                    Rtx = cp.linalg.norm(centers_gpu - tx[None, :], axis=1)
                    Rrx = cp.linalg.norm(centers_gpu - rx[None, :], axis=1)
                    R_oneway = 0.5 * (Rtx + Rrx)
                    dR = R_oneway - Rref_gpu[p]
                    phase = cp.exp(const * (f_gpu[None, :] * (dR[:, None])))

                s_pf = cp.sum(a_gpu[:, None] * phase, axis=0)
                s_freq[p, :] = cp.asnumpy(s_pf)

        else:
            const = (1j * 4.0 * np.pi / C0)
            for p in range(n_pulses):
                tx = tx_traj[p]
                rx = rx_traj[p]

                if use_farfield:
                    ref = np.asarray(ref_point, dtype=np.float64)
                    los = tx - ref
                    los = los / (np.linalg.norm(los) + 1e-12)
                    d = centers - ref[None, :]
                    dR = np.sum(d * los[None, :], axis=1)
                    phase = np.exp(const * (f[None, :] * (dR[:, None])))
                else:
                    Rtx = np.linalg.norm(centers - tx[None, :], axis=1)
                    Rrx = np.linalg.norm(centers - rx[None, :], axis=1)
                    R_oneway = 0.5 * (Rtx + Rrx)
                    dR = R_oneway - Rref[p]
                    phase = np.exp(const * (f[None, :] * (dR[:, None])))

                s_freq[p, :] = np.sum(a[:, None] * phase, axis=0)

        return s_freq, f, Rref


    def generate_phase_history_scene(
        self,
        # object facets
        centers_obj, normals_obj, areas_obj,
        # terrain facets
        centers_ter, normals_ter, areas_ter, roughness_ter,
        # scene mesh (for raytrace)
        scene_mesh,
        # trajectories
        tx_traj, rx_traj,
        # weights
        W_of_theta=None,
        theta_ref_point=np.array([0.0, 0.0, 0.0]),
        # hybrid params
        hybrid_top_k=8000
    ):
        n_pulses = tx_traj.shape[0]

        use_cupy = self.use_cupy
        if use_cupy:
            try:
                import cupy as cp
            except Exception:
                use_cupy = False

        s_freq = np.zeros((n_pulses, self.n_freq), dtype=np.complex128)

        if use_cupy:
            import cupy as cp
            freqs_gpu = cp.asarray(self.freqs, dtype=cp.float64)
            const = (2.0 * cp.pi / C0)

        for p in range(n_pulses):
            tx = tx_traj[p]
            rx = rx_traj[p]

            # ---- (pulse마다) sigma 계산 ----
            sigma_obj, mu_obj, al_obj = self.facet_sigma_po_lite(
                centers_obj, normals_obj, areas_obj,
                tx, rx,
                roughness=self.roughness_obj,
                k_spec=self.k_spec_obj,
                k_diff=self.k_diff_obj
            )

            sigma_ter, mu_ter, al_ter = self.facet_sigma_po_lite(
                centers_ter, normals_ter, areas_ter,
                tx, rx,
                roughness=roughness_ter,   # (2) roughness map 반영
                k_spec=self.k_spec_ter,
                k_diff=self.k_diff_ter
            )

            # ---- (5) W(theta) ----
            if W_of_theta is not None:
                v = theta_ref_point - tx
                theta_deg = (np.degrees(np.arctan2(v[1], v[0])) + 360.0) % 360.0
                w = float(W_of_theta(theta_deg))
                sigma_obj *= w
                sigma_ter *= w

            # ---- (1) shadowing 분리 ----
            # terrain: fast만
            vis_ter = self.shadow_fast_backface(centers_ter, normals_ter, tx)

            # object: hybrid
            bright = sigma_obj  # 밝기 프록시: sigma 자체가 가장 직관적
            vis_obj = self.hybrid_shadow_object(
                scene_mesh=scene_mesh,
                centers_obj=centers_obj,
                normals_obj=normals_obj,
                tx_pos=tx,
                bright_score=bright,
                top_k=hybrid_top_k
            )

            sigma_obj *= vis_obj.astype(np.float64)
            sigma_ter *= vis_ter.astype(np.float64)

            # ---- GPU streaming 누적 ----
            if use_cupy:
                import cupy as cp
                acc = cp.zeros((self.n_freq,), dtype=cp.complex128)

                # object chunk
                acc += self._accumulate_chunks_gpu(cp, acc, centers_obj, sigma_obj, tx, rx, freqs_gpu, const)

                # terrain chunk
                acc += self._accumulate_chunks_gpu(cp, acc, centers_ter, sigma_ter, tx, rx, freqs_gpu, const)

                s_freq[p, :] = cp.asnumpy(acc)
            else:
                acc = np.zeros((self.n_freq,), dtype=np.complex128)
                acc += self._accumulate_chunks_cpu(centers_obj, sigma_obj, tx, rx)
                acc += self._accumulate_chunks_cpu(centers_ter, sigma_ter, tx, rx)
                s_freq[p, :] = acc

            if (p % max(1, n_pulses//10)) == 0:
                print(f"[SAR] pulse {p+1}/{n_pulses}")

        return s_freq, self.freqs

    def _accumulate_chunks_cpu(self, centers, sigma, tx, rx):
        acc = np.zeros((self.n_freq,), dtype=np.complex128)
        for i0 in range(0, centers.shape[0], self.chunk_facets):
            i1 = min(i0 + self.chunk_facets, centers.shape[0])
            s = sigma[i0:i1]
            nz = s > 0
            if not np.any(nz):
                continue
            c = centers[i0:i1][nz]
            s = s[nz]
            Rtx = np.linalg.norm(c - tx[None, :], axis=1)
            Rrx = np.linalg.norm(c - rx[None, :], axis=1)
            Rsum = Rtx + Rrx
            amp = np.sqrt(s)
            phase = np.exp(-1j * (2.0*np.pi/C0) * (Rsum[:, None] * self.freqs[None, :]))
            acc += np.sum(amp[:, None] * phase, axis=0)
        return acc

    def _accumulate_chunks_gpu(self, cp, acc, centers, sigma, tx, rx, freqs_gpu, const):
        """
        (C) GPU chunk streaming
        - centers/sigma는 CPU에 있고 chunk 단위로 GPU로 옮겨 누적
        """
        out = cp.zeros_like(acc)
        for i0 in range(0, centers.shape[0], self.chunk_facets):
            i1 = min(i0 + self.chunk_facets, centers.shape[0])

            s = sigma[i0:i1]
            nz = s > 0
            if not np.any(nz):
                continue

            c = centers[i0:i1][nz]
            s = s[nz]

            c_gpu = cp.asarray(c, dtype=cp.float64)
            tx_gpu = cp.asarray(tx, dtype=cp.float64)
            rx_gpu = cp.asarray(rx, dtype=cp.float64)

            Rtx = cp.linalg.norm(c_gpu - tx_gpu[None, :], axis=1)
            Rrx = cp.linalg.norm(c_gpu - rx_gpu[None, :], axis=1)
            Rsum = Rtx + Rrx

            amp = cp.sqrt(cp.asarray(s, dtype=cp.float64))
            phase = cp.exp(-1j * const * (Rsum[:, None] * freqs_gpu[None, :]))
            out += cp.sum(amp[:, None] * phase, axis=0)
        return out
