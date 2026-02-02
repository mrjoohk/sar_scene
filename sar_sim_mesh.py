import numpy as np
from scipy.constants import speed_of_light, pi
from gpu_functions import interp1d_gpu_uniform, _get_bp_rawkernel, _get_ph_rawkernel_farfield_c64, _get_ph_rawkernel_nearfield_c64
from scipy.fftpack import ifft, fftshift

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
        az_start=-3,
        az_end=3,
        radius=500,
        window_type = "Hanning",
        targets=None,
        rcs=None,
        # for BP imaging
        x_span=50,
        y_span=50,
        chunk_facets=1000,
        use_cupy=False
    ):
        self.f0 = f0
        self.bandwidth = bandwidth
        self.az_start = az_start
        self.az_end = az_end
        self.radius = radius
        self.wavelength = speed_of_light / self.f0        
        self.window_type = window_type
        self.targets = np.array(targets)
        self.rcs = rcs
        
        self.na = 0
        self.nf = 0
        self.fft_length = 0
        self.freq_space = None
        self.az_space = None
        
        self.ref_point = None
        
        self.x_span = x_span
        self.y_span = y_span
        
        self.use_cupy = use_cupy
        self.chunk_facets=chunk_facets
        self.xp = None
        if use_cupy:
            import cupy as cp
            self.xp = cp
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()
        else:
            self.xp = np
            
        self.sensor_xyz = None
            
        self.refine_targets()
        self.set_ref_point()
        self.set_freq_space()
        self.set_az_space()
        self.set_fft_length()
        
    def refine_targets(self):
        mask = (
            (self.targets[:,0] >= -50) & (self.targets[:,0] <= 50) &
            (self.targets[:,1] >= -50) & (self.targets[:,1] <= 50)
        )

        self.targets = self.targets[mask]
            
    def set_ref_point(self):
        """
        타겟점들의 무게중심 (단순 평균 or RCS 가중 무게중심)
        만약 spotlight SAR이면 고정된 ref_point가 아닌 다음과 같이 pulse-dependent ref_point
            ref_point[p] = spotlight_aim_point[p]

        """
        if self.rcs is None:
            self.ref_point = np.mean(self.targets, axis=0)
        else:
            self.ref_point = np.sum(self.targets * self.rcs[:, None], axis=0) / np.sum(self.rcs)
        
    def set_freq_space(self):
        """
        range방향 sampling nyquist 조건 충족
            df <= c / 2r
        """
        r = np.sqrt(self.x_span **2 + self.y_span **2)
        df = speed_of_light / (2.0 * r)
        nf = np.floor(self.bandwidth / df)
        
        self.freq_space = np.linspace(self.f0, self.f0 + self.bandwidth, int(nf))
        self.nf = int(nf)
        
    def set_az_space(self):
        """
        az방향 sampling nyquist 조건 충족
            dphi = ( 4pi / wavelength ) * (장면 내 위치 변화) -> 위상크기 변화
            dR = r * dtheta -> 거리 변화(1차 근사)
                dphi = ( 4pi / wavelength ) * r * dtheta
                - nyquist 조건 (위상 변화 <= phi)
                    ( 4pi / wavelength ) * r * dtheta <= pi
                    -> dtheta <= ( wavelength / 4r ) --> 보통 여유두고 dtheta <= ( wavelength / 2r )
            
        """
        r = np.sqrt(self.x_span **2 + self.y_span **2)
        da = speed_of_light / (2.0 * r * self.f0)
        na = np.round((self.az_end - self.az_start) / da)
        
        self.az_space = np.linspace(self.az_start, self.az_end, int(na))
        self.na = int(na)
    
    def set_fft_length(self):
        self.fft_length = int(8 * 2**self.xp.ceil(self.xp.log2(self.nf)))
    
    def get_sensor_positions(self):
        
        if self.az_space is None:
            self.set_az_space
            
        center = self.ref_point
        print(center.shape)
            
        sensor_x = center[0] + self.radius * np.cos(np.radians(self.az_space))
        sensor_y = center[1] + self.radius * np.sin(np.radians(self.az_space))
        sensor_z = center[2] + self.radius * np.zeros_like(self.az_space)
        sensor_xyz = np.stack((sensor_x, sensor_y, sensor_z), axis=1, dtype=np.float64)
        
        print("sensor xyz shape: ", sensor_xyz.shape)
        return sensor_xyz
    
    def generate_phase_history(self, use_farfield=False):
        signal_freq = self.xp.zeros((int(self.na), int(self.nf)), dtype=complex)
        amp_gpu = self.xp.ones((self.targets.shape[0],), dtype=np.complex128)
        if self.rcs is not None:
            amp_gpu = self.xp.asarray(self.rcs, dtype=np.complex128)
        
        # wavenumber
        kc = self.xp.asarray(1j * 2 * 2 * pi / speed_of_light, dtype=self.xp.complex128)
        traj = self.get_sensor_positions()
        self.sensor_xyz = traj
        ref_point_gpu = self.xp.asarray(self.ref_point, dtype=self.xp.float64) # ref_point_gpu.shape: (3, )
        targets_gpu = self.xp.asarray(self.targets, dtype=self.xp.float64)
        freq_gpu = self.xp.asarray(self.freq_space, dtype=self.xp.float64)
        print(f"number of antenna(pulses) / frequency: {self.na} / {self.nf}")
        
        for ia in range(self.na):
            tx = traj[ia] # tx.shape: (3, )
            rx = traj[ia]
            
            if use_farfield:
                los = tx - ref_point_gpu
                los = los / (self.xp.linalg.norm(los) + 1e-12) # los.shape: (3,)
                
            else:
                Rref_tx = self.xp.linalg.norm(ref_point_gpu - tx)              # scalar
                Rref_rx = self.xp.linalg.norm(ref_point_gpu - rx)              # scalar
                Rref = 0.5 * (Rref_tx + Rref_rx)             
            
            out = self.xp.zeros((self.nf,), dtype=self.xp.complex128)
            for i0 in range(0, targets_gpu.shape[0], self.chunk_facets):
                i1 = min(i0 + self.chunk_facets, targets_gpu.shape[0])
                
                sigma = amp_gpu[i0:i1]
                nz = sigma > 0
                if not self.xp.any(nz):
                    continue
                
                ti = targets_gpu[i0:i1][nz]
                sigma = sigma[nz]
                
                if use_farfield:
                    # one-way delta range ≈ dot(los, (xi-ref))
                    d = ti - ref_point_gpu[None, :] # d.shape: (N, 3) (N: the number of gpu chunk)
                    dR = self.xp.sum(d * los[None, :], axis=1)  # (N,)
                    # range_center는 이미 Rref로 처리하므로, 여기선 dR만 사용
                else:
                    Rtx = self.xp.linalg.norm(ti - tx[None, :], axis=1)
                    Rrx = self.xp.linalg.norm(ti - rx[None, :], axis=1)
                    R_oneway = 0.5 * (Rtx + Rrx)                   # scalar

                    dR = R_oneway - Rref # (N,)
                    
                phase = self.xp.exp(kc * (freq_gpu[None, :] * (dR[:, None])))   # (N targets, nf) -> 매우 큰 매트릭스 생성 -> 병목            
                out += self.xp.sum(sigma[:, None] * phase, axis=0)
                
            signal_freq[ia, :] = out

            if (ia % max(1, int(self.na)//10)) == 0:
                print(f"[PH] {ia+1}/{self.na}")
            
        return signal_freq, freq_gpu, ref_point_gpu
    
    def generate_phase_history_fused(self, use_farfield=False):
        
        na = int(self.na)
        nf = int(self.nf)

        # --- trajectories (CPU) ---
        traj = self.get_sensor_positions()
        self.sensor_xyz = np.asarray(traj, dtype=np.float32)  # (na,3) CPU

        # --- GPU constants/arrays ---
        refp = self.xp.asarray(self.ref_point, dtype=self.xp.float32) # ref_point_gpu.shape: (3, )
        freqs = self.xp.asarray(self.freq_space, dtype=self.xp.float32)

        # targets + amp        
        targets = self.xp.asarray(self.targets, dtype=self.xp.float32)
        Nt = int(targets.shape[0])

        if self.rcs is None:
            sigma = self.xp.ones((Nt,), dtype=self.xp.complex64)
        else:
            sigma = self.xp.asarray(np.asarray(self.rcs, dtype=np.complex64), dtype=self.xp.complex64)

        # output
        signal_freq = self.xp.zeros((na, nf), dtype=self.xp.complex64)

        # kernels
        k_near = _get_ph_rawkernel_nearfield_c64(self.xp)
        k_far  = _get_ph_rawkernel_farfield_c64(self.xp)

        threads = 256
        grid = (nf,)  # one block per frequency bin

        # chunk over targets (메모리/캐시)
        chunk = int(self.chunk_facets)

        # temp buffers for small vectors
        tx_buf  = self.xp.empty((3,), dtype=self.xp.float32)
        los_buf = self.xp.empty((3,), dtype=self.xp.float32)

        print(f"number of antenna(pulses) / frequency: {na} / {nf}")

        for ia in range(na):
            # per-pulse sensor pos (CPU -> GPU small copy)
            tx = self.sensor_xyz[ia]  # (3,) float32 CPU
            tx_buf[...] = self.xp.asarray(tx, dtype=self.xp.float32)

            # out accumulator for this pulse
            out = self.xp.zeros((nf,), dtype=self.xp.complex64)

            if use_farfield:
                # los = (tx - ref) / ||tx-ref||
                v = tx - refp.get()  # CPU 계산(3차원이라 미미)
                n = float(np.linalg.norm(v) + 1e-12)
                los = (v / n).astype(np.float32)
                los_buf[...] = self.xp.asarray(los, dtype=self.xp.float32)

            # targets chunk loop
            for i0 in range(0, Nt, chunk):
                i1 = min(i0 + chunk, Nt)
                t_chunk = targets[i0:i1]  # (Nc,3) float32
                s_chunk = sigma[i0:i1]    # (Nc,)  complex64
                Nc = int(i1 - i0)

                # flatten targets to (3*Nc) contiguous
                t_flat = t_chunk.reshape(-1)

                if use_farfield:
                    k_far(grid, (threads,),
                        (los_buf, refp, t_flat, s_chunk, freqs,
                        np.int32(Nc), np.int32(nf), out))
                else:
                    k_near(grid, (threads,),
                        (tx_buf, refp, t_flat, s_chunk, freqs,
                            np.int32(Nc), np.int32(nf), out))

            signal_freq[ia, :] = out

            if (ia % max(1, na//10)) == 0:
                print(f"[PH fused] {ia+1}/{na}")

        # 기존 반환 형식에 맞춤
        return signal_freq, freqs, refp
    
    def range_compress_batched(self, signal, range_window):
        """
        개선점:
        - pulse loop 안에서 ifft 하지 말고, batched ifft 한 방에
        - range_window도 생성해서 반환
        """

        # batched IFFT along frequency axis=1
        rc = self.xp.fft.fftshift(self.xp.fft.ifft(signal, n=int(self.fft_length), axis=1), axes=1)

        return rc
    
    def postprocess(self, signal):
        if self.window_type == 'Hanning':
            w_az = self.xp.asarray(np.hanning(self.na), dtype=self.xp.float64)
            w_f  = self.xp.asarray(np.hanning(self.nf), dtype=self.xp.float64)
        elif self.window_type == "Hamming":
            w_az = self.xp.asarray(np.hamming(self.na), dtype=self.xp.float64)
            w_f  = self.xp.asarray(np.hamming(self.nf), dtype=self.xp.float64)
        else:
            w_az = self.xp.ones((self.na,), dtype=self.xp.float64)
            w_f  = self.xp.ones((self.nf,), dtype=self.xp.float64)
            
        return signal * (w_az[:, None] * w_f[None, :])
    
    def backprojection_image(self, signal, nx=500, ny=500, z0=0, range_gate=None,
                             # ----auto gate---- 옵션
                             auto_gate=True, # True면 coarse로 range_gate 자동 생성
                             nx_coarse=128, ny_coarse=128,
                             coarse_pulse_stride=4, # 1이면 모든 pulse, 4면 1/4만 사용 (coarse 가속)
                             gate_margin_m=5.0 # gate 여유 (m)
                             ):
        
        # Set up the image space
        xi = self.xp.linspace(-0.5 * self.x_span, 0.5 * self.x_span, nx)
        yi = self.xp.linspace(-0.5 * self.y_span, 0.5 * self.y_span, ny)

        x_image, y_image = self.xp.meshgrid(xi, yi)
        z_image = self.xp.full_like(x_image, float(z0), dtype=self.xp.float64)
        # Initialize the image
        bp_image = self.xp.zeros_like(x_image, dtype=self.xp.complex128)

        # Loop over all pulses in the data
        term = 1j * 4.0 * pi * self.f0 / speed_of_light

        sensor_xyz = self.sensor_xyz
        r0 = self.radius * self.xp.ones(len(sensor_xyz))
        # To work with stripmap
        # if not isinstance(self.radius, list):
        #     self.radius *= self.xp.ones(len(sensor_xyz))
        
        df = float(self.freq_space[1] - self.freq_space[0])
        range_extent = speed_of_light / (2.0 * df)

        # range_window: (-extent/2 .. +extent/2)
        range_window = self.xp.linspace(-0.5 * range_extent, 0.5 * range_extent, int(self.fft_length), dtype=self.xp.float64)
        
        # ---------------------------------------
        # 내부 유틸: "주어진 격자"로 BP 한번 수행
        # (signal은 chunk로 RC해서 pulse loop로 누적)
        # pulse_indices: 사용할 pulse 인덱스 리스트 (subsampling 용)
        # gate: (rmin,rmax) or None
        # ---------------------------------------
        def _bp_one_pass(nx_, ny_, gate, pulse_indices):
            xi = self.xp.linspace(-0.5 * self.x_span, 0.5 * self.x_span, int(nx_), dtype=self.xp.float64)
            yi = self.xp.linspace(-0.5 * self.y_span, 0.5 * self.y_span, int(ny_), dtype=self.xp.float64)
            X, Y = self.xp.meshgrid(xi, yi)
            Z = self.xp.full_like(X, float(z0), dtype=self.xp.float64)

            img = self.xp.zeros_like(X, dtype=self.xp.complex128)

            # pulse_indices가 None이면 전체 사용
            if pulse_indices is None:
                pulse_indices = np.arange(int(self.na), dtype=np.int32)

            pulse_set = set(map(int, pulse_indices.tolist()))  # 빠른 포함 검사(파이썬 set)
            # chunk loop (pulse chunk)
            for i0 in range(0, int(self.na), int(self.chunk_facets)):
                i1 = min(i0 + int(self.chunk_facets), int(self.na))

                # 이 chunk에서 사용할 pulse가 하나도 없으면 RC 자체를 건너뜀(매우 중요)
                has_any = False
                for p in range(i0, i1):
                    if p in pulse_set:
                        has_any = True
                        break
                if not has_any:
                    continue

                s = signal[i0:i1]  # (chunk, nf)
                # 사용자 코드: batched RC (chunk 단위)
                rc = self.range_compress_batched(s, range_window=range_window)  # (chunk, fft_length)

                # per pulse in chunk
                for p in range(i0, i1):
                    if p not in pulse_set:
                        continue

                    xyz = sensor_xyz[p]
                    prof = rc[p - i0]

                    rng = self.xp.sqrt((xyz[0]-X)**2 + (xyz[1]-Y)**2 + (xyz[2]-Z)**2) - float(r0[p])

                    if gate is not None:
                        rmin, rmax = gate
                        mask = (rng >= rmin) & (rng <= rmax)
                    else:
                        mask = None

                    f = interp1d_gpu_uniform(range_window, prof, xp=self.xp, kind='linear', fill_value=0.0)
                    samp = f(rng)
                    if mask is not None:
                        samp = samp * mask.astype(self.xp.float64)

                    img += samp * self.xp.exp(term * rng)

            return img, xi, yi
        
        
        # ---------------------------------------
        # (A) auto_gate: coarse pass → peak → gate 계산
        # ---------------------------------------
        auto_gate_used = False
        if range_gate is None and auto_gate:
            auto_gate_used = True

            # coarse에서 쓸 pulse 인덱스(서브샘플링)
            stride = max(1, int(coarse_pulse_stride))
            pulse_idx = np.arange(0, int(self.na), stride, dtype=np.int32)

            # coarse BP (게이트 없이)
            img_c, xi_c, yi_c = _bp_one_pass(nx_coarse, ny_coarse, gate=None, pulse_indices=pulse_idx)

            # peak 찾기 (GPU면 xp.argmax로 찾고, 인덱스만 CPU로)
            metric = self.xp.abs(img_c)  # |img|
            flat_idx = int(self.xp.argmax(metric).get())
            iy = flat_idx // int(nx_coarse)
            ix = flat_idx %  int(nx_coarse)

            x_est = float(xi_c[ix].get())
            y_est = float(yi_c[iy].get())
            z_est = float(z0)

            # peak 기반 gate 계산(ΔR = |sensor - p_est| - r0)
            sensor_cpu = sensor_xyz.get().astype(np.float64)  # (na,3)
            p_est = np.array([x_est, y_est, z_est], dtype=np.float64)
            dR = np.linalg.norm(sensor_cpu - p_est[None, :], axis=1) - r0.get()

            rmin = float(np.min(dR) - gate_margin_m)
            rmax = float(np.max(dR) + gate_margin_m)

            # range_window 범위로 clamp
            rw_cpu = range_window.get()
            rw_min = float(np.min(rw_cpu))
            rw_max = float(np.max(rw_cpu))
            rmin = max(rmin, rw_min)
            rmax = min(rmax, rw_max)

            # 너무 좁아지는 경우 방지
            if rmax <= rmin:
                # 대략 range resolution 기준 2~3 bin 확보
                dR_res = speed_of_light / (2.0 * float(self.bandwidth))
                mid = 0.5 * (rmin + rmax)
                rmin = mid - 3.0 * dR_res
                rmax = mid + 3.0 * dR_res

            range_gate = (rmin, rmax)
            print(f"[AUTO-GATE] coarse peak (x,y,z)=({x_est:.3f},{y_est:.3f},{z_est:.3f})")
            print(f"[AUTO-GATE] range_gate = ({range_gate[0]:.3f}, {range_gate[1]:.3f}) [m]")

        # ---------------------------------------
        # (B) refine/full BP (gate 적용)
        # ---------------------------------------
        bp_image, xi, yi = _bp_one_pass(nx, ny, gate=range_gate, pulse_indices=None)
        
        if self.use_cupy:
            # return bp_image.get(), xi.get(), yi.get()
            return self.xp.asnumpy(bp_image), self.xp.asnumpy(xi), self.xp.asnumpy(yi)

        return bp_image, xi, yi
    
    
    def backprojection_image_fused(self, signal, nx=500, ny=500, z0=0, range_gate=None,
                                   # ----dtype----
                                   use_complex64=True,
                                   # ----auto gate---- 옵션
                                   auto_gate=True, # True면 coarse로 range_gate 자동 생성
                                   nx_coarse=128, ny_coarse=128,
                                   coarse_pulse_stride=4, # 1이면 모든 pulse, 4면 1/4만 사용 (coarse 가속)
                                   gate_margin_m=5.0, # gate 여유 (m)
                                   use_power_peak=True
                                   ):
        
        sensor_xyz = self.sensor_xyz
        ref_point = self.ref_point
        ref_point[2] = 0
        # r0 = self.radius * self.xp.ones(len(sensor_xyz))
        r0_vec = np.linalg.norm(sensor_xyz - ref_point[None, :], axis=1)
        
        # To work with stripmap
        # if not isinstance(self.radius, list):
        #     self.radius *= self.xp.ones(len(sensor_xyz))
        
        df = float(self.freq_space[1] - self.freq_space[0])
        range_extent = speed_of_light / (2.0 * df)

        # range_window: (-extent/2 .. +extent/2)
        range_window = self.xp.linspace(-0.5 * range_extent, 0.5 * range_extent, int(self.fft_length), dtype=self.xp.float64)
        x0 = float(range_window[0].get())
        dx = float((range_window[1]-range_window[0]).get())
        M = int(range_window.shape[0])
                
        # Loop over all pulses in the data
        term = 1j * 4.0 * pi * self.f0 / speed_of_light
        term_re = 0.0
        term_im = float(4.0 * pi * float(self.f0 / speed_of_light))
        
        # dtype
        coord_dtype = self.xp.float32 if use_complex64 else self.xp.float64
        c_dtype = self.xp.complex64 if use_complex64 else self.xp.complex128

        # RawKernel 준비
        bp_k = _get_bp_rawkernel(self.xp, use_complex64=use_complex64)

        # 내부: grid 만들기 (한 번만)
        def _make_grid(nx_, ny_):
            xi = self.xp.linspace(-0.5 * self.x_span, 0.5 * self.x_span, int(nx_), dtype=coord_dtype)
            yi = self.xp.linspace(-0.5 * self.y_span, 0.5 * self.y_span, int(ny_), dtype=coord_dtype)
            X2, Y2 = self.xp.meshgrid(xi, yi)
            Z2 = self.xp.full_like(X2, float(z0), dtype=coord_dtype)
            return xi, yi, X2.ravel(), Y2.ravel(), Z2.ravel()

        # 내부: chunked RC + fused BP 1-pass
        def _bp_pass(nx_, ny_, gate, pulse_stride):
            xi, yi, Xf, Yf, Zf = _make_grid(nx_, ny_)
            N = int(Xf.size)
            img = self.xp.zeros((N,), dtype=c_dtype)

            if gate is None:
                rmin, rmax = (-1e30, 1e30)
            else:
                rmin, rmax = map(float, gate)

            threads = 256
            blocks = (N + threads - 1) // threads

            stride = max(1, int(pulse_stride))

            # pulse chunk loop
            for i0 in range(0, self.na, int(self.chunk_facets)):
                i1 = min(i0 + int(self.chunk_facets), self.na)

                # 이 chunk에서 실제로 사용할 pulse가 있는지 확인 (stride에 의해 skip 가능)
                # coarse 가속에서 매우 중요
                has_any = False
                for p in range(i0, i1):
                    if (p % stride) == 0:
                        has_any = True
                        break
                if not has_any:
                    continue

                s = signal[i0:i1]  # (chunk, nf)

                # ✅ 당신이 이미 쓰는 방식: chunk 단위 range compression
                # 반환 shape: (chunk, M)
                rc = self.range_compress_batched(s, range_window=range_window)

                # dtype 정리
                if rc.dtype != c_dtype:
                    rc = rc.astype(c_dtype, copy=False)

                # pulse loop inside chunk
                for p in range(i0, i1):
                    if (p % stride) != 0:
                        continue

                    prof = rc[p - i0]  # (M,)
                    sx, sy, sz = sensor_xyz[p].tolist()
                    r0 = r0_vec[p]
                    
                    # print(type(prof), type(Xf), type(Yf), type(Zf), type(img))
                    # print(prof.shape, Xf.shape, Yf.shape, Zf.shape, img.shape)

                    if use_complex64:
                        bp_k((blocks,), (threads,),
                            (prof, Xf, Yf, Zf, img,
                            np.int32(N),
                            np.float32(sx), np.float32(sy), np.float32(sz),
                            np.float32(r0),
                            np.float32(x0), np.float32(dx), np.int32(M),
                            np.float32(term_im),
                            np.float32(rmin), np.float32(rmax)))
                    else:
                        bp_k((blocks,), (threads,),
                            (prof, Xf, Yf, Zf, img,
                            np.int32(N),
                            np.float64(sx), np.float64(sy), np.float64(sz),
                            np.float64(r0),
                            np.float64(x0), np.float64(dx), np.int32(M),
                            np.float64(term_im),
                            np.float64(rmin), np.float64(rmax)))

                print(f"[BP fused chunk] pulses {i0}-{i1-1} done")

            return img.reshape((int(ny_), int(nx_))), xi, yi

        # -------------------------
        # (A) auto_gate: coarse pass -> peak -> gate 생성
        # -------------------------
        auto_gate_used = False
        if (range_gate is None) and auto_gate:
            auto_gate_used = True

            img_c, xi_c, yi_c = _bp_pass(nx_coarse, ny_coarse, gate=None, pulse_stride=coarse_pulse_stride)

            metric = self.xp.abs(img_c)
            if use_power_peak:
                metric = metric * metric

            flat = int(self.xp.asnumpy(self.xp.argmax(metric)))
            iy = flat // int(nx_coarse)
            ix = flat % int(nx_coarse)

            x_est = float(self.xp.asnumpy(xi_c[ix]))
            y_est = float(self.xp.asnumpy(yi_c[iy]))
            z_est = float(z0)

            # pulse 전체 ΔR min/max (CPU에서 가볍게)
            p_est = np.array([x_est, y_est, z_est], dtype=np.float64)
            dR = np.linalg.norm(sensor_xyz - p_est[None, :], axis=1) - r0_vec

            rmin = float(np.min(dR) - gate_margin_m)
            rmax = float(np.max(dR) + gate_margin_m)

            # range_window 범위로 clamp
            rw0 = x0
            rw1 = x0 + dx * (M - 1)
            rw_min, rw_max = (min(rw0, rw1), max(rw0, rw1))
            rmin = max(rmin, rw_min)
            rmax = min(rmax, rw_max)

            if rmax <= rmin:
                mid = 0.5 * (rmin + rmax)
                rmin = mid - 3.0 * abs(dx)
                rmax = mid + 3.0 * abs(dx)

            range_gate = (rmin, rmax)
            print(f"[AUTO-GATE] coarse peak=({x_est:.2f},{y_est:.2f},{z_est:.2f})  gate=({rmin:.2f},{rmax:.2f}) [m]")

        # -------------------------
        # (B) refine pass: full BP with gate
        # -------------------------
        img, xi, yi = _bp_pass(nx, ny, gate=range_gate, pulse_stride=1)

        # 반환
        if self.use_cupy:
            return self.xp.asnumpy(img), self.xp.asnumpy(xi), self.xp.asnumpy(yi), range_gate, auto_gate_used
        return img, xi, yi, range_gate, auto_gate_used
                