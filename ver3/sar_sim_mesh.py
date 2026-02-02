import numpy as np
from scipy.constants import speed_of_light, pi
from gpu_functions import interp1d_gpu_uniform
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
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        else:
            self.xp = np
            
        self.sensor_xyz = None
            
        self.set_ref_point()
        self.set_freq_space()
        self.set_az_space()
        self.set_fft_length()
            
    def set_ref_point(self):
        """
        타겟점들의 무게중심 (단순 평균 or RCS 가중 무게중심)
        만약 spotlight SAR이면 고정된 ref_point가 아닌 다음과 같이 pulse-dependent ref_point
            ref_point[p] = spotlight_aim_point[p]

        """
        if self.rcs is None:
            self.ref_point = self.xp.mean(self.targets, axis=0)
        else:
            self.ref_point = self.xp.sum(self.targets * self.rcs[:, None], axis=0) / self.xp.sum(self.rcs)
        
    def set_freq_space(self):
        """
        range방향 sampling nyquist 조건 충족
            df <= c / 2r
        """
        r = self.xp.sqrt(self.x_span **2 + self.y_span **2)
        df = speed_of_light / (2.0 * r)
        nf = self.xp.floor(self.bandwidth / df)
        
        self.freq_space = self.xp.linspace(self.f0, self.f0 + self.bandwidth, int(nf))
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
        r = self.xp.sqrt(self.x_span **2 + self.y_span **2)
        da = speed_of_light / (2.0 * r * self.f0)
        na = self.xp.round((self.az_end - self.az_start) / da)
        
        self.az_space = self.xp.linspace(self.az_start, self.az_end, int(na))
        self.na = int(na)
    
    def set_fft_length(self):        
        self.fft_length = int(8 * 2**self.xp.ceil(self.xp.log2(self.nf)))
    
    def get_sensor_positions(self):
        
        if self.az_space is None:
            self.set_az_space
            
        center = self.ref_point
            
        sensor_x = center[0] + self.radius * self.xp.cos(self.xp.radians(self.az_space))
        sensor_y = center[1] + self.radius * self.xp.sin(self.xp.radians(self.az_space))
        sensor_z = center[2] + self.radius * self.xp.zeros_like(self.az_space)
        sensor_xyz = self.xp.stack((sensor_x, sensor_y, sensor_z), axis=1, dtype=self.xp.float64)
        
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
    
    def backprojection_image(self, signal, nx=500, ny=500, z0=0, range_gate=None):
        
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
        
        for i0 in range(0, self.na, self.chunk_facets): # signal.shape[0] == self.na
            i1 = min(i0 + self.chunk_facets, self.na)
            
            s = signal[i0:i1]
            rc = self.range_compress_batched(s, range_window=range_window)
            
            for p in range(i0, i1):# per pulse            
                
                xyz = sensor_xyz[p]
                prof = rc[p-i0]
                # range_image = |sensor - pixel| - r0
                rng = self.xp.sqrt((xyz[0]-x_image)**2 + (xyz[1]-y_image)**2 + (xyz[2]-z_image)**2) - r0[p]

                if range_gate is not None:
                    rmin, rmax = range_gate
                    mask = (rng >= rmin) & (rng <= rmax)
                else:
                    mask = None
                # Create the interpolation for this pulse
                # f = interp1d(range_window, range_profile, kind='linear', bounds_error=False, fill_value=0.0)
                f = interp1d_gpu_uniform(range_window, prof, xp=self.xp, kind='linear', fill_value=0.0)
                samp = f(rng)

                if mask is not None:
                    samp = samp * mask.astype(self.xp.float64)

                bp_image += samp * self.xp.exp(term * rng)
            
            print(f"[BP] {i1+1}/{self.na}")
        
        if self.use_cupy:
            # return bp_image.get(), xi.get(), yi.get()
            return self.xp.asnumpy(bp_image), self.xp.asnumpy(xi), self.xp.asnumpy(yi)

        return bp_image, xi, yi