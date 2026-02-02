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
        self.xp = None
        if use_cupy:
            import cupy as cp
            self.xp = cp
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
        self.nf = nf
        
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
        na = round((self.az_end - self.az_start) / da)
        
        self.az_space = np.linspace(self.az_start, self.az_end, int(na))
        self.na = na
    
    def set_fft_length(self):        
        self.fft_length = int(8 * 2**np.ceil(np.log2(self.nf)))
    
    def get_sensor_positions(self):
        
        if self.az_space is None:
            self.set_az_space
            
        center = self.ref_point
            
        sensor_x = center[0] + self.radius * np.cos(np.radians(self.az_space))
        sensor_y = center[1] + self.radius * np.sin(np.radians(self.az_space))
        sensor_z = center[2] + self.radius * np.zeros_like(self.az_space)
        sensor_xyz = np.stack((sensor_x, sensor_y, sensor_z), axis=1)
        
        print("sensor xyz shape: ", sensor_xyz.shape)
        return sensor_xyz
    
    def generate_phase_history(self, use_farfield=False):
        signal_freq = np.zeros((int(self.na), int(self.nf)), dtype=complex)
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
            tx = self.xp.asarray(traj[ia], dtype=self.xp.float64) # tx.shape: (3, )
            rx = self.xp.asarray(traj[ia], dtype=self.xp.float64)
            
            if use_farfield:
                los = tx - ref_point_gpu
                los = los / (self.xp.linalg.norm(los) + 1e-12) # los.shape: (3,)
                # one-way delta range ≈ dot(los, (xi-ref))
                d = targets_gpu - ref_point_gpu[None, :] # d.shape: (N, 3) (N: the number of gpu chunk)
                dR = self.xp.sum(d * los[None, :], axis=1)  # (N,)
                # range_center는 이미 Rref로 처리하므로, 여기선 dR만 사용
                phase = self.xp.exp(kc * (freq_gpu[None, :] * (dR[:, None])))
                # print(phase)
            else:
                Rtx = self.xp.linalg.norm(targets_gpu - tx[None, :], axis=1)
                Rrx = self.xp.linalg.norm(targets_gpu - rx[None, :], axis=1)
                R_oneway = 0.5 * (Rtx + Rrx)

                Rref_tx = self.xp.linalg.norm(ref_point_gpu - tx)              # scalar
                Rref_rx = self.xp.linalg.norm(ref_point_gpu - rx)              # scalar
                Rref = 0.5 * (Rref_tx + Rref_rx)                                # scalar

                dR = R_oneway - Rref                                            # (N,)
                phase = self.xp.exp(kc * (freq_gpu[None, :] * (dR[:, None])))
            
            signal_pf = self.xp.sum(amp_gpu[:, None] * phase, axis=0)
            
            if self.use_cupy:
                signal_freq[ia, :] = self.xp.asnumpy(signal_pf)
            else:
                signal_freq[ia, :] = self.xp.array(signal_pf)
            
        return signal_freq, freq_gpu, ref_point_gpu
    
    def postprocess(self, signal):
        if self.window_type == 'Hanning':    
            coefficients = np.outer(np.hanning(self.na), np.hanning(self.nf))
        elif self.window_type == 'Hamming':            
            coefficients = np.outer(np.hamming(self.na), np.hamming(self.nf))
        else:            
            coefficients = np.ones_like(signal)
            
        return signal * coefficients
    
    def backprojection_image(self, signal, nx=500, ny=500):
        frequency_step = self.freq_space[1] - self.freq_space[0]

        # Calculate the maximum scene size and resolution
        range_extent = speed_of_light / (2.0 * frequency_step)
        # Calculate the range window for the pulses
        range_window = self.xp.linspace(-0.5 * range_extent, 0.5 * range_extent, self.fft_length)
        
        # Set up the image space
        xi = self.xp.linspace(-0.5 * self.x_span, 0.5 * self.x_span, nx)
        yi = self.xp.linspace(-0.5 * self.y_span, 0.5 * self.y_span, ny)

        [x_image, y_image] = self.xp.meshgrid(xi, yi)
        z_image = self.xp.zeros_like(x_image)
        # Initialize the image
        bp_image = self.xp.zeros_like(x_image, dtype=complex)

        # Loop over all pulses in the data
        term = 1j * 4.0 * pi * self.f0 / speed_of_light

        sensor_xyz = self.sensor_xyz
        r0 = self.radius * self.xp.ones(len(sensor_xyz))
        # To work with stripmap
        # if not isinstance(self.radius, list):
        #     self.radius *= self.xp.ones(len(sensor_xyz))        

        signal_gpu = self.xp.asarray(signal, dtype=self.xp.complex128)
        index = 0
        for xyz in sensor_xyz:

            # Calculate the range profile
            range_profile = self.xp.fft.fftshift(self.xp.fft.ifft(signal_gpu[index, :], self.fft_length))
            # Create the interpolation for this pulse
            # f = interp1d(range_window, range_profile, kind='linear', bounds_error=False, fill_value=0.0)
            f = interp1d_gpu_uniform(range_window, range_profile, xp=self.xp, kind='linear', fill_value=0.0)
            # Calculate the range to each pixel
            range_image = self.xp.sqrt((xyz[0] - x_image) ** 2 + (xyz[1] - y_image) ** 2 + (xyz[2] - z_image) ** 2) - r0[index]

            # Interpolate the range profile onto the image grid and multiply by the range phase
            # For large scenes, should check the range window and index
            bp_image += f(range_image) * self.xp.exp(term * range_image)

            index += 1
            if (index %10 == 0):
                print(f"{str(index)}-th BP pulses")

        print(f"{str(index)}-th BP pulses")
        
        if self.use_cupy:
            # return bp_image.get(), xi.get(), yi.get()
            return self.xp.asnumpy(bp_image), self.xp.asnumpy(xi), self.xp.asnumpy(yi)

        return bp_image, xi, yi