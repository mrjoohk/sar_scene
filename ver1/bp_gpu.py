import numpy as np
from geom import C0  # 또는 from sar_scene.geom import C0
from numpy import sqrt, linspace, zeros_like, exp, sin, cos, ones
from scipy.interpolate import interp1d

class BackprojectionGPU:
    """
    Reference-style BP:
      - Input: range-compressed profiles (rc) and range_window
      - For each pulse:
          bp += interp(range_profile_p, range_image) * exp(j*4π*f_start/c * range_image)
      - range_image = R_oneway(pixel) - Rref[p]
    """

    def __init__(self, use_cupy=True, chunk_pixels=250000):
        self.use_cupy = use_cupy
        self.chunk_pixels = chunk_pixels

    @staticmethod
    def _interp_linear_uniform(xp, fp, xq):
        """
        Uniform-grid linear interpolation:
          xp: (M,) uniformly spaced increasing
          fp: (M,) complex
          xq: (N,) query
        returns: (N,) complex, out-of-range -> 0

        NOTE: GPU/CPU 공용으로 쓰기 위해, xp spacing을 이용해 index 계산.
        """
        # xp must be uniform
        dx = xp[1] - xp[0]
        x0 = xp[0]
        M = xp.shape[0]

        # idx_float = (xq - x0) / dx
        idxf = (xq - x0) / dx
        i0 = np.floor(idxf).astype(np.int64)
        t = (idxf - i0).astype(np.float64)

        # out-of-range mask
        valid = (i0 >= 0) & (i0 < (M - 1))

        out = np.zeros_like(xq, dtype=np.complex128)
        if np.any(valid):
            i0v = i0[valid]
            tv = t[valid]
            out[valid] = (1.0 - tv) * fp[i0v] + tv * fp[i0v + 1]
        return out

    def bp_2d_rangeprofile_streaming(
        self,
        rc,                 # (n_pulses, fft_length) complex  (fftshift된 range profile)
        range_window,       # (fft_length,) meters  [-extent/2 .. +extent/2]
        tx_traj, rx_traj,   # (n_pulses,3)
        Rref,               # (n_pulses,) one-way reference range (range_center). meters
        x_grid, y_grid,     # 1D grids
        z0=0.0,             # imaging plane z
        f_start=None,       # Hz (carrier term uses start frequency)
        bistatic=True,      # True면 one-way = 0.5*(Rtx+Rrx), False면 one-way=Rtx
    ):
        """
        참고 코드 방식 2D BP.

        rc는 반드시 sar_sim.range_compress()의 출력(fftshift 적용된 range profile)을 넣어주세요.
        range_window도 sar_sim.range_compress()에서 나온 값을 그대로 넣어야 정합됩니다.
        """
        rc = np.asarray(rc, dtype=np.complex128)
        range_window = np.asarray(range_window, dtype=np.float64)
        tx_traj = np.asarray(tx_traj, dtype=np.float64)
        rx_traj = np.asarray(rx_traj, dtype=np.float64)
        Rref = np.asarray(Rref, dtype=np.float64)

        n_pulses, fft_len = rc.shape

        if f_start is None:
            raise ValueError("f_start (start_frequency) is required for carrier phase term.")

        # term = 1j * 4π * f_start / c
        term = 1j * 4.0 * np.pi * float(f_start) / C0

        # pixel list
        X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
        P = np.stack([X.ravel(), Y.ravel(), z0 * np.ones_like(X).ravel()], axis=1)
        n_pix = P.shape[0]

        use_cupy = self.use_cupy
        if use_cupy:
            try:
                import cupy as cp
            except Exception:
                use_cupy = False
        
        index = 0
        for xs, ys, zs in tx_traj:

            # Create the interpolation for this pulse
            f = interp1d(range_window, rc, kind='linear', bounds_error=False, fill_value=0.0)

            # Calculate the range to each pixel
            range_image = sqrt((xs - X) ** 2 + (ys - Y) ** 2 + (zs - z0 * np.ones_like(X)) ** 2) - Rref[index]

            # Interpolate the range profile onto the image grid and multiply by the range phase
            # For large scenes, should check the range window and index
            bp_image += f(range_image) * exp(term * range_image)

            index += 1

        return bp_image
            


        # if use_cupy:
        #     import cupy as cp

        #     # GPU constants
        #     xp_gpu = cp.asarray(range_window, dtype=cp.float64)
        #     dx = xp_gpu[1] - xp_gpu[0]
        #     x0 = xp_gpu[0]
        #     M = xp_gpu.shape[0]
        #     term_gpu = cp.asarray(term, dtype=cp.complex128)

        #     img = cp.zeros((n_pix,), dtype=cp.complex128)

        #     # CPU->GPU로 P는 chunk로 올림 (메모리 절약)
        #     for p in range(n_pulses):
        #         prof = cp.asarray(rc[p], dtype=cp.complex128)
        #         tx = cp.asarray(tx_traj[p], dtype=cp.float64)
        #         rx = cp.asarray(rx_traj[p], dtype=cp.float64)
        #         Rref_p = cp.asarray(Rref[p], dtype=cp.float64)

        #         for i0 in range(0, n_pix, self.chunk_pixels):
        #             i1 = min(i0 + self.chunk_pixels, n_pix)
        #             Pc = cp.asarray(P[i0:i1], dtype=cp.float64)

        #             # one-way range
        #             if bistatic:
        #                 Rtx = cp.linalg.norm(Pc - tx[None, :], axis=1)
        #                 Rrx = cp.linalg.norm(Pc - rx[None, :], axis=1)
        #                 R_one = 0.5 * (Rtx + Rrx)
        #             else:
        #                 R_one = cp.linalg.norm(Pc - tx[None, :], axis=1)

        #             rng = R_one - Rref_p  # range_image in reference code

        #             # --- uniform linear interp on GPU ---
        #             idxf = (rng - x0) / dx
        #             i_base = cp.floor(idxf).astype(cp.int64)
        #             t = (idxf - i_base).astype(cp.float64)

        #             valid = (i_base >= 0) & (i_base < (M - 1))

        #             samp = cp.zeros((i1 - i0,), dtype=cp.complex128)
        #             if cp.any(valid):
        #                 ib = i_base[valid]
        #                 tv = t[valid]
        #                 samp[valid] = (1.0 - tv) * prof[ib] + tv * prof[ib + 1]

        #             # carrier phase term
        #             img[i0:i1] += samp * cp.exp(term_gpu * rng.astype(cp.complex128))

        #         if (p % max(1, n_pulses // 10)) == 0:
        #             print(f"[BP-ref] pulse {p+1}/{n_pulses}")

        #     img = cp.asnumpy(img).reshape((len(y_grid), len(x_grid)))
        #     return img

        # else:
        #     # CPU fallback
        #     img = np.zeros((n_pix,), dtype=np.complex128)
        #     xp = range_window

        #     for p in range(n_pulses):
        #         prof = rc[p]
        #         tx = tx_traj[p]
        #         rx = rx_traj[p]
        #         Rref_p = Rref[p]

        #         for i0 in range(0, n_pix, self.chunk_pixels):
        #             i1 = min(i0 + self.chunk_pixels, n_pix)
        #             Pc = P[i0:i1]

        #             if bistatic:
        #                 Rtx = np.linalg.norm(Pc - tx[None, :], axis=1)
        #                 Rrx = np.linalg.norm(Pc - rx[None, :], axis=1)
        #                 R_one = 0.5 * (Rtx + Rrx)
        #             else:
        #                 R_one = np.linalg.norm(Pc - tx[None, :], axis=1)

        #             rng = R_one - Rref_p

        #             samp = self._interp_linear_uniform(xp, prof, rng)
        #             img[i0:i1] += samp * np.exp(term * rng)

        #         if (p % max(1, n_pulses // 10)) == 0:
        #             print(f"[BP-ref] pulse {p+1}/{n_pulses}")

        #     return img.reshape((len(y_grid), len(x_grid)))
