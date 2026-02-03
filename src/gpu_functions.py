import numpy as np
from scipy.constants import speed_of_light, pi

class UniformGridInterp1D:
    """
    GPU-friendly 1D interpolation on a uniform grid (linspace-like).

    Supported kind:
      - 'nearest'
      - 'previous', 'next'
      - 'linear', 'slinear'
      - 'quadratic'  (3-point Lagrange, centered on segment)
      - 'cubic'      (Catmull-Rom cubic convolution, 4-point stencil)

    Notes:
      * 'cubic' here is NOT SciPy's not-a-knot cubic spline. It's a local cubic (Catmull-Rom).
      * Edge handling: for quadratic/cubic, if stencil neighbors are missing near boundaries,
        it falls back to 'linear' (inside range), and still uses fill_value outside range.
    """
    def __init__(self, x_grid, y_grid, xp, kind="linear", fill_value=0.0):
        self.xp = xp
        self.kind = kind
        self.y = y_grid
        self.n = int(y_grid.size)
        self.fill_value = fill_value

        # uniform spacing (assumes linspace-like)
        self.x0 = x_grid[0]
        self.xN = x_grid[-1]
        self.dx = (x_grid[-1] - x_grid[0]) / (x_grid.size - 1)
        self.inv_dx = 1.0 / self.dx

        # normalize aliases
        if self.kind == "slinear":
            self.kind = "linear"

    def __call__(self, xq):
        xp = self.xp

        # Map query x -> fractional index u
        u = (xq - self.x0) * self.inv_dx  # float
        i0 = xp.floor(u).astype(xp.int64)
        t = u - i0.astype(u.dtype)

        # Outside x range -> fill_value
        in_range = (xq >= self.x0) & (xq <= self.xN)

        kind = self.kind
        if kind in ("linear",):
            yq = self._linear(i0, t)
        elif kind == "nearest":
            yq = self._nearest(i0, t)
        elif kind == "previous":
            yq = self._previous(i0, t)
        elif kind == "next":
            yq = self._next(i0, t)
        elif kind == "quadratic":
            yq = self._quadratic(i0, t, in_range)
        elif kind == "cubic":
            yq = self._cubic_catmull_rom(i0, t, in_range)
        else:
            raise ValueError(f"Unsupported kind: {kind}")

        # apply fill outside range
        yq = self._apply_fill(yq, in_range)
        return yq

    # ---------- basic helpers ----------
    def _apply_fill(self, yq, in_range):
        xp = self.xp
        if xp is np:
            return np.where(in_range, yq, self.fill_value).astype(self.y.dtype, copy=False)
        else:
            fv = xp.asarray(self.fill_value, dtype=self.y.dtype)
            return xp.where(in_range, yq, fv)

    def _gather(self, idx):
        # idx assumed already clipped to valid [0, n-1]
        return self.y[idx]

    # ---------- kinds ----------
    def _linear(self, i0, t):
        xp = self.xp
        # valid segment indices: i0 in [0, n-2]
        valid = (i0 >= 0) & (i0 < (self.n - 1))

        i0c = xp.clip(i0, 0, self.n - 2)
        i1c = i0c + 1

        y0 = self._gather(i0c)
        y1 = self._gather(i1c)
        yq = y0 + (y1 - y0) * t

        # outside segment but maybe still "in_range" is handled later;
        # for safety, zero invalid segments here
        if xp is np:
            yq = np.where(valid, yq, 0).astype(self.y.dtype, copy=False)
        else:
            yq = xp.where(valid, yq, xp.asarray(0, dtype=self.y.dtype))
        return yq

    def _nearest(self, i0, t):
        xp = self.xp
        # nearest index is round(u) == floor(u + 0.5)
        idx = i0 + (t >= 0.5)
        idx = idx.astype(xp.int64)
        idxc = xp.clip(idx, 0, self.n - 1)
        return self._gather(idxc)

    def _previous(self, i0, t):
        xp = self.xp
        # left-hold: y[i0], with i0 clipped
        idxc = xp.clip(i0, 0, self.n - 1)
        return self._gather(idxc)

    def _next(self, i0, t):
        xp = self.xp
        # right-hold: y[i0+1] (except at end)
        idx = i0 + 1
        idxc = xp.clip(idx, 0, self.n - 1)
        return self._gather(idxc)

    def _quadratic(self, i0, t, in_range):
        """
        3-point quadratic Lagrange using points (i0-1, i0, i0+1)
        for segment between i0 and i0+1.
        Needs i0 in [1, n-2]. Near edges fallback to linear (but still in_range).
        """
        xp = self.xp
        # stencil-valid within range
        stencil_ok = (i0 >= 1) & (i0 <= (self.n - 2))

        # fallback linear for missing neighbors (still within x-range)
        y_lin = self._linear(i0, t)

        i_m1 = xp.clip(i0 - 1, 0, self.n - 1)
        i_0  = xp.clip(i0,     0, self.n - 1)
        i_p1 = xp.clip(i0 + 1, 0, self.n - 1)

        ym1 = self._gather(i_m1)
        y0  = self._gather(i_0)
        yp1 = self._gather(i_p1)

        # Lagrange basis for x = 0..2, evaluating at x = 1 + t
        x = 1.0 + t
        L0 = (x - 1.0) * (x - 2.0) / ((0.0 - 1.0) * (0.0 - 2.0))  # (x-1)(x-2)/2
        L1 = (x - 0.0) * (x - 2.0) / ((1.0 - 0.0) * (1.0 - 2.0))  # -x(x-2)
        L2 = (x - 0.0) * (x - 1.0) / ((2.0 - 0.0) * (2.0 - 1.0))  # x(x-1)/2
        y_quad = ym1 * L0 + y0 * L1 + yp1 * L2

        # Use quadratic only where stencil_ok AND in_range; else linear (still inside range)
        use_quad = stencil_ok & in_range
        if xp is np:
            return np.where(use_quad, y_quad, y_lin).astype(self.y.dtype, copy=False)
        else:
            return xp.where(use_quad, y_quad, y_lin)

    def _cubic_catmull_rom(self, i0, t, in_range):
        """
        4-point Catmull-Rom cubic convolution on stencil (i0-1, i0, i0+1, i0+2)
        Needs i0 in [1, n-3]. Near edges fallback to linear (but still in_range).
        """
        xp = self.xp
        stencil_ok = (i0 >= 1) & (i0 <= (self.n - 3))
        y_lin = self._linear(i0, t)

        i_m1 = xp.clip(i0 - 1, 0, self.n - 1)
        i_0  = xp.clip(i0,     0, self.n - 1)
        i_p1 = xp.clip(i0 + 1, 0, self.n - 1)
        i_p2 = xp.clip(i0 + 2, 0, self.n - 1)

        p0 = self._gather(i_m1)
        p1 = self._gather(i_0)
        p2 = self._gather(i_p1)
        p3 = self._gather(i_p2)

        # Catmull-Rom spline (a = -0.5) in Hermite-like form:
        # y = 0.5 * (2p1 + (-p0+p2)t + (2p0-5p1+4p2-p3)t^2 + (-p0+3p1-3p2+p3)t^3)
        t2 = t * t
        t3 = t2 * t
        y_cub = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2
            + (-p0 + 3.0*p1 - 3.0*p2 + p3) * t3
        )

        use_cub = stencil_ok & in_range
        if xp is np:
            return np.where(use_cub, y_cub, y_lin).astype(self.y.dtype, copy=False)
        else:
            return xp.where(use_cub, y_cub, y_lin)


def interp1d_gpu_uniform(x_grid, y_grid, xp, kind="linear", fill_value=0.0):
    """
    Factory: returns callable f(xq) like interp1d (bounds_error=False, fill_value=fill_value)
    Works on GPU if xp=cupy and inputs are cupy arrays.
    """
    return UniformGridInterp1D(x_grid, y_grid, xp=xp, kind=kind, fill_value=fill_value)

def _get_bp_rawkernel(xp, use_complex64: bool):
    """
    RawKernel을 캐시해서 재사용.
    use_complex64=True  -> complex<float> 커널
    use_complex64=False -> complex<double> 커널
    """
    if not hasattr(_get_bp_rawkernel, "_cache"):
        _get_bp_rawkernel._cache = {}

    key = ("c64" if use_complex64 else "c128")
    if key in _get_bp_rawkernel._cache:
        return _get_bp_rawkernel._cache[key]

    if use_complex64:
        code = r'''
        #include <cuComplex.h>
        extern "C" __global__
        void bp_fused_c64(
            const cuFloatComplex* __restrict__ prof,  // (M,)
            const float* __restrict__ X,              // (N,)
            const float* __restrict__ Y,
            const float* __restrict__ Z,
            cuFloatComplex* __restrict__ img,         // (N,)
            const int N,
            const float sx, const float sy, const float sz,
            const float r0,
            const float x0, const float dx, const int M,
            const float term_im,
            const float rmin, const float rmax
        ){
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if(idx >= N) return;

            float x = X[idx];
            float y = Y[idx];
            float z = Z[idx];

            float rx = x - sx;
            float ry = y - sy;
            float rz = z - sz;
            float R = sqrtf(rx*rx + ry*ry + rz*rz) - r0;

            if(R < rmin || R > rmax) return;

            float t = (R - x0) / dx;
            int i0 = (int)floorf(t);
            float a = t - (float)i0;

            if(i0 < 0 || i0 >= (M-1)) return;

            cuFloatComplex s0 = prof[i0];
            cuFloatComplex s1 = prof[i0+1];
            // samp = (1-a)*s0 + a*s1
            cuFloatComplex samp;
            samp.x = (1.0f - a)*s0.x + a*s1.x;
            samp.y = (1.0f - a)*s0.y + a*s1.y;

            float ph = term_im * R;
            float c = cosf(ph);
            float s = sinf(ph);

            // samp * (c + j s)
            cuFloatComplex out;
            out.x = samp.x*c - samp.y*s;
            out.y = samp.x*s + samp.y*c;

            // img += out
            cuFloatComplex cur = img[idx];
            cur.x += out.x;
            cur.y += out.y;
            img[idx] = cur;
        }
        '''
        mod = xp.RawModule(code=code, options=('-std=c++11',), name_expressions=['bp_fused_c64'])
        fn = mod.get_function('bp_fused_c64')
    else:
        code = r'''
        extern "C" __global__
        void bp_fused_c128(
            const complex<double>* __restrict__ prof, // (M,)
            const double* __restrict__ X,             // (N,)
            const double* __restrict__ Y,
            const double* __restrict__ Z,
            complex<double>* __restrict__ img,        // (N,)
            const int N,
            const double sx, const double sy, const double sz,
            const double r0,
            const double x0, const double dx, const int M,
            const double term_im,
            const double rmin, const double rmax
        ){
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if(idx >= N) return;

            double x = X[idx];
            double y = Y[idx];
            double z = Z[idx];

            double rx = x - sx;
            double ry = y - sy;
            double rz = z - sz;
            double R = sqrt(rx*rx + ry*ry + rz*rz) - r0;

            if(R < rmin || R > rmax) return;

            double t = (R - x0) / dx;
            long i0 = (long)floor(t);
            double a = t - (double)i0;

            if(i0 < 0 || i0 >= (M-1)) return;

            complex<double> s0 = prof[i0];
            complex<double> s1 = prof[i0+1];
            complex<double> samp = (1.0 - a)*s0 + a*s1;

            double ph = term_im * R;
            double c = cos(ph);
            double s = sin(ph);
            complex<double> ej = complex<double>(c, s);

            img[idx] += samp * ej;
        }
        '''
        mod = xp.RawModule(code=code, options=('-std=c++11',), name_expressions=['bp_fused_c128'])
        fn = mod.get_function('bp_fused_c128')

    _get_bp_rawkernel._cache[key] = fn
    return fn

def _get_ph_rawkernel_nearfield_c64(xp):
    if hasattr(_get_ph_rawkernel_nearfield_c64, "_k"):
        return _get_ph_rawkernel_nearfield_c64._k

    code = r'''
    #include <cuComplex.h>

    extern "C" __global__
    void ph_accum_nearfield_c64(
        const float* __restrict__ tx,              // (3,) [sx,sy,sz]
        const float* __restrict__ refp,            // (3,)
        const float* __restrict__ targets,         // (3*N) xyzxyz...
        const cuFloatComplex* __restrict__ sigma,  // (N,)
        const float* __restrict__ freqs,           // (nf,) frequency in Hz, baseband +/-B/2
        const int N,
        const int nf,
        cuFloatComplex* __restrict__ out           // (nf,) accumulate
    ){
        // blockIdx.x: frequency index
        int f = (int)blockIdx.x;
        if(f >= nf) return;

        float sx = tx[0], sy = tx[1], sz = tx[2];
        float rxp = refp[0], ryp = refp[1], rzp = refp[2];

        // Rref = |ref - tx|
        float dxr = rxp - sx;
        float dyr = ryp - sy;
        float dzr = rzp - sz;
        float Rref = sqrtf(dxr*dxr + dyr*dyr + dzr*dzr);

        // k * freq : k = (4*pi/c) * j, same as python kc
        // exp(j * (4*pi/c) * freq * dR)
        float w = (4.0f * 3.14159265358979323846f / (float)%(C)s) * freqs[f];

        // thread-local partial sum (complex)
        float acc_re = 0.0f;
        float acc_im = 0.0f;

        for(int i = (int)threadIdx.x; i < N; i += (int)blockDim.x){
            float x = targets[3*i + 0];
            float y = targets[3*i + 1];
            float z = targets[3*i + 2];

            float dx = x - sx;
            float dy = y - sy;
            float dz = z - sz;
            float R = sqrtf(dx*dx + dy*dy + dz*dz);
            float dR = R - Rref;

            // phase = exp(j*w*dR) = cos + j sin
            float ph = w * dR;
            float c = cosf(ph);
            float s = sinf(ph);

            cuFloatComplex a = sigma[i]; // complex amplitude
            // a * (c + j s)
            float re = a.x*c - a.y*s;
            float im = a.x*s + a.y*c;

            acc_re += re;
            acc_im += im;
        }

        // block reduction (shared)
        __shared__ float sh_re[256];
        __shared__ float sh_im[256];

        int t = (int)threadIdx.x;
        sh_re[t] = acc_re;
        sh_im[t] = acc_im;
        __syncthreads();

        // assume blockDim.x == 256
        for(int stride = 128; stride > 0; stride >>= 1){
            if(t < stride){
                sh_re[t] += sh_re[t + stride];
                sh_im[t] += sh_im[t + stride];
            }
            __syncthreads();
        }

        if(t == 0){
            // atomic add to out[f]
            atomicAdd(&(out[f].x), sh_re[0]);
            atomicAdd(&(out[f].y), sh_im[0]);
        }
    }
    ''' % {"C": speed_of_light}

    mod = xp.RawModule(code=code, options=('-std=c++11',), name_expressions=['ph_accum_nearfield_c64'])
    k = mod.get_function('ph_accum_nearfield_c64')
    _get_ph_rawkernel_nearfield_c64._k = k
    return k

def _get_ph_rawkernel_farfield_c64(xp):
    if hasattr(_get_ph_rawkernel_farfield_c64, "_k"):
        return _get_ph_rawkernel_farfield_c64._k

    code = r'''
    #include <cuComplex.h>

    extern "C" __global__
    void ph_accum_farfield_c64(
        const float* __restrict__ los,             // (3,) unit LOS
        const float* __restrict__ refp,            // (3,)
        const float* __restrict__ targets,         // (3*N)
        const cuFloatComplex* __restrict__ sigma,  // (N,)
        const float* __restrict__ freqs,           // (nf,)
        const int N,
        const int nf,
        cuFloatComplex* __restrict__ out
    ){
        int f = (int)blockIdx.x;
        if(f >= nf) return;

        float lx = los[0], ly = los[1], lz = los[2];
        float rxp = refp[0], ryp = refp[1], rzp = refp[2];

        float w = (4.0f * 3.14159265358979323846f / (float)%(C)s) * freqs[f];

        float acc_re = 0.0f;
        float acc_im = 0.0f;

        for(int i = (int)threadIdx.x; i < N; i += (int)blockDim.x){
            float x = targets[3*i + 0] - rxp;
            float y = targets[3*i + 1] - ryp;
            float z = targets[3*i + 2] - rzp;

            float dR = x*lx + y*ly + z*lz;

            float ph = w * dR;
            float c = cosf(ph);
            float s = sinf(ph);

            cuFloatComplex a = sigma[i];
            float re = a.x*c - a.y*s;
            float im = a.x*s + a.y*c;

            acc_re += re;
            acc_im += im;
        }

        __shared__ float sh_re[256];
        __shared__ float sh_im[256];
        int t = (int)threadIdx.x;
        sh_re[t] = acc_re;
        sh_im[t] = acc_im;
        __syncthreads();

        for(int stride = 128; stride > 0; stride >>= 1){
            if(t < stride){
                sh_re[t] += sh_re[t + stride];
                sh_im[t] += sh_im[t + stride];
            }
            __syncthreads();
        }

        if(t == 0){
            atomicAdd(&(out[f].x), sh_re[0]);
            atomicAdd(&(out[f].y), sh_im[0]);
        }
    }
    ''' % {"C": speed_of_light}

    mod = xp.RawModule(code=code, options=('-std=c++11',), name_expressions=['ph_accum_farfield_c64'])
    k = mod.get_function('ph_accum_farfield_c64')
    _get_ph_rawkernel_farfield_c64._k = k
    return k
