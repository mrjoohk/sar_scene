import numpy as np

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
