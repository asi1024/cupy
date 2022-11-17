"""Module for RBF interpolation."""
import warnings
from itertools import combinations_with_replacement

import cupy as cp
from scipy.spatial import KDTree        # FIXME
from scipy.special import comb          # FIXME


# individual kernels et al

def linear(r):
    return -r


def thin_plate_spline(r):
    if r == 0:
        return 0.0
    else:
        return r**2*cp.log(r)


def cubic(r):
    return r**3


def quintic(r):
    return -r**5


def multiquadric(r):
    return -cp.sqrt(r**2 + 1)


def inverse_multiquadric(r):
    return 1/cp.sqrt(r**2 + 1)


def inverse_quadratic(r):
    return 1/(r**2 + 1)


def gaussian(r):
    return cp.exp(-r**2)


NAME_TO_FUNC = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian
}


def kernel_vector(x, y, kernel_func, out):
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in range(y.shape[0]):
        out[i] = kernel_func(cp.linalg.norm(x - y[i]))


def polynomial_vector(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    for i in range(powers.shape[0]):
        out[i] = cp.prod(x**powers[i])


def kernel_matrix(x, kernel_func, out):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(i+1):
            out[i, j] = kernel_func(cp.linalg.norm(x[i] - x[j]))
            out[j, i] = out[i, j]


def polynomial_matrix(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = cp.prod(x[i]**powers[j])


# pythran export _kernel_matrix(float[:, :], str)
def _kernel_matrix(x, kernel):
    """Return RBFs, with centers at `x`, evaluated at `x`."""
    out = cp.empty((x.shape[0], x.shape[0]), dtype=float)
    kernel_func = NAME_TO_FUNC[kernel]
    kernel_matrix(x, kernel_func, out)
    return out


# pythran export _polynomial_matrix(float[:, :], int[:, :])
def _polynomial_matrix(x, powers):
    """Return monomials, with exponents from `powers`, evaluated at `x`."""
    out = cp.empty((x.shape[0], powers.shape[0]), dtype=float)
    polynomial_matrix(x, powers, out)
    return out


# pythran export _build_system(float[:, :],
#                              float[:, :],
#                              float[:],
#                              str,
#                              float,
#                              int[:, :])
def _build_system(y, d, smoothing, kernel, epsilon, powers):
    """Build the system used to solve for the RBF interpolant coefficients.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    lhs : (P + R, P + R) float ndarray
        Left-hand side matrix.
    rhs : (P + R, S) float ndarray
        Right-hand side matrix.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]

    # Shift and scale the polynomial domain to be between -1 and 1
    mins = cp.min(y, axis=0)
    maxs = cp.max(y, axis=0)
    shift = (maxs + mins)/2
    scale = (maxs - mins)/2
    # The scale may be zero if there is a single point or all the points have
    # the same value for some dimension. Avoid division by zero by replacing
    # zeros with ones.
    scale[scale == 0.0] = 1.0

    yeps = y*epsilon
    yhat = (y - shift)/scale

    # Transpose to make the array fortran contiguous. This is required for
    # dgesv to not make a copy of lhs.
    lhs = cp.empty((p + r, p + r), dtype=float).T
    kernel_matrix(yeps, kernel_func, lhs[:p, :p])
    polynomial_matrix(yhat, powers, lhs[:p, p:])
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    for i in range(p):
        lhs[i, i] += smoothing[i]

    # Transpose to make the array fortran contiguous.
    rhs = cp.empty((s, p + r), dtype=float).T
    rhs[:p] = d
    rhs[p:] = 0.0

    return lhs, rhs, shift, scale


# pythran export _build_evaluation_coefficients(float[:, :],
#                          float[:, :],
#                          str,
#                          float,
#                          int[:, :],
#                          float[:],
#                          float[:])
def _build_evaluation_coefficients(x, y, kernel, epsilon, powers,
                                   shift, scale):
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Shifts the polynomial domain for numerical stability.
    scale : (N,) float ndarray
        Scales the polynomial domain for numerical stability.

    Returns
    -------
    (Q, P + R) float ndarray

    """
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]

    yeps = y*epsilon
    xeps = x*epsilon
    xhat = (x - shift)/scale

    vec = cp.empty((q, p + r), dtype=float)
    for i in range(q):
        kernel_vector(xeps[i], yeps, kernel_func, vec[i, :p])
        polynomial_vector(xhat[i], powers, vec[i, p:])

    return vec


###############################################################################

# These RBFs are implemented.
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian"
}


# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms.
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2
}


def _monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = comb(degree + ndim, ndim, exact=True)
    out = cp.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out


def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
    """Build and solve the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    coeffs : (P + R, S) float ndarray
        Coefficients for each RBF and monomial.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    lhs, rhs, shift, scale = _build_system(
        y, d, smoothing, kernel, epsilon, powers
    )
    coeffs = cp.linalg.solve(lhs, rhs)
    return shift, scale, coeffs


class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions.

    Parameters
    ----------
    y : (P, N) array_like
        Data point coordinates.
    d : (P, ...) array_like
        Data values at `y`.
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    smoothing : float or (P,) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    kernel : str, optional
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r**2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r**2)``
            - 'inverse_quadratic'    : ``1/(1 + r**2)``
            - 'gaussian'             : ``exp(-r**2)``

        Default is 'thin_plate_spline'.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. If `kernel` is
        'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
        1 and can be ignored because it has the same effect as scaling the
        smoothing parameter. Otherwise, this must be specified.
    degree : int, optional
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2

        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.

    Notes
    -----
    An RBF is a scalar valued function in N-dimensional space whose value at
    :math:`x` can be expressed in terms of :math:`r=||x - c||`, where :math:`c`
    is the center of the RBF.

    An RBF interpolant for the vector of data values :math:`d`, which are from
    locations :math:`y`, is a linear combination of RBFs centered at :math:`y`
    plus a polynomial with a specified degree. The RBF interpolant is written
    as

    .. math::
        f(x) = K(x, y) a + P(x) b,

    where :math:`K(x, y)` is a matrix of RBFs with centers at :math:`y`
    evaluated at the points :math:`x`, and :math:`P(x)` is a matrix of
    monomials, which span polynomials with the specified degree, evaluated at
    :math:`x`. The coefficients :math:`a` and :math:`b` are the solution to the
    linear equations

    .. math::
        (K(y, y) + \\lambda I) a + P(y) b = d

    and

    .. math::
        P(y)^T a = 0,

    where :math:`\\lambda` is a non-negative smoothing parameter that controls
    how well we want to fit the data. The data are fit exactly when the
    smoothing parameter is 0.

    The above system is uniquely solvable if the following requirements are
    met:

        - :math:`P(y)` must have full column rank. :math:`P(y)` always has full
          column rank when `degree` is -1 or 0. When `degree` is 1,
          :math:`P(y)` has full column rank if the data point locations are not
          all collinear (N=2), coplanar (N=3), etc.
        - If `kernel` is 'multiquadric', 'linear', 'thin_plate_spline',
          'cubic', or 'quintic', then `degree` must not be lower than the
          minimum value listed above.
        - If `smoothing` is 0, then each data point location must be distinct.

    When using an RBF that is not scale invariant ('multiquadric',
    'inverse_multiquadric', 'inverse_quadratic', or 'gaussian'), an appropriate
    shape parameter must be chosen (e.g., through cross validation). Smaller
    values for the shape parameter correspond to wider RBFs. The problem can
    become ill-conditioned or singular when the shape parameter is too small.

    The memory required to solve for the RBF interpolation coefficients
    increases quadratically with the number of data points, which can become
    impractical when interpolating more than about a thousand data points.
    To overcome memory limitations for large interpolation problems, the
    `neighbors` argument can be specified to compute an RBF interpolant for
    each evaluation point using only the nearest data points.

    .. versionadded:: 1.7.0

    See Also
    --------
    NearestNDInterpolator
    LinearNDInterpolator
    CloughTocher2DInterpolator

    References
    ----------
    .. [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
        World Scientific Publishing Co.

    .. [2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

    .. [3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    .. [4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

    Examples
    --------
    Demonstrate interpolating scattered data to a grid in 2-D.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import RBFInterpolator
    >>> from scipy.stats.qmc import Halton

    >>> rng = np.random.default_rng()
    >>> xobs = 2*Halton(2, seed=rng).random(100) - 1
    >>> yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

    >>> xgrid = np.mgrid[-1:1:50j, -1:1:50j]
    >>> xflat = xgrid.reshape(2, -1).T
    >>> yflat = RBFInterpolator(xobs, yobs)(xflat)
    >>> ygrid = yflat.reshape(50, 50)

    >>> fig, ax = plt.subplots()
    >>> ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    >>> p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    >>> fig.colorbar(p)
    >>> plt.show()

    """

    def __init__(self, y, d,
                 neighbors=None,
                 smoothing=0.0,
                 kernel="thin_plate_spline",
                 epsilon=None,
                 degree=None):
        y = cp.asarray(y, dtype=float, order="C")
        if y.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")

        ny, ndim = y.shape

        d_dtype = complex if cp.iscomplexobj(d) else float
        d = cp.asarray(d, dtype=d_dtype, order="C")
        if d.shape[0] != ny:
            raise ValueError(
                f"Expected the first axis of `d` to have length {ny}."
            )

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))
        # If `d` is complex, convert it to a float array with twice as many
        # columns. Otherwise, the LHS matrix would need to be converted to
        # complex and take up 2x more memory than necessary.
        d = d.view(float)

        if cp.isscalar(smoothing):
            smoothing = cp.full(ny, smoothing, dtype=float)
        else:
            smoothing = cp.asarray(smoothing, dtype=float, order="C")
            if smoothing.shape != (ny,):
                raise ValueError(
                    "Expected `smoothing` to be a scalar or have shape "
                    f"({ny},)."
                )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of "
                    f"{_SCALE_INVARIANT}."
                )
        else:
            epsilon = float(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif degree < min_degree:
                warnings.warn(
                    f"`degree` should not be below {min_degree} when `kernel` "
                    f"is '{kernel}'. The interpolant may not be uniquely "
                    "solvable, and the smoothing parameter may have an "
                    "unintuitive effect.",
                    UserWarning
                )

        if neighbors is None:
            nobs = ny
        else:
            # Make sure the number of nearest neighbors used for interpolation
            # does not exceed the number of observations.
            neighbors = int(min(neighbors, ny))
            nobs = neighbors

        powers = _monomial_powers(ndim, degree)
        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}."
            )

        if neighbors is None:
            shift, scale, coeffs = _build_and_solve_system(
                y, d, smoothing, kernel, epsilon, powers
            )

            # Make these attributes private since they do not always exist.
            self._shift = shift
            self._scale = scale
            self._coeffs = coeffs

        else:
            self._tree = KDTree(y)

        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.powers = powers

    def _chunk_evaluator(
            self,
            x,
            y,
            shift,
            scale,
            coeffs,
            memory_budget=1000000
    ):
        """
        Evaluate the interpolation while controlling memory consumption.
        We chunk the input if we need more memory than specified.

        Parameters
        ----------
        x : (Q, N) float ndarray
            array of points on which to evaluate
        y: (P, N) float ndarray
            array of points on which we know function values
        shift: (N, ) ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs: (P+R, S) float ndarray
            Coefficients in front of basis functions
        memory_budget: int
            Total amount of memory (in units of sizeof(float)) we wish
            to devote for storing the array of coefficients for
            interpolated points. If we need more memory than that, we
            chunk the input.

        Returns
        -------
        (Q, S) float ndarray
        Interpolated array
        """
        nx, ndim = x.shape
        if self.neighbors is None:
            nnei = len(y)
        else:
            nnei = self.neighbors
        # in each chunk we consume the same space we already occupy
        chunksize = memory_budget // ((self.powers.shape[0] + nnei)) + 1
        if chunksize <= nx:
            out = cp.empty((nx, self.d.shape[1]), dtype=float)
            for i in range(0, nx, chunksize):
                vec = _build_evaluation_coefficients(
                    x[i:i + chunksize, :],
                    y,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                    shift,
                    scale)
                out[i:i + chunksize, :] = cp.dot(vec, coeffs)
        else:
            vec = _build_evaluation_coefficients(
                x,
                y,
                self.kernel,
                self.epsilon,
                self.powers,
                shift,
                scale)
            out = cp.dot(vec, coeffs)
        return out

    def __call__(self, x):
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) array_like
            Evaluation point coordinates.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """
        x = cp.asarray(x, dtype=float, order="C")
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError("Expected the second axis of `x` to have length "
                             f"{self.y.shape[1]}.")

        # Our memory budget for storing RBF coefficients is
        # based on how many floats in memory we already occupy
        # If this number is below 1e6 we just use 1e6
        # This memory budget is used to decide how we chunk
        # the inputs
        memory_budget = max(x.size + self.y.size + self.d.size, 1000000)

        if self.neighbors is None:
            out = self._chunk_evaluator(
                x,
                self.y,
                self._shift,
                self._scale,
                self._coeffs,
                memory_budget=memory_budget)
        else:
            # Get the indices of the k nearest observation points to each
            # evaluation point.
            _, yindices = self._tree.query(x, self.neighbors)
            if self.neighbors == 1:
                # `KDTree` squeezes the output when neighbors=1.
                yindices = yindices[:, None]

            # Multiple evaluation points may have the same neighborhood of
            # observation points. Make the neighborhoods unique so that we only
            # compute the interpolation coefficients once for each
            # neighborhood.
            yindices = cp.sort(yindices, axis=1)
            yindices, inv = cp.unique(yindices, return_inverse=True, axis=0)
            # `inv` tells us which neighborhood will be used by each evaluation
            # point. Now we find which evaluation points will be using each
            # neighborhood.
            xindices = [[] for _ in range(len(yindices))]
            for i, j in enumerate(inv):
                xindices[j].append(i)

            out = cp.empty((nx, self.d.shape[1]), dtype=float)
            for xidx, yidx in zip(xindices, yindices):
                # `yidx` are the indices of the observations in this
                # neighborhood. `xidx` are the indices of the evaluation points
                # that are using this neighborhood.
                xnbr = x[xidx]
                ynbr = self.y[yidx]
                dnbr = self.d[yidx]
                snbr = self.smoothing[yidx]
                shift, scale, coeffs = _build_and_solve_system(
                    ynbr,
                    dnbr,
                    snbr,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                )
                out[xidx] = self._chunk_evaluator(
                    xnbr,
                    ynbr,
                    shift,
                    scale,
                    coeffs,
                    memory_budget=memory_budget)

        out = out.view(self.d_dtype)
        out = out.reshape((nx, ) + self.d_shape)
        return out
