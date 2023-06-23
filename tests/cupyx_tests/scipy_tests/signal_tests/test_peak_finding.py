import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing
import cupyx.scipy.signal  # NOQA

import numpy as np

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@pytest.mark.xfail(
    runtime.is_hip and driver.get_build_version() < 5_00_00000,
    reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestPeakProminences:

    @pytest.mark.parametrize('x', [[1, 2, 3], []])
    @testing.numpy_cupy_allclose(scipy_name="scp", type_check=False)
    def test_empty(self, x, xp, scp):
        """
        Test if an empty array is returned if no peaks are provided.
        """
        out = scp.signal.peak_prominences(x, [])
        return out

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        """
        Test if height of prominences is correctly calculated in signal with
        rising baseline (peak widths are 1 sample).
        """
        # Prepare basic signal
        x = xp.array([-1, 1.2, 1.2, 1, 3.2, 1.3, 2.88, 2.1])
        peaks = xp.array([1, 2, 4, 6])
        # Test if calculation matches handcrafted result
        out = scp.signal.peak_prominences(x, peaks)
        return out

    @pytest.mark.parametrize(
        'x', [[0.0, 2, 1, 2, 1, 2, 0], [0, 1.0, 0, 1, 0, 1, 0]])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_edge_cases(self, x, xp, scp):
        """
        Test edge cases.
        """
        # Peaks have same height, prominence and bases
        x = xp.asarray(x)
        peaks = xp.asarray([1, 3, 5])
        out = scp.signal.peak_prominences(x, peaks)
        return out

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_non_contiguous(self, xp, scp):
        """
        Test with non-C-contiguous input arrays.
        """
        x = xp.repeat(xp.asarray([-9, 9, 9, 0, 3, 1.0]), 2)
        peaks = xp.repeat(xp.asarray([1, 2, 4]), 2)
        out = scp.signal.peak_prominences(x[::2], peaks[::2])
        return out

    @pytest.mark.parametrize('wlen', [8, 7, 6, 5, 3.2, 3, 1.1])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_wlen(self, wlen, xp, scp):
        """
        Test if wlen actually shrinks the evaluation range correctly.
        """
        x = xp.asarray([0, 1, 2, 3.0, 1, 0, -1])
        peak = xp.asarray([3])
        return scp.signal.peak_prominences(x, peak, wlen)

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_exceptions(self, mod):
        """
        Verify that exceptions and warnings are raised.
        """
        xp, scp = mod
        # x with dimension > 1
        with pytest.raises(ValueError, match='1-D array'):
            scp.signal.peak_prominences([[0, 1, 1, 0]], [1, 2])
        # peaks with dimension > 1
        with pytest.raises(ValueError, match='1-D array'):
            scp.signal.peak_prominences([0, 1, 1, 0], [[1, 2]])
        # x with dimension < 1
        with pytest.raises(ValueError, match='1-D array'):
            scp.signal.peak_prominences(3, [0,])

        # empty x with supplied
        with pytest.raises(ValueError, match='not a valid index'):
            scp.signal.peak_prominences([], [0])
        # invalid indices with non-empty x
        for p in [-100, -1, 3, 1000]:
            with pytest.raises(ValueError, match='not a valid index'):
                scp.signal.peak_prominences([1, 0, 2], [p])

        # wlen < 3
        with pytest.raises(ValueError, match='wlen'):
            scp.signal.peak_prominences(xp.arange(10), [3, 5], wlen=1)


@pytest.mark.xfail(
    runtime.is_hip and driver.get_build_version() < 5_00_00000,
    reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestFindPeaks:

    # Keys of optionally returned properties
    property_keys = {'peak_heights', 'left_thresholds', 'right_thresholds',
                     'prominences', 'left_bases', 'right_bases', 'widths',
                     'width_heights', 'left_ips', 'right_ips'}

    @testing.numpy_cupy_allclose(scipy_name="scp", type_check=False)
    def test_constant(self, xp, scp):
        """
        Test behavior for signal without local maxima.
        """
        open_interval = (None, None)
        peaks, props = scp.signal.find_peaks(
            xp.ones(10), height=open_interval, threshold=open_interval,
            prominence=open_interval, width=open_interval)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize(
        'plateau_size', [(None, None), 4, (None, 3.5), (5, 50)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_plateau_size(self, plateau_size, xp, scp):
        """
        Test plateau size condition for peaks.
        """
        # Prepare signal with peaks with peak_height == plateau_size
        plateau_sizes = xp.array([1, 2, 3, 4, 8, 20, 111])
        x = xp.zeros(plateau_sizes.size * 2 + 1)
        x[1::2] = plateau_sizes
        repeats = xp.ones(x.size, dtype=int)
        repeats[1::2] = x[1::2]
        x = xp.repeat(x, repeats.tolist())

        # Test full output
        peaks, props = scp.signal.find_peaks(x, plateau_size=plateau_size)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize(
        'height', [(None, None), 0.5, (None, 3), (2, 3)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_height_condition(self, height, xp, scp):
        """
        Test height condition for peaks.
        """
        x = xp.asarray([0., 1/3, 0., 2.5, 0, 4., 0])
        peaks, props = scp.signal.find_peaks(x, height=height)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize(
        'threshold', [(None, None), 2, 3.5, (None, 5), (None, 4), (2, 4)])
    @testing.numpy_cupy_allclose(scipy_name="scp", type_check=False)
    def test_threshold_condition(self, threshold, xp, scp):
        """
        Test threshold condition for peaks.
        """
        x = xp.asarray([0, 2, 1, 4, -1])
        peaks, props = scp.signal.find_peaks(x, threshold=threshold)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize('distance', [3, 3.0001])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_distance_condition(self, distance, xp, scp):
        """
        Test distance condition for peaks.
        """
        # Peaks of different height with constant distance 3
        peaks_all = xp.arange(1, 21, 3)
        x = xp.zeros(21)
        x[peaks_all] += xp.linspace(1, 2, peaks_all.size)
        peaks, props = scp.signal.find_peaks(x, distance=distance)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_distance_priority(self, xp, scp):
        # Test priority of peak removal
        x = xp.asarray([-2, 1, -1, 0, -3])
        # use distance > x size
        peaks, props = scp.signal.find_peaks(x, distance=10)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_prominence_condition(self, xp, scp):
        """
        Test prominence condition for peaks.
        """
        x = xp.linspace(0, 10, 100)
        peaks_true = xp.arange(1, 99, 2)
        offset = xp.linspace(1, 10, peaks_true.size)
        x[peaks_true] += offset
        interval = (3, 9)

        peaks, props = scp.signal.find_peaks(x, prominence=interval)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name="scp", type_check=False)
    def test_width_condition(self, xp, scp):
        """
        Test width condition for peaks.
        """
        x = xp.array([1, 0, 1, 2, 1, 0, -1, 4, 0])
        peaks, props = scp.signal.find_peaks(
            x, width=(None, 2), rel_height=0.75)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_properties(self, xp, scp):
        """
        Test returned properties.
        """
        open_interval = (None, None)
        x = xp.asarray([0, 1, 0, 2, 1.5, 0, 3, 0, 5, 9])
        peaks, props = scp.signal.find_peaks(
            x, height=open_interval, threshold=open_interval,
            prominence=open_interval, width=open_interval)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_raises(self, mod):
        """
        Test exceptions raised by function.
        """
        xp, scp = mod
        with pytest.raises(ValueError, match="1-D array"):
            scp.signal.find_peaks(xp.array(1))
        with pytest.raises(ValueError, match="1-D array"):
            scp.signal.find_peaks(xp.ones((2, 2)))
        with pytest.raises(ValueError, match="distance"):
            scp.signal.find_peaks(xp.arange(10), distance=-1)

    @pytest.mark.filterwarnings("ignore:some peaks have a prominence of 0",
                                "ignore:some peaks have a width of 0")
    @testing.numpy_cupy_allclose(scipy_name="scp", type_check=False)
    def test_wlen_smaller_plateau(self, xp, scp):
        """
        Test behavior of prominence and width calculation if the given window
        length is smaller than a peak's plateau size.

        Regression test for gh-9110.
        """
        peaks, props = scp.signal.find_peaks(
            [0, 1, 1, 1, 0], prominence=(None, None),
            width=(None, None), wlen=2)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])

    @pytest.mark.parametrize("kwargs", [
        {},
        {"distance": 3.0},
        {"prominence": (None, None)},
        {"width": (None, 2)},

    ])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_readonly_array(self, kwargs, xp, scp):
        """
        Test readonly arrays are accepted.
        """
        x = np.linspace(0, 10, 15)
        x_readonly = x.copy()
        x_readonly.flags.writeable = False

        peaks, props = scp.signal.find_peaks(x_readonly, **kwargs)
        return (peaks,) + tuple(
            [props[k] for k in self.property_keys if k in props])
