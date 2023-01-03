import logging
import os.path
from itertools import product
from unittest import TestCase

import numpy as np
from parameterized import parameterized, parameterized_class

from aspire.noise import CustomNoiseAdder, WhiteNoiseAdder, WhiteNoiseEstimator
from aspire.operators import FunctionFilter, ScalarFilter
from aspire.source.simulation import Simulation
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)

RESOLUTIONS = [64, 65]
#RESOLUTIONS = [64, 65, 128, 129, 256, 257, 512, 511]
VARIANCES = [10 ** (-x) for x in range(2, 5)]


@parameterized_class(("L"), [(r,) for r in RESOLUTIONS])
class SimTestCase(TestCase):
    # Note L needs to be large enough that we have sufficient image "corners".
    L = 64

    def setUp(self):

        # Setup a sim with no noise, no ctf, no shifts,
        #   using a compactly supported volume.
        # ie, clean centered projections.
        self.sim = Simulation(
            vols=AsymmetricVolume(L=self.L, C=1).generate(),
            n=16,
            offsets=0,
        )

    def tearDown(self):
        pass

    def testWhiteNoise0(self):
        """
        Test a clean sim (defaults to no NoiseAdder) estimates clean.
        """
        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        # Using a compactly supported volume with no additional noise
        #   should yield virtually no noise in the image corners.
        self.assertTrue(np.isclose(noise_variance, 0))


@parameterized_class(("L", "target_noise_var"), product(RESOLUTIONS, VARIANCES))
class FromSNRSimCase(TestCase):
    # Note L needs to be large enough that we have sufficient image "corners".
    L = 64
    target_noise_var = 0.1
    n = 128

    def setUp(self):

        # Setup a sim with no ctf, no shifts,
        #   using a compactly supported volume,
        # Later we will add a noise_adder of prescribed variance
        self.sim = Simulation(
            vols=AsymmetricVolume(L=self.L, C=1).generate(),
            n=self.n,
            offsets=0,
            amplitudes=1.0,
        )
        # Compute the variance of all pixels for a reference value.
        self.signal_var = np.var(self.sim.images[:].asnumpy().flatten())
        # Append the white noise adder
        self.sim.noise_adder = WhiteNoiseAdder(var=self.target_noise_var)

        self.target_snr = self.signal_var / self.target_noise_var

        # Setup another sim using a compactly supported volume,
        #   using `from_snr`.
        self.sim_from_snr = Simulation.from_snr(
            self.target_snr,
            vols=AsymmetricVolume(L=self.L, C=1).generate(),
            n=self.n,
            offsets=0,
            amplitudes=1.0,
        )

    def testWhiteNoise(self):
        """
        Test that prescribing noise directly by var, `from_snr`,
        and the `WhiteNoiseEstimator`estimator are close for a variety of inputs.
        """
        logger.info(f"xxx L={self.L}, target_noise_var={self.target_noise_var}")
        logger.info(
            f"xxxy {self.sim.estimate_signal_var()}, {self.sim_from_snr.estimate_signal_var()}"
        )
        # Sanity check
        self.assertTrue(
            np.isclose(
                self.sim_from_snr.estimate_signal_var(),
                self.sim.estimate_signal_var(),
                rtol=0.05,
            )
        )

        # Estimated noise variance when directly prescribed.
        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        # Using a compactly supported volume should yield
        #   virtually only generated noise in the image corners.
        self.assertTrue(np.isclose(noise_variance, self.target_noise_var, rtol=0.05))

        # Estimated noise variance when setup using `from_snr)
        noise_estimator_from_snr = WhiteNoiseEstimator(self.sim_from_snr, batchSize=512)
        noise_variance_from_snr = noise_estimator_from_snr.estimate()
        self.assertTrue(
            np.isclose(noise_variance_from_snr, self.target_noise_var, rtol=0.05)
        )


@parameterized_class(("L"), [(r,) for r in RESOLUTIONS])
class NoiseAdder(TestCase):
    L = 64
    dtype = np.float32

    def setUp(self):

        # Setup a sim with no noise, no ctf, no shifts,
        #   using a compactly supported volume.
        # ie, clean centered projections.
        self.sim = Simulation(
            vols=AsymmetricVolume(L=self.L, C=1, dtype=self.dtype).generate(),
            n=16,
            offsets=0,
        )

    def tearDown(self):
        pass

    def testWhiteRepr(self):
        """Test __repr__ does not crash."""
        x = WhiteNoiseAdder(var=1)
        logger.info(f"Example repr:\n{repr(x)}")

    def testWhiteStr(self):
        """Test __str__ does not crash."""
        x = WhiteNoiseAdder(var=1)
        logger.info(f"Example str:\n{str(x)}")

    def testCustomRepr(self):
        """Test __repr__ does not crash."""
        custom_filter = ScalarFilter(dim=2, value=1)
        x = CustomNoiseAdder(noise_filter=custom_filter)
        logger.info(f"Example repr:\n{repr(x)}")

    def testCustomStr(self):
        """Test __str__ does not crash."""
        custom_filter = ScalarFilter(dim=2, value=1)
        x = CustomNoiseAdder(noise_filter=custom_filter)
        logger.info(f"Example str:\n{str(x)}")

    @parameterized.expand([(10 ** (-x),) for x in range(1, 4)])
    def testCustomNoiseAdder(self, noise_var):
        """
        Custom Noise adder uses custom `Filter`.
        """
        logger.debug(
            f"testCustomNoiseAdder dtype={self.dtype} L={self.L} noise_var={noise_var}"
        )

        def pinkish_spectrum(x, y):
            s = x[-1] - x[-2]
            f = 2 * s / (np.hypot(x, y) + s)
            m = np.mean(f)
            return f * noise_var / m

        custom_filter = FunctionFilter(f=pinkish_spectrum)

        # Check we are achieving an estimate near the target
        self.sim.noise_adder = CustomNoiseAdder(noise_filter=custom_filter)
        est_noise_var = self.sim.noise_adder.noise_var
        logger.debug(f"Estimated Noise Variance {est_noise_var}")
        self.assertTrue(np.isclose(est_noise_var, noise_var, rtol=0.1))

    @parameterized.expand([(v,) for v in VARIANCES])
    def testWhiteNoiseAdder(self, noise_var):
        logger.debug(
            f"testWhiteNoiseAdder dtype={self.dtype} L={self.L} noise_var={noise_var}"
        )
        self.sim.noise_adder = WhiteNoiseAdder(var=noise_var)

        # Assert we have passed through the var exactly
        self.assertTrue(self.sim.noise_adder.noise_var == noise_var)

        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        # Match estimate within 1%
        self.assertTrue(np.isclose(noise_var, noise_estimator.estimate(), rtol=0.01))
