import os
from unittest import TestCase
from shutil import copyfile
import tempfile

import numpy as np

from aspire.ctf import estimate_ctf

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

class CtfEstimatorTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testEstimateCTF(self):
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            # Copy input file
            copyfile(os.path.join(DATA_DIR,'sample.mrc'),
                     os.path.join(tmp_input_dir, 'input1.mrc'))
            
            with tempfile.TemporaryDirectory() as tmp_output_dir:
                # Returns results in output_dir
                estimate_ctf(
                    data_folder=tmp_input_dir,
                    pixel_size=1,
                    cs=2.,
                    amplitude_contrast=0.07,
                    voltage=300.,
                    num_tapers=2,
                    psd_size=512,
                    g_min=30.,
                    g_max=5.,
                    corr=1.,
                    output_dir=tmp_output_dir,
                    repro=True)
                

    
