import unittest

import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context

from common import gpu_test


class TestNNabla(unittest.TestCase):
    def test_addition(self):
        nd_a = np.random.random()
        nd_b = np.random.random()

        # entry variables
        a = nn.Variable.from_numpy_array(nd_a)
        b = nn.Variable.from_numpy_array(nd_b)

        # add operation
        c = a + b

        # forward
        c.forward()

        self.assertEqual(c.d, nd_a + nd_b)

    @gpu_test
    def test_cuda_ext(self):
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)
