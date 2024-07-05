import unittest

import numpy as np
from scipy.linalg import logm

from quantfinlib.distance.matrix_divergence import (_assert_positive_definite,
                                                    log_det_divergence,
                                                    von_neuman_divergence)


class TestDivergences(unittest.TestCase):

    def setUp(self) -> None:
        self.pos_def_matrix1 = np.array([[2, 1], [1, 2]])
        self.pos_def_matrix2 = np.array([[2, -1], [-1, 2]])
        self.identity_matrix = np.eye(2)
        self.non_pos_def_matrix1 = np.array([[1, 1], [1, 1]])
        self.non_pos_def_matrix2 = np.array([[1, 2], [2, 1]])

    def test_assert_positive_definite(self):
        try:
            _assert_positive_definite(self.pos_def_matrix1, self.pos_def_matrix2)
        except AssertionError:
            self.fail("AssertionError raised when it shouldn't be")

        with self.assertRaises(AssertionError):
            _assert_positive_definite(self.pos_def_matrix1, self.non_pos_def_matrix1)

        with self.assertRaises(AssertionError):
            _assert_positive_definite(self.pos_def_matrix2, self.non_pos_def_matrix1)
        
        with self.assertRaises(AssertionError):
            _assert_positive_definite(self.non_pos_def_matrix2, self.non_pos_def_matrix2)
        
    def test_log_det_divergence(self):

        result = log_det_divergence(self.pos_def_matrix1, self.pos_def_matrix2)
        self.assertIsInstance(result, float, "log_det_divergence did not return a float")

        result_identity = log_det_divergence(self.identity_matrix, self.identity_matrix)
        self.assertAlmostEqual(result_identity, 0, "log_det_divergence did not return 0 for identity matrices")

        with self.assertRaises(AssertionError):
            log_det_divergence(self.pos_def_matrix1, self.non_pos_def_matrix1)
        
        result = log_det_divergence(self.pos_def_matrix1, self.identity_matrix)
        expected = np.trace(self.pos_def_matrix1) - np.linalg.slogdet(self.pos_def_matrix1)[1] - self.pos_def_matrix1.shape[0]
        self.assertAlmostEqual(result, expected, f"log_det_divergence between matrix = {self.pos_def_matrix1} and identity matrix did not return the expected value.")

    def test_von_neuman_divergence(self):

        result = von_neuman_divergence(self.pos_def_matrix1, self.pos_def_matrix2)
        self.assertIsInstance(result, float, "von_neuman_divergence did not return a float")

        result_identity = von_neuman_divergence(self.identity_matrix, self.identity_matrix)
        self.assertAlmostEqual(result_identity, 0, "von_neuman_divergence did not return 0 for identity matrices")

        with self.assertRaises(AssertionError):
            von_neuman_divergence(self.pos_def_matrix1, self.non_pos_def_matrix1)
        
        result = von_neuman_divergence(self.pos_def_matrix1, self.identity_matrix)
        log_m = logm(self.pos_def_matrix1)
        trace_m_log_m = np.trace(self.pos_def_matrix1 @ log_m)
        expected = trace_m_log_m - np.trace(self.pos_def_matrix1) + self.pos_def_matrix1.shape[0]
        self.assertAlmostEqual(result, expected, f"von_neuman_divergence between matrix = {self.pos_def_matrix1} and identity matrix did not return the expected value.")

        
    def tearDown(self) -> None:
        del self.pos_def_matrix1, self.pos_def_matrix2, self.identity_matrix, self.non_pos_def_matrix1, self.non_pos_def_matrix2