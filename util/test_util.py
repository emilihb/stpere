import unittest
import numpy as np
import numpy.testing as npt
import util
import mapping_ops_3dof as mops3dof


class UtilTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(util.normalize(0.0), 0.0)
        self.assertLessEqual(util.normalize(4.0), 0.0)

    def test_compose_3dof(self):
        a = np.array([5., 5., np.pi])
        c = np.array([1., 2., np.pi])

        Pa = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        Pc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        r, Pr = mops3dof.compose(a, c)
        npt.assert_almost_equal(r, np.array([4., 3., 0.]))

        r, Pr = mops3dof.compose(a, c, Pa, Pc)
        npt.assert_almost_equal(r, np.array([4., 3., 0.]))
        self.assertEqual(Pr.shape, (3, 3))

    def test_inv_3dof(self):
        a = np.array([1.0, 1.0, np.pi])
        Pa = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])

        r, Pr = mops3dof.inv(a, Pa)
        npt.assert_almost_equal(r, np.array([1., 1., -np.pi]))
        self.assertEqual(Pr.shape, (3, 3))

if __name__ == '__main__':
    unittest.main()
