import unittest
import numpy as np
import numpy.testing as npt
import util
import mapping_ops_3dof as mops3dof
import mapping_ops_6dof as mops6dof


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
        npt.assert_almost_equal(mops3dof.compose(r, a)[0], np.zeros(3))
        self.assertEqual(Pr.shape, (3, 3))

    def test_compose_6dof(self):
        a = np.array([2, 2, 2, np.pi, np.pi, np.pi])
        b = np.array([2, 2, 2])
        c = np.array([2, 2, 2, np.pi, np.pi, np.pi])

        Pa = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        Pc = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        r_expected = np.array([4.,  4.,  4.,  0.,  0.,  0.])

        r, Pr = mops6dof.compose(a, b)
        npt.assert_almost_equal(r, r_expected[0:3])

        r, Pr = mops6dof.compose(a, c, Pa, Pc)
        npt.assert_almost_equal(r, r_expected)
        self.assertEqual(Pr.shape, (6, 6))

    def test_inv_6dof(self):
        a = np.array([2, 2, 2, np.pi, np.pi, np.pi])
        Pa = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        r, Pr = mops6dof.inv(a, Pa)
        npt.assert_almost_equal(mops6dof.compose(r, a)[0], np.zeros(6))
        self.assertEqual(Pr.shape, (6, 6))

if __name__ == '__main__':
    unittest.main()
