from unittest import TestCase
from unittest.mock import patch

from aspire import __version__
from aspire.utils import get_full_version
from aspire.utils.misc import powerset


class UtilsTestCase(TestCase):
    def testGetFullVersion(self):
        """Test typical version string response is coherent with package."""

        self.assertTrue(get_full_version().startswith(__version__))

    @patch('os.path.isdir')
    def testGetFullVersionPath(self, d_mock):
        """Test not isdir case of get_full_version."""

        d_mock.return_value = False

        self.assertTrue(get_full_version() == __version__)

    @patch('subprocess.check_output')
    def testGetFullVersionSrc(self, p_mock):
        """Test subprocess exception case of get_full_version."""

        p_mock.side_effect = FileNotFoundError

        self.assertTrue(get_full_version() == __version__ + '.src')

    @patch('subprocess.check_output')
    def testGetFullVersionUnexpected(self, p_mock):
        """Test unexpected exception case of get_full_version."""

        p_mock.side_effect = RuntimeError

        self.assertTrue(get_full_version() == __version__ + '.x')

    def testPowerset(self):
        ref = sorted([(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)])
        s = range(1, 4)
        self.assertTrue(sorted(list(powerset(s))) == ref)