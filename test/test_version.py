import re
import unittest


class TestVersion(unittest.TestCase):

    def test_package_version(self):
        version_pattern = r"""
            v?
            (?:
                (?:(?P<epoch>[0-9]+)!)?                           # epoch
                (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
                (?P<pre>                                          # pre-release
                    [-_\.]?
                    (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
                    [-_\.]?
                    (?P<pre_n>[0-9]+)?
                )?
                (?P<post>                                         # post release
                    (?:-(?P<post_n1>[0-9]+))
                    |
                    (?:
                        [-_\.]?
                        (?P<post_l>post|rev|r)
                        [-_\.]?
                        (?P<post_n2>[0-9]+)?
                    )
                )?
                (?P<dev>                                          # dev release
                    [-_\.]?
                    (?P<dev_l>dev)
                    [-_\.]?
                    (?P<dev_n>[0-9]+)?
                )?
            )
            (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
        """

        regex = re.compile(
            r"^\s*" + version_pattern + r"\s*$",
            re.VERBOSE | re.IGNORECASE,
        )

        # read setup.py file
        def read(file):
            with open(file, "r", encoding="UTF-8") as input_file:
                text = input_file.read()
            return text

        try:
            version = read("version.txt")
        except FileNotFoundError:
            version = read("../version.txt")

        self.assertTrue(regex.match(version) is not None)
