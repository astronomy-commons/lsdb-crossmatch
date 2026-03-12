from citation_compass import find_in_citations

import lsdb_crossmatch


def test_version():
    """Check to see that we can get the package version"""
    assert lsdb_crossmatch.__version__ is not None


def test_citation_compass_integration():
    """Check to see that the package is properly integrated with the citation compass."""

    # HATS and LSDB should always be cited when lsdb_crossmatch is imported.
    assert len(find_in_citations("LSDB - Caplar et. al. 2025")) > 0
    assert len(find_in_citations("HATS - Caplar et. al. 2025")) > 0
