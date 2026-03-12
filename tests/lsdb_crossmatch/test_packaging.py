import lsdb_crossmatch


def test_version():
    """Check to see that we can get the package version"""
    assert lsdb_crossmatch.__version__ is not None
