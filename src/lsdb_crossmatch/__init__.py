from ._version import __version__
from citation_compass import cite_inline

__all__ = ["__version__"]

# Add default citations for the core packages (LSDB and HATS)
cite_inline(
    "LSDB", "LSDB - Caplar et. al. 2025 (https://ui.adsabs.harvard.edu/abs/2025arXiv250102103C/abstract)"
)
cite_inline("HATS", "HATS - Caplar et. al. 2025 (https://www.ivoa.net/documents/Notes/HATS/)")
