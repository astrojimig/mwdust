###############################################################################
#
#   Combined24: extinction model obtained from a combination of the Combined19
#               map [itself a combination of Marshall et al. (2006), Green et al. (2019),
#               Drimmel et al. (2003)], and updated values from
#               Schultheis et al (2014) for the inner galaxy.
#
###############################################################################
import os, os.path
import numpy
from mwdust.util.download import dust_dir, downloader
from mwdust.DustMap3D import DustMap3D
from mwdust.util.extCurves import aebv
from mwdust.Schultheis14 import Schultheis14
from mwdust.Combined19 import Combined19
_DEGTORAD = numpy.pi/180.

class Combined24(DustMap3D):
    """
    Combination of Marshall et al. (2006), Green et al. (2019),
    Drimmel et al. (2003), and Schultheis et al (2014).
    Preferentially uses Schultheis14 when available, or if out of bounds,
    uses Combined19.
    """
    def __init__(self,filter=None,sf10=True):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize the Schultheis et al. (2014) dust map
        INPUT:
           filter= filter to return the extinction in
           sf10= (True) if True, use the Schlafly & Finkbeiner calibrations
        OUTPUT:
           object
        HISTORY:
           2024-07-03 - Added - Imig (STScI)
        """
        DustMap3D.__init__(self,filter=filter)
        self._sf10 = sf10
        #Read the data
        self.Combined19 = Combined19()
        self.Schultheis14 = Schultheis14()
        return None
    
    def _evaluate(self, l, b, d):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the dust-map
        INPUT:
           l- Galactic longitude (deg)
           b- Galactic latitude (deg)
           d- distance (kpc) can be array
        OUTPUT:
           extinction
        HISTORY:
           2024-07-03 - Added - Imig (STScI)
        """
        # If out of bounds of Schultheis14:
        if l <= self.Schultheis14._lmin or l >= self.Schultheis14._lmax \
                or b <= self.Schultheis14._bmin or b >= self.Schultheis14._bmax:
            out = self.Combined19(l,b,d)
        else:
            out = self.Schultheis14(l,b,d)
        return out


    @classmethod
    def download(cls, test=False):
        # Download dustmaps
        Combined19.download(test=test)
        Schultheis14.download(test=test)
        return None
