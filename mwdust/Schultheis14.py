###############################################################################
#
#   Schultheis14: extinction model from Schultheis et al. (2014)
#   https://ui.adsabs.harvard.edu/abs/2014A%26A...566A.120S/abstract
#
###############################################################################
import os, os.path
import numpy
import h5py
import tarfile
from mwdust.util.download import dust_dir, downloader
from mwdust.HierarchicalHealpixMap import HierarchicalHealpixMap
_DEGTORAD= numpy.pi/180.
_schultheis14dir= os.path.join(dust_dir, 'schultheis14')
class Schultheis14(HierarchicalHealpixMap):
    """extinction model from Schultheis et al. (2014)
     https://ui.adsabs.harvard.edu/abs/2014A%26A...566A.120S/abstract
     """
    def __init__(self,filter=None,sf10=True,load_samples=False,
                 interpk=1):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize the Schultheis et al. (2014) dust map
        INPUT:
           filter= filter to return the extinction in
           sf10= (True) if True, use the Schlafly & Finkbeiner calibrations
           load_samples= (False) if True, also load the samples
           interpk= (1) interpolation order
        OUTPUT:
           object
        HISTORY:
           2024-07-03 - Added - Imig (STScI)
        """
        HierarchicalHealpixMap.__init__(self,filter=filter,sf10=sf10)
        #Read the map
        with h5py.File(os.path.join(_schultheis14dir,'bayestar2017.h5'),'r') \
                as schultheisdata:
            self._pix_info= schultheisdata['/pixel_info'][:]
            if load_samples:
                self._samples= schultheisdata['/samples'][:]
            self._best_fit= schultheisdata['/best_fit'][:]
            self._GR= schultheisdata['/GRDiagnostic'][:]
        # Utilities
        self._distmods= numpy.linspace(4,19,31)
        self._minnside= numpy.amin(self._pix_info['nside'])
        self._maxnside= numpy.amax(self._pix_info['nside'])
        nlevels= int(numpy.log2(self._maxnside//self._minnside))+1
        self._nsides= [self._maxnside//2**ii for ii in range(nlevels)]
        self._indexArray= numpy.arange(len(self._pix_info['healpix_index']))
        # For the interpolation
        self._intps= numpy.zeros(len(self._pix_info['healpix_index']),
                                 dtype='object') #array to cache interpolated extinctions
        self._interpk= interpk
        return None

    def substitute_sample(self,samplenum):
        """
        NAME:
           substitute_sample
        PURPOSE:
           substitute a sample for the best fit to get the extinction from a sample with the same tools; need to have setup the instance with load_samples=True
        INPUT:
           samplenum - sample's index to load
        OUTPUT:
           (none; just resets the instance to use the sample rather than the best fit; one cannot go back to the best fit after this))
        HISTORY:
           2024-07-03 - Added - Imig (STScI)
        """
        # Substitute the sample
        self._best_fit= self._samples[:,samplenum,:]
        # Reset the cache
        self._intps= numpy.zeros(len(self._pix_info['healpix_index']),
                                 dtype='object') #array to cache interpolated extinctions
        return None

    @classmethod
    def download(cls, test=False):
        # Download Schultheisdata et al. 2014 data
        schultheis14_path = os.path.join(dust_dir, "schultheis14")
        if not os.path.exists(os.path.join(schultheis14_path, "table1hk.dat")):
            if not os.path.exists(schultheis14_path):
                os.mkdir(schultheis14_path)
            _SCHULTHEIS14_URLS = ["https://cdsarc.cds.unistra.fr/ftp/J/A+A/566/A120/ReadMe",
                                  "https://cdsarc.cds.unistra.fr/ftp/J/A+A/566/A120/table1hk.dat.gz",
                                  "https://cdsarc.cds.unistra.fr/ftp/J/A+A/566/A120/table1jk.dat.gz"]
            
            for url in _SCHULTHEIS14_URLS:
                schultheis14_file = url.split('/')[-1] #isolate filename
                downloader(url, os.path.join(schultheis14_path, schultheis14_file), cls.__name__, test=test)
                if schultheis14_file.endswith('gz'): # unzip gz files
                     os.system('gunzip ' + os.path.join(schultheis14_path, schultheis14_file))
        return None
