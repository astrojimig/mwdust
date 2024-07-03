###############################################################################
#
#   Schultheis14: extinction model from Schultheis et al. (2014)
#   https://ui.adsabs.harvard.edu/abs/2014A%26A...566A.120S/abstract
#
###############################################################################
import os, os.path
import numpy
from astropy.io import ascii
from astropy.table import join
from scipy import interpolate
from mwdust.util.download import dust_dir, downloader
from mwdust.DustMap3D import DustMap3D
from mwdust.util.extCurves import aebv
_DEGTORAD = numpy.pi/180.
_schultheis14dir = os.path.join(dust_dir, 'schultheis14')
class Schultheis14(DustMap3D):
    """extinction model from Schultheis et al. (2014)
     https://ui.adsabs.harvard.edu/abs/2014A%26A...566A.120S/abstract
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
        self._table1hk_data = ascii.read(os.path.join(_schultheis14dir, 'table1hk.dat'), 
                                         readme = os.path.join(_schultheis14dir,'ReadMe'),
                                         guess=False, format='cds', fill_values=[('', '-999')]) 
        self._table1jk_data = ascii.read(os.path.join(_schultheis14dir, 'table1jk.dat'), 
                                         readme = os.path.join(_schultheis14dir,'ReadMe'),
                                         guess=False, format='cds', fill_values=[('', '-999')])
        # Join both tables together
        self._schultheis14_data = join(self._table1hk_data, self._table1jk_data)
        
        # Some summaries
        self._lmin= numpy.amin(self._schultheis14_data['GLON'])
        self._lmax= numpy.amax(self._schultheis14_data['GLON'])
        self._bmin= numpy.amin(self._schultheis14_data['GLAT'])
        self._bmax= numpy.amax(self._schultheis14_data['GLAT'])
        self._dl= self._lmax - self._lmin
        self._db= self._bmax - self._bmin
        # Define distance bins
        self._distance_bins_definitions = [
            {"bin_name": "00", "d_min": 0.0, "d_max": 0.5},
            {"bin_name": "05", "d_min": 0.5, "d_max": 1.0},
            {"bin_name": "10", "d_min": 1.0, "d_max": 1.5},
            {"bin_name": "15", "d_min": 1.5, "d_max": 2.0},
            {"bin_name": "20", "d_min": 2.0, "d_max": 2.5},
            {"bin_name": "25", "d_min": 2.5, "d_max": 3.0},
            {"bin_name": "30", "d_min": 3.0, "d_max": 3.5},
            {"bin_name": "35", "d_min": 3.5, "d_max": 4.0},
            {"bin_name": "40", "d_min": 4.0, "d_max": 4.5},
            {"bin_name": "45", "d_min": 4.5, "d_max": 5.0},
            {"bin_name": "50", "d_min": 5.0, "d_max": 5.5},
            {"bin_name": "55", "d_min": 5.5, "d_max": 6.0},
            {"bin_name": "60", "d_min": 6.0, "d_max": 6.5},
            {"bin_name": "65", "d_min": 6.5, "d_max": 7.0},
            {"bin_name": "70", "d_min": 7.0, "d_max": 7.5},
            {"bin_name": "75", "d_min": 7.5, "d_max": 8.0},
            {"bin_name": "80", "d_min": 8.0, "d_max": 8.5},
            {"bin_name": "85", "d_min": 8.5, "d_max": 9.0},
            {"bin_name": "90", "d_min": 9.0, "d_max": 9.5},
            {"bin_name": "95", "d_min": 9.5, "d_max": 10.0},
            {"bin_name": "100", "d_min": 10.0, "d_max": 10.5}
            ]
        self._ndistbin= len(self._distance_bins_definitions)
        self._ds= numpy.arange(0.25, 10.5, 0.5) # distance bins (assumed center)
        self._ds[0] = 0 #fix first bin
        # For dust_vals
        self._sintheta= numpy.sin((90.-self._schultheis14_data['GLAT'])*_DEGTORAD)
        self._costheta= numpy.cos((90.-self._schultheis14_data['GLAT'])*_DEGTORAD)
        self._sinphi= numpy.sin(self._schultheis14_data['GLON']*_DEGTORAD)
        self._cosphi= numpy.cos(self._schultheis14_data['GLON']*_DEGTORAD)
        # array to cache interpolated extinctions
        self._intps= numpy.zeros(len(self._schultheis14_data),dtype='object')
        return None
    
    def _evaluate(self, l, b, d, filt='(H-K)', _lbIndx=None):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the dust-map
        INPUT:
           l- Galactic longitude (deg)
           b- Galactic latitude (deg)
           d- distance (kpc) can be array
           filt- extinction value to use: "(H-K)" by default or "(J-K)"
        OUTPUT:
           extinction
        HISTORY:
           2024-07-03 - Adapted from Sale14 - Imig (STScI)
        """
        available_filters = ["(H-K)", "(J-K)"]
        if filt not in available_filters:
            raise ValueError(f"Filter {filt} not recognized. Please choose from: {available_filters} ")
        if isinstance(l,numpy.ndarray) or isinstance(b,numpy.ndarray):
            raise NotImplementedError("array input for l and b for Schultheis14 et al. (2014) dust map not implemented")
        if _lbIndx is None: lbIndx= self._lbIndx(l,b)
        else: lbIndx= _lbIndx
        if self._intps[lbIndx] != 0:
            out= self._intps[lbIndx](d)
        else:
            tlbData= self.lbData(l,b,filt=filt)
            interpData=\
                interpolate.InterpolatedUnivariateSpline(self._ds,
                                                         tlbData['a0'],
                                                         k=1)
            out= interpData(d)
            self._intps[lbIndx]= interpData
        if self._filter is None:
            return out/aebv('2MASS Ks',sf10=self._sf10)
        else:
            return out/aebv('2MASS Ks',sf10=self._sf10)\
                *aebv(self._filter,sf10=self._sf10)    
            
    def _lbIndx(self,l,b):
        """Return the index in the _schultheis14_data array corresponding to this (l,b)"""
        if l <= self._lmin or l >= self._lmax \
                or b <= self._bmin or b >= self._bmax:
            raise IndexError("Given (l,b) pair not within the region covered by the Schultheis et al. (2014) dust map")
        return numpy.argmin((l-self._schultheis14_data['GLON'])**2./self._dl**2.\
                                +(b-self._schultheis14_data['GLAT'])**2./self._db**2.)
    
    def lbData(self,l,b,filt):
        """
        NAME:
           lbData
        PURPOSE:
           return the Schultheis et al. (2014) data corresponding to a given line of sight
        INPUT:
           l- Galactic longitude (deg)
           b- Galactic latitude (deg)
           filt- extinction value to use: "(H-K)" by default or "(J-K)"
        OUTPUT:
            out- recarray of extinction values and errors for each distance bin
        HISTORY:
           2024-07-03 - Written - Imig (STScI)
        """
        #Find correct entry
        lbIndx= self._lbIndx(l,b)
        #Build output array
        out= numpy.recarray((self._ndistbin,),
                            dtype=[('a0', 'f8'),
                                   ('e_a0','f8')])
        
        for ii,distbin in enumerate(self._distance_bins_definitions):
            out[ii]['a0']= self._schultheis14_data[lbIndx][f"E{filt}{distbin['bin_name']}"]
            out[ii]['e_a0']= self._schultheis14_data[lbIndx][f"e_E{filt}{distbin['bin_name']}"]
        return out


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
