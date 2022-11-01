import uproot
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

from scipy.interpolate import RegularGridInterpolator, interp1d


from gammapy.maps import WcsNDMap, WcsGeom, MapAxis, RegionGeom, Map


class mscwReader():

    def __init__(self):
        pass
        # filename
        # self.object
        # self.data

    def setTarget(self, target):
        self.target = SkyCoord.from_name(target)
        self.targetCamX = (self.target.ra.deg - np.rad2deg(self.tel_ra)) * np.cos(self.tel_dec)
        self.targetCamY = self.target.dec.deg - np.rad2deg(self.tel_dec)




    def readFile(self, filename):

        self.filename = filename
        dataFile =  uproot.open(self.filename)

        data = dataFile["data"].arrays(library="numpy")

        # Real data
        try :

            pointingReduced = dataFile["pointingDataReduced"].arrays(library="numpy")

            self.tel_ra = np.median(pointingReduced["TelRAJ2000"])
            self.tel_dec = np.median(pointingReduced["TelDecJ2000"])
        # Simulated data
        except Exception as e:
            # Give default RA/Dec
            self.tel_ra = 0
            self.tel_dec = 0

        # Close the file to help with memory
        dataFile.close()
        return data


    def loadData(self, filename):

        # Obtain data
        data = self.readFile(filename)

        # Get coords of telescope pointing
        self.pointing = SkyCoord(ra = self.tel_ra *u.rad, dec=self.tel_dec *u.rad, frame='icrs')
        # self.pointing = pointing.transform_to('fk5')
        # Get geometry
        self.geom = WcsGeom.create(
                        npix=(40000, 40000), binsz=0.0001, skydir= self.pointing,
                        proj="TAN", frame="icrs"#, axes=[energy_axis]
                )

        # get world coordinate system
        self.wcs = WCS(self.geom.to_header())

        # camera coords to pixel
        ff = interp1d(np.arange(-2,2, 0.0001), np.arange(1,40001,1), kind="linear", bounds_error=False, fill_value=-999)

        # X is flipped lr
        x = ff(-data["Xoff_derot"])
        y = ff(data["Yoff_derot"])

        # Get RA/Dec of each event
        coords = self.wcs.pixel_to_world(x, y)
        ra = np.array([r.deg for r in coords.ra])
        dec = np.array([d.deg for d in coords.dec])
        
        # Mask out failed events
        emask = data["ErecS"] >0
        # Store data to dictionary
        self.data_dict = {
            "MSCW" : data["MSCW"][emask],
            "MSCL" : data["MSCL"][emask],
            "EmissionHeight" : data["EmissionHeight"][emask],
            "EChi2S" : data["EChi2S"][emask],
            "ENERGY" : data["ErecS"][emask], # required colnames 
            "dES" : data["dES"][emask],
            "SizeSecondMax" : data["SizeSecondMax"][emask],
            "EVENT_ID" : data["eventNumber"][emask], # required colnames 
            "RA" :  ra[emask], # required colnames 
            "DEC" :  dec[emask], # required colnames 
            "Xoff_derot" :  data["Xoff_derot"][emask], 
            "Yoff_derot" :  data["Yoff_derot"][emask], 
            "Xoff" :  data["Xoff"][emask], 
            "Yoff" :  data["Yoff"][emask], 
            "Xcore" :  data["Xcore"][emask], 
            "Ycore" :  data["Ycore"][emask], 
            "Core" : np.sqrt(data["Xcore"][emask]**2 + data["Ycore"][emask]**2),
            "NImages" : data["NImages"][emask],
            "TIME" : np.zeros(len(data["Yoff_derot"][emask])) # required colnames 
        }

        # Get the MC (true energy) for simulated data
        if "MCe0" in data.keys():
            self.data_dict["ENERGY_MC"] =  data["MCe0"][emask]


    def extractSimulatedSpectrum(self,fname):
    
        fdata = uproot.open(fname)
        
        # Open up the histogram and the headers
        histo = fdata["MChistos"]
        header = fdata["MC_runheader"]
        
        # Find the simulated spectral index
        fVspect = histo.member("fVSpectralIndex")
        specbins = np.array(fVspect.tolist())
        fsimSpec = -header.member("spectral_index")
        indx = np.argmin(np.abs(specbins - fsimSpec))
        
        # Get the energy spectrum of simulated events
        h1 = histo.member("hVEmc")[16][indx]
        counts, bins = h1.to_numpy()
        
        # Get the theta2 binning (simulated up to 5 degrees)
        rbw = 0.5
        theta2 = np.arange(0, 25 + rbw, rbw)
        theta2c = theta2[:-1] + 0.5*rbw

        # Uniformly scattered theta2
        simulated = np.stack([counts for i in range(len(theta2c))])
        simulated = simulated / int(theta2c.shape[0])
        
        fdata.close()
        return bins, counts, theta2, simulated


    def getMetaData(self, fname):
        pass

    def writeSimulatedData(self, fname, outname):
        # Get the simulated spectrum
        bins, counts, theta2, simulated = self.extractSimulatedSpectrum(fname)
        # Get the event-wise data
        self.loadData(fname)

    
        table = Table(
            self.data_dict,
        )
        
        phdu = fits.PrimaryHDU()
        tabData = fits.BinTableHDU(Table(self.data_dict))
        tabEBin = fits.BinTableHDU(Table({"ebin" : bins}))
        tabThetaBin = fits.BinTableHDU(Table({"theta2" : theta2}))
        imgSim = fits.ImageHDU(data=simulated)

        hdul = fits.HDUList([phdu, tabData, tabEBin, tabThetaBin, imgSim])
        hdul[1].name = "MSCW"
        hdul[2].name = "EBIN"
        hdul[3].name = "THETA2"
        hdul[4].name = "SIMULATED"
        
        hdul.writeto(outname, overwrite = True)


    def writeFits(self, outname, ):
        
        table = Table(
            self.data_dict,
        )

        return table
