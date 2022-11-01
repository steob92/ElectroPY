#import sys
#sys.path.append('../utils/')

from utils.VSkyCoordinatesUtility import *  
import numpy as np
from astropy.coordinates import SkyCoord
import uproot
import pandas as pd
from pyslalib import slalib
#from VSkyCoordinatesUtility import *


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
        #self.targetCamX = (self.target.ra.deg - np.rad2deg(self.tel_ra)) * np.cos(self.tel_dec)
        #self.targetCamY = self.target.dec.deg - np.rad2deg(self.tel_dec)




    def readFile(self, filename):

        self.filename = filename
        dataFile =  uproot.open(self.filename)

        data = dataFile["data"].arrays(library="numpy")

        # Real data
        try :

            pointingReduced = dataFile["pointingDataReduced"].arrays(library="numpy")
            self.tel_ra = np.rad2deg(np.mean(pointingReduced["TelRAJ2000"]))
            self.tel_dec = np.rad2deg(np.mean(pointingReduced["TelDecJ2000"]))
            self.pointing = SkyCoord(self.tel_ra, self.tel_dec, unit='deg', frame='icrs')


            #self.tel_ra = np.median(pointingReduced["TelRAJ2000"])
            #self.tel_dec = np.median(pointingReduced["TelDecJ2000"])
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
        # Mask out failed events
        emask = data["ErecS"] >0
        emask *= data["theta2"] <= 2. 
        # Store data to dictionary

        self.data_dict = {
            "runNumber": data["runNumber"][emask],
            "eventNumber" : data["eventNumber"][emask],
            "timeOfDay": data["Time"][emask],
            "MJD": data["MJD"][emask],
            "Erec" : data["ErecS"][emask],
            "Erec_Err" : data["dES"][emask],
            "NImages" : data["NImages"][emask],
            "ImgSel": data["ImgSel"][emask],
            "MeanPedvar": data["meanPedvar_Image"][emask],
            "MSCW" : data["MSCW"][emask],
            "MSCL" : data["MSCL"][emask],
            "EmissionHeight" : data["EmissionHeight"][emask],
            "Xoff_derot": data["Xoff_derot"][emask],
            "Yoff_derot": data["Yoff_derot"][emask],
            "EChi2S" : data["EChi2S"][emask],
            "SizeSecondMax" : data["SizeSecondMax"][emask],
            "XCore" : data["Xcore"][emask],
            "YCore" : data["Ycore"][emask],
            "Xoff" : data["Xoff"][emask],
            "Yoff": data["Yoff"][emask]
        }


    def convert_to_dl3_format(self):
        df = pd.DataFrame(self.data_dict)

        # convert Xoff_derot, Yoff_derot from current epoch into J2000 epoch
        derot = np.array(list(map(convert_derotatedCoordinates_to_J2000, getUTC(df.MJD, df.timeOfDay),np.repeat(self.target.ra.deg, len(df)),np.repeat(self.target.dec.deg, len(df)), df['Xoff_derot'], df['Yoff_derot'])))


        df['Xderot'] = derot[:,0]
        df['Yderot'] = derot[:,1]

        # take Xderot and Yderot and convert it into RA and DEC for each event

        radec = list(map(slalib.sla_dtp2s, np.deg2rad(df.Xderot), np.deg2rad(df.Yderot),
                         np.repeat(np.deg2rad(self.pointing.ra.deg), len(df)),
                         np.repeat(np.deg2rad(self.pointing.dec.deg), len(df)),
                         ))

        df['RA'] = np.rad2deg([radec[0] for radec in radec])
        df['DEC'] = np.rad2deg([radec[1] for radec in radec])

        # convert RA and DEC of each event into elevation and azimuth


        elaz = list(map(getHorizontalCoordinates, df.MJD, df.timeOfDay, df.DEC, df.RA))


        df['El'] = [elaz[0] for elaz in elaz]
        df['Az'] = [elaz[1] for elaz in elaz]


        # These are all the required coulmns we need for DL3 style output formatting
        
        required_col = ['runNumber', 'eventNumber', 'timeOfDay', 'MJD', 'Erec',
                'Erec_Err', 'XCore', 'YCore', 'Xderot', 'Yderot', 'NImages',
                'ImgSel', 'MeanPedvar', 'MSCW', 'MSCL', 'RA',
                'DEC', 'Az', 'El', 'EmissionHeight', 'Xoff', 'Yoff']

        # this is DL3 output file
        self.DL3data = df[required_col]



        

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
