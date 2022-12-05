<<<<<<< HEAD
from utils.VSkyCoordinatesUtility import *  
=======
# import sys
# sys.path.append('../utils/')
>>>>>>> 3b63424 (Formatting changes for readability and fixing paths)
import numpy as np
from astropy.coordinates import SkyCoord
import uproot
import pandas as pd
from pyslalib import slalib
<<<<<<< HEAD
=======
# from VSkyCoordinatesUtility import *
# from electropy.utils.VSkyCoordinatesUtility import *
# Make it a little easier to know where this is coming from
from electropy.utils import VSkyCoordinatesUtility as VSK



>>>>>>> 3b63424 (Formatting changes for readability and fixing paths)
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator, interp1d

# Needed?
# from gammapy.maps import WcsNDMap, WcsGeom, MapAxis, RegionGeom, Map


class mscwReader():

    def __init__(self, sim):
        self.simulation_data = sim
        if self.simulation_data:
            self.target = SkyCoord(0, 0, unit='deg', frame='icrs')
        else:
            print("Please select the target name for the data run, run setTarget(target='Crab')")
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
            self.pointing = SkyCoord(self.tel_ra, self.tel_dec, unit='deg', frame='icrs')

        # Close the file to help with memory
        dataFile.close()
        return data


    def loadData(self, filename):

        # Obtain data
        data = self.readFile(filename)
        # Mask out failed events
        emask = data["ErecS"] >0
#        emask *= data["theta2"] <= 2. 
        # Store data to dictionary
 #       VTS_REFERENCE_MJD = 53402.0
        if self.simulation_data:
            self.data_dict = {
                "runNumber": data["runNumber"][emask],
                "EVENT_ID" : data["eventNumber"][emask],
                "timeOfDay": data["Time"][emask],
                "MJD": data["MJD"][emask],
                "ENERGY" : data["ErecS"][emask],
                "dES" : data["dES"][emask],
                "MCe0": data["MCe0"][emask],
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
                "Core" : np.sqrt(data["Xcore"][emask]**2 + data["Ycore"][emask]**2),
                "Xoff" : data["Xoff"][emask],
                "Yoff": data["Yoff"][emask],
                "El" : 90-data["Ze"][emask],
                "Az" : data["Az"][emask],
                "RA": np.zeros(len(data["Yoff_derot"][emask])), # we dont care about ra and dec
                "DEC": np.zeros(len(data["Yoff_derot"][emask])), 
                "TIME": np.zeros(len(data["Yoff_derot"][emask])) # required colnames 
            }

        else:
            self.data_dict = {
                "runNumber": data["runNumber"][emask],
                "EVENT_ID" : data["eventNumber"][emask],
                "timeOfDay": data["Time"][emask],
                "MJD": data["MJD"][emask],
                "ENERGY" : data["ErecS"][emask],
                "dES" : data["dES"][emask],
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
                "Core" : np.sqrt(data["Xcore"][emask]**2 + data["Ycore"][emask]**2),
                "Xoff" : data["Xoff"][emask],
                "Yoff": data["Yoff"][emask],
                "TIME": np.zeros(len(data["Yoff_derot"][emask])) # required colnames
            }


<<<<<<< HEAD

    def convertToDL3Format(self):

=======
    # Name change needed Here we're just getting the RA/Dec
    def convertToDL3Format(self):
        df = pd.DataFrame(self.data_dict)

        # convert Xoff_derot, Yoff_derot from current epoch into J2000 epoch
        derot = np.array(
            list(
                map(
                    VSK.convert_derotatedCoordinates_to_J2000, 
                    VSK.getUTC(df.MJD, df.timeOfDay),
                    np.repeat(self.target.ra.deg, len(df)), 
                    np.repeat(self.target.dec.deg, len(df)), 
                    df['Xoff_derot'], df['Yoff_derot']
                    )
                )
            )


        df['Xderot'] = derot[:,0]
        df['Yderot'] = derot[:,1]

        # take Xderot and Yderot and convert it into RA and DEC for each event

        radec = list(
            map(
                slalib.sla_dtp2s, 
                np.deg2rad(df.Xderot), 
                np.deg2rad(df.Yderot),
                np.repeat(np.deg2rad(self.pointing.ra.deg), len(df)),
                np.repeat(np.deg2rad(self.pointing.dec.deg), len(df)),
                )
            )

        df['RA'] = np.rad2deg([radec[0] for radec in radec])
        df['DEC'] = np.rad2deg([radec[1] for radec in radec])

        # convert RA and DEC of each event into elevation and azimuth


        elaz = list(
            map(
                VSK.getHorizontalCoordinates, 
                df.MJD, 
                df.timeOfDay, 
                df.DEC, 
                df.RA
                )
            )


        df['El'] = [elaz[0] for elaz in elaz]
        df['Az'] = [elaz[1] for elaz in elaz]


        # These are all the required coulmns we need for DL3 style output formatting
>>>>>>> 3b63424 (Formatting changes for readability and fixing paths)
        
        #df = pd.DataFrame(self.data_dict)

        if self.simulation_data:
            pass

        else:
            # convert Xoff_derot, Yoff_derot from current epoch into J2000 epoch
            derot = np.array(list(map(convert_derotatedCoordinates_to_J2000, getUTC(self.data_dict['MJD'], self.data_dict["timeOfDay"]),
                          np.repeat(self.target.ra.deg, len(self.data_dict["Xoff_derot"])),
                          np.repeat(self.target.dec.deg, len(self.data_dict["Xoff_derot"])), 
                                    self.data_dict['Xoff_derot'],self.data_dict['Yoff_derot'])))

            


            self.data_dict['Xoff_derot'] = np.array(derot[:,0])
            self.data_dict['Yoff_derot'] = np.array(derot[:,1])

            # take Xderot and Yderot and convert it into RA and DEC for each event

            radec = list(map(slalib.sla_dtp2s, np.deg2rad(self.data_dict['Xoff_derot']), np.deg2rad(self.data_dict['Yoff_derot']),
                             np.repeat(np.deg2rad(self.pointing.ra.deg), len(self.data_dict["Xoff_derot"])),
                             np.repeat(np.deg2rad(self.pointing.dec.deg), len(self.data_dict["Xoff_derot"])),
                             ))

            self.data_dict['RA'] = np.array(np.rad2deg([radec[0] for radec in radec]))
            self.data_dict['DEC'] = np.array(np.rad2deg([radec[1] for radec in radec]))

            # convert RA and DEC of each event into elevation and azimuth


            elaz = list(map(getHorizontalCoordinates, self.data_dict['MJD'], self.data_dict['timeOfDay'], self.data_dict['DEC'], self.data_dict['RA']))


            self.data_dict['El'] = np.array([elaz[0] for elaz in elaz])
            self.data_dict['Az'] = np.array([elaz[1] for elaz in elaz])


            # These are all the required coulmns we need for DL3 style output formatting
  
        



        

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


    # Dummy to eventually write meta (run id, NSB, wobble, etc to a header)
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
