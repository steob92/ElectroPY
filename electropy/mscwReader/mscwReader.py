import numpy as np
from astropy.coordinates import SkyCoord
import uproot
import pandas as pd
from pyslalib import slalib
# Make it a little easier to know where this is coming from
from electropy.utils import VSkyCoordinatesUtility as VSK



from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator, interp1d


class mscwReader():

    def __init__(self, sim = False):
        self.simulation_data = sim
        if self.simulation_data:
            self.target = SkyCoord(0, 0, unit='deg', frame='icrs')
        # else:
            # print("Please select the target name for the data run, run setTarget(target='Crab')")
        # filename
        # self.object
        # self.data
        self.target = None

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

            if not self.target:
                try :
                    runpara = dataFile["evndispLog"]
                    # Loop over log and grab the RA/Dec from the DB details
                    for line in runpara.all_members["fLines"]:
                        if ("J2000" in line) and ("RA" in line):
                            target = line.split()
                            ra = target[1].split("=")[1]
                            dec = target[2].split("=")[1]
                            self.target = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='fk5')
                            break
                except Exception as e:
                    print ("Warning target not found, please use mscwReader.setTarget(target_name)")
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


        # For stage 1 files get everything from the file
        self.data_dict = {
                        "runNumber" : data["runNumber"][emask],
                        "EVENT_ID" : data["eventNumber"][emask],
                        "MJD" : data["MJD"][emask],
                        "Time" : data["Time"][emask],
                        "TargetElev" : data["TargetElev"][emask],
                        "TargetAz" : data["TargetAz"][emask],
                        "TargetDec" : data["TargetDec"][emask],
                        "TargetRA" : data["TargetRA"][emask],
                        "WobbleN" : data["WobbleN"][emask],
                        "WobbleE" : data["WobbleE"][emask],
                        "LTrig" : data["LTrig"][emask],
                        "NTrig" : data["NTrig"][emask],
                        "NImages" : data["NImages"][emask],
                        "ImgSel" : data["ImgSel"][emask],
                        "NTtype" : data["NTtype"][emask],
                        "img2_ang" : data["img2_ang"][emask],
                        "Ze" : data["Ze"][emask],
                        "Az" : data["Az"][emask],
                        "ra" : data["ra"][emask],
                        "dec" : data["dec"][emask],
                        "Xoff" : data["Xoff"][emask],
                        "Yoff" : data["Yoff"][emask],
                        "Xoff_derot" : data["Xoff_derot"][emask],
                        "Yoff_derot" : data["Yoff_derot"][emask],
                        "theta2" : data["theta2"][emask],
                        "XCore" : data["Xcore"][emask],
                        "YCore" : data["Ycore"][emask],
                        "MeanPedvar" : data["meanPedvar_Image"][emask],
                        "NMSCW" : data["NMSCW"][emask],
                        "MSCW" : data["MSCW"][emask],
                        "MSCL" : data["MSCL"][emask],
                        "MWR" : data["MWR"][emask],
                        "MLR" : data["MLR"][emask],
                        "ENERGY" : data["ErecS"][emask],
                        "EChi2S" : data["EChi2S"][emask],
                        "dES" : data["dES"][emask],
                        "EmissionHeight" : data["EmissionHeight"][emask],
                        "EmissionHeightChi2" : data["EmissionHeightChi2"][emask],
                        "NTelPairs" : data["NTelPairs"][emask],
                        "SizeSecondMax" : data["SizeSecondMax"][emask],
                        "Core" : np.sqrt(data["Xcore"][emask]**2 + data["Ycore"][emask]**2 ),
                        "TIME": np.zeros(len(data["Yoff_derot"][emask])), # required colnames
                        "timeOfDay": data["Time"][emask]  # Needs to be converted to MET in "TIME" above
                    }

        # Adding some telescope level parameters
        # Loss (fraction of shower outside of shower), size (2, the 2nd brightest pixels) and time gradient (the time gradient across the camera)
        for i in range(4):
            self.data_dict["loss_%d"%i] = data["loss"][:,i][emask]
            self.data_dict["tgrad_x_%d"%i] = data["tgrad_x"][:,i][emask]
            self.data_dict["size2_%d"%i] = data["size2"][:,i][emask]
            
        # Adding MC entries
        if self.simulation_data:
            self.data_dict["MCe0"] =  data["MCe0"][emask]
            self.data_dict["El"] =  90-data["Ze"][emask]
            self.data_dict["Az"] =  data["Az"][emask]
            self.data_dict["RA"] =  np.zeros(len(data["Yoff_derot"][emask])) # we dont care about ra and dec
            self.data_dict["DEC"] =  np.zeros(len(data["Yoff_derot"][emask])) 
            self.data_dict["TIME"] =  np.zeros(len(data["Yoff_derot"][emask])) # required colnames 
        


    # Name change needed Here we're just getting the RA/Dec
    def calculateRADec(self):

        if not self.simulation_data:
            # convert Xoff_derot, Yoff_derot from current epoch into J2000 epoch
            derot = np.array(
                list(
                    map(
                        VSK.convert_derotatedCoordinates_to_J2000,
                        VSK.getUTC(self.data_dict['MJD'], self.data_dict["timeOfDay"]),
                        np.repeat(self.target.ra.deg, len(self.data_dict["Xoff_derot"])),
                        np.repeat(self.target.dec.deg, len(self.data_dict["Xoff_derot"])), 
                        self.data_dict['Xoff_derot'],
                        self.data_dict['Yoff_derot']
                    )
                )
            )

            


            self.data_dict['Xoff_derot'] = np.array(derot[:,0])
            self.data_dict['Yoff_derot'] = np.array(derot[:,1])

            # take Xderot and Yderot and convert it into RA and DEC for each event

            radec = list(
                map(
                    slalib.sla_dtp2s, 
                    np.deg2rad(self.data_dict['Xoff_derot']),
                    np.deg2rad(self.data_dict['Yoff_derot']),
                    np.repeat(np.deg2rad(self.pointing.ra.deg), len(self.data_dict["Xoff_derot"])),
                    np.repeat(np.deg2rad(self.pointing.dec.deg), len(self.data_dict["Xoff_derot"])),
                )
            )

            self.data_dict['RA'] = np.array(np.rad2deg([radec[0] for radec in radec]))
            self.data_dict['DEC'] = np.array(np.rad2deg([radec[1] for radec in radec]))

            # convert RA and DEC of each event into elevation and azimuth


            elaz = list(
                map(
                    VSK.getHorizontalCoordinates,
                    self.data_dict['MJD'],
                    self.data_dict['timeOfDay'],
                    self.data_dict['DEC'],
                    self.data_dict['RA']
                )
            )


            self.data_dict['El'] = np.array([elaz[0] for elaz in elaz])
            self.data_dict['Az'] = np.array([elaz[1] for elaz in elaz])



    def dataToPD(self):

        df = pd.DataFrame(self.data_dict)

        required_col = ['runNumber', 'EVENT_ID', 'timeOfDay', 'MJD', 'ENERGY',
                        'dES','EChi2S','SizeSecondMax', 'XCore', 'YCore', 'Core', 'Xoff_derot', 'Yoff_derot', 'NImages',
                        'ImgSel', 'MeanPedvar', 'MSCW', 'MSCL', 'RA',
                        'DEC', 'Az', 'El', 'EmissionHeight', 'Xoff', 'Yoff', 'TIME']


        # this is DL3 output file
        DL3data = df[required_col]
        DL3data.rename(columns = {'Xderot':'Xoff_derot', 'Yderot':'Yoff_derot'}, inplace = True)
        return DL3data
        

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
