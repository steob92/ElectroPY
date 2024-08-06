import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import circmean
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
        self.simulationData = sim
        if self.simulationData:
            self.target = SkyCoord(0, 0, unit='deg', frame='icrs')
        # else:
            # print("Please select the target name for the data run, run setTarget(target='Crab')")
        # filename
        # self.object
        # self.data
        self.target = None
        self.metaData = {}

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

            dt, _ = dataFile["deadTimeHistograms/hScalarDeadTimeFraction_on"].to_numpy()
            self.metaData["DeadTime"] = np.median(dt)
            self.metaData["RA_PNT"] = self.tel_ra
            self.metaData["DEC_PNT"] = self.tel_dec

            if not self.target:
                try :
                    runpara = dataFile["evndispLog"]
                    # Loop over log and grab the RA/Dec from the DB details
                    for line in runpara.all_members["fLines"]:
                        if "Duration" in line:
                            self.metaData["Duration"] = float(line.split()[5])

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
            # Extract some simulation meta data
            mcHead = dataFile["MC_runheader"].members
            self.metaData["ELMIN"] = (np.rad2deg(mcHead["alt_range"][0]), "Minimum Elevation Angle")
            self.metaData["ELMAX"] = (np.rad2deg(mcHead["alt_range"][1]), "Maximum Elevation Angle")
            self.metaData["EL"] = (np.mean(np.rad2deg(mcHead["alt_range"])), "Mean Elevation Angle")
            self.metaData["AZMIN"] = (np.rad2deg(mcHead["az_range"][0]), "Minimum Azimuth Angle")
            self.metaData["AZMAX"] = (np.rad2deg(mcHead["az_range"][1]), "Maximum Azimuth Angle")
            self.metaData["PRIMARY"] = (mcHead["primary_id"], "Primary Particle ID") 
            self.metaData["EMIN"] = (mcHead["E_range"][0], "Minimum Energy")
            self.metaData["EMAX"] = (mcHead["E_range"][1], "Maximum Energy")
            self.metaData["INDEX"] = (mcHead["spectral_index"], "Spectral Index")
            

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
        self.dataDict = {
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
                        "Theta" : np.sqrt(data["Xoff"][emask]**2 + data["Xoff"][emask]**2),
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
            self.dataDict["loss_%d"%i] = data["loss"][:,i][emask]
            self.dataDict["tgrad_x_%d"%i] = data["tgrad_x"][:,i][emask]
            self.dataDict["size2_%d"%i] = data["size2"][:,i][emask]
            
        # Adding MC entries
        if self.simulationData:
            self.dataDict["ENERGY_MC"] =  data["MCe0"][emask]
            self.dataDict["MCxoff"] =  data["MCxoff"][emask]
            self.dataDict["MCyoff"] =  data["MCyoff"][emask]
            self.dataDict["MCTheta"] =  np.sqrt(data["MCxoff"][emask]**2 + data["MCyoff"][emask]**2)
            self.dataDict["El"] =  90-data["Ze"][emask]
            self.dataDict["Az"] =  data["Az"][emask]
            self.dataDict["RA"] =  np.zeros(len(data["Yoff_derot"][emask])) # we dont care about ra and dec
            self.dataDict["DEC"] =  np.zeros(len(data["Yoff_derot"][emask])) 
            self.dataDict["TIME"] =  np.zeros(len(data["Yoff_derot"][emask])) # required colnames 
        


    # Name change needed Here we're just getting the RA/Dec
    def calculateRADec(self):

        if not self.simulationData:
            # convert Xoff_derot, Yoff_derot from current epoch into J2000 epoch
            derot = np.array(
                list(
                    map(
                        VSK.convert_derotatedCoordinates_to_J2000,
                        VSK.getUTC(self.dataDict['MJD'], self.dataDict["timeOfDay"]),
                        np.repeat(self.target.ra.deg, len(self.dataDict["Xoff_derot"])),
                        np.repeat(self.target.dec.deg, len(self.dataDict["Xoff_derot"])), 
                        self.dataDict['Xoff_derot'],
                        self.dataDict['Yoff_derot']
                    )
                )
            )

            


            self.dataDict['Xoff_derot'] = np.array(derot[:,0])
            self.dataDict['Yoff_derot'] = np.array(derot[:,1])

            # take Xderot and Yderot and convert it into RA and DEC for each event

            radec = list(
                map(
                    slalib.sla_dtp2s, 
                    np.deg2rad(self.dataDict['Xoff_derot']),
                    np.deg2rad(self.dataDict['Yoff_derot']),
                    np.repeat(np.deg2rad(self.pointing.ra.deg), len(self.dataDict["Xoff_derot"])),
                    np.repeat(np.deg2rad(self.pointing.dec.deg), len(self.dataDict["Xoff_derot"])),
                )
            )

            self.dataDict['RA'] = np.array(np.rad2deg([radec[0] for radec in radec]))
            self.dataDict['DEC'] = np.array(np.rad2deg([radec[1] for radec in radec]))

            # convert RA and DEC of each event into elevation and azimuth


            elaz = list(
                map(
                    VSK.getHorizontalCoordinates,
                    self.dataDict['MJD'],
                    self.dataDict['timeOfDay'],
                    self.dataDict['DEC'],
                    self.dataDict['RA']
                )
            )


            self.dataDict['El'] = np.array([elaz[0] for elaz in elaz])
            self.dataDict['Az'] = np.array([elaz[1] for elaz in elaz])

            # For interpolating IRFs over the run
            # ToDo: Look at event wise interpolation
            nslices = 10
            nsamp = len(self.dataDict['El']) // (nslices + 1)
            self.metaData['El'] = np.zeros(nslices)
            self.metaData['Az'] = np.zeros(nslices)
            self.metaData['MeanPedvar'] = np.zeros(nslices)
            
            for i in range(nslices):
                # Using circular mean for angles
                self.metaData['El'][i] = circmean(self.dataDict["El"][i*nsamp:(i+1)*nsamp] * u.deg).value
                self.metaData['Az'][i] = circmean(self.dataDict["Az"][i*nsamp:(i+1)*nsamp] * u.deg).value
                # Standard mean for pedvar
                self.metaData['MeanPedvar'][i] = np.mean(self.dataDict["MeanPedvar"][i*nsamp:(i+1)*nsamp])


    def dataToPD(self):

        df = pd.DataFrame(self.dataDict)

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
        rbw = 0.25
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

    def writeSimulatedData(self, fname, outname, outdir = "./"):
        # Get the simulated spectrum
        bins, counts, theta2, simulated = self.extractSimulatedSpectrum(fname)
        # Get the event-wise data
        self.loadData(fname)

    
        table = Table(
            self.dataDict,
        )
        
        phdu = fits.PrimaryHDU()
        tabData = fits.BinTableHDU(Table(self.dataDict))
        tabEBin = fits.BinTableHDU(Table({"ebin" : bins}))
        tabThetaBin = fits.BinTableHDU(Table({"theta2" : theta2}))
        imgSim = fits.ImageHDU(data=simulated)

        hdul = fits.HDUList([phdu, tabData, tabEBin, tabThetaBin, imgSim])
        hdul[1].name = "MSCW"
        hdul[2].name = "EBIN"
        hdul[3].name = "THETA2"
        hdul[4].name = "SIMULATED"

        for k in self.metaData.keys():
            hdul[0].header[k] = self.metaData[k]
        
        hdul.writeto(outdir + "/" + outname, overwrite = True)

    def writeData(self, fname, outname, outdir = "./"):

        # Get the event-wise data
        self.loadData(fname)
        self.calculateRADec()

        phdu = fits.PrimaryHDU()
        tabData = fits.BinTableHDU(Table(self.dataDict))

        tabs = [phdu, tabData]
        hdul = fits.HDUList(tabs)

        # Add some useful headers
        hdul[0].header["RUNNUM"] = (self.dataDict["runNumber"][0], "VERITAS Run Number") 
        hdul[0].header["MJD"] = (np.mean(self.dataDict["MJD"]), "MJD of run") 
        hdul[0].header["PEDVAR"] = (np.mean(self.metaData["MeanPedvar"]), "Mean PedVar")
        hdul[0].header["EL"] = (circmean(self.metaData['El']*u.deg).value, "Mean Elevation")
        hdul[0].header["Az"] = (circmean(self.metaData['Az']*u.deg).value, "Mean Azimuth")
        hdul[0].header["DEADTIME"] = (self.metaData["DeadTime"], "Fractional Deadtime")
        hdul[0].header["DURATION"] = (self.metaData["Duration"], "Duration (s)")
        hdul[0].header["RA_PNT"] = (self.metaData["RA_PNT"], "Pointing Right Ascension (deg)")
        hdul[0].header["DEC_PNT"] = (self.metaData["DEC_PNT"], "Pointing Declination (deg)")
        
        hdul[1].name = "MSCW"
        hdul.writeto(outdir + "/" + outname, overwrite = True)


    def writeFits(self, fname, outname, outdir = "./"):
        
        if (self.simulationData):
            self.writeSimulatedData(fname, outname, outdir)
        else:
            self.writeData(fname, outname, outdir)
