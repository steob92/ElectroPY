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


class MscwReader():

    def __init__(self, sim = False):
        self.simulation_data = sim
        if self.simulation_data:
            self.target = SkyCoord(0, 0, unit='deg', frame='icrs')
        # else:
            # print("Please select the target name for the data run, run set_target(target='Crab')")
        # filename
        # self.object
        # self.data
        self.target = None
        self.meta_data = {}

    def set_target(self, target):
        
        self.target = SkyCoord.from_name(target)
        #self.targetCamX = (self.target.ra.deg - np.rad2deg(self.tel_ra)) * np.cos(self.tel_dec)
        #self.targetCamY = self.target.dec.deg - np.rad2deg(self.tel_dec)




    def read_file(self, filename):

        self.filename = filename
        data_file =  uproot.open(self.filename)

        data = data_file["data"].arrays(library="numpy")

        # Real data
        try :

            pointing_reduced = data_file["pointingDataReduced"].arrays(library="numpy")
            self.tel_ra = np.rad2deg(np.mean(pointing_reduced["TelRAJ2000"]))
            self.tel_dec = np.rad2deg(np.mean(pointing_reduced["TelDecJ2000"]))
            self.pointing = SkyCoord(self.tel_ra, self.tel_dec, unit='deg', frame='icrs')

            dt, _ = data_file["deadTimeHistograms/hScalarDeadTimeFraction_on"].to_numpy()
            self.meta_data["DeadTime"] = np.median(dt)
            self.meta_data["RA_PNT"] = self.tel_ra
            self.meta_data["DEC_PNT"] = self.tel_dec

            if not self.target:
                try :
                    runpara = data_file["evndispLog"]
                    # Loop over log and grab the RA/Dec from the DB details
                    for line in runpara.all_members["fLines"]:
                        if "Duration" in line:
                            self.meta_data["Duration"] = float(line.split()[5])

                        if ("J2000" in line) and ("RA" in line):
                            target = line.split()
                            ra = target[1].split("=")[1]
                            dec = target[2].split("=")[1]
                            self.target = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='fk5')
                            break
                except Exception as e:
                    print ("Warning target not found, please use MscwReader.set_target(target_name)")
            #self.tel_ra = np.median(pointing_reduced["TelRAJ2000"])
            #self.tel_dec = np.median(pointing_reduced["TelDecJ2000"])

        # Simulated data
        except Exception as e:
            # Give default RA/Dec
            self.tel_ra = 0
            self.tel_dec = 0
            self.pointing = SkyCoord(self.tel_ra, self.tel_dec, unit='deg', frame='icrs')
            # Extract some simulation meta data
            mc_header = data_file["MC_runheader"].members
            self.meta_data["ELMIN"] = (np.rad2deg(mc_header["alt_range"][0]), "Minimum Elevation Angle")
            self.meta_data["ELMAX"] = (np.rad2deg(mc_header["alt_range"][1]), "Maximum Elevation Angle")
            self.meta_data["EL"] = (np.mean(np.rad2deg(mc_header["alt_range"])), "Mean Elevation Angle")
            self.meta_data["AZMIN"] = (np.rad2deg(mc_header["az_range"][0]), "Minimum Azimuth Angle")
            self.meta_data["AZMAX"] = (np.rad2deg(mc_header["az_range"][1]), "Maximum Azimuth Angle")
            self.meta_data["PRIMARY"] = (mc_header["primary_id"], "Primary Particle ID") 
            self.meta_data["EMIN"] = (mc_header["E_range"][0], "Minimum Energy")
            self.meta_data["EMAX"] = (mc_header["E_range"][1], "Maximum Energy")
            self.meta_data["INDEX"] = (mc_header["spectral_index"], "Spectral Index")
            

        # Close the file to help with memory
        data_file.close()
        return data


    def load_data(self, filename):

        # Obtain data
        data = self.read_file(filename)
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
            self.data_dict["loss_%d"%i] = data["loss"][:,i][emask]
            self.data_dict["tgrad_x_%d"%i] = data["tgrad_x"][:,i][emask]
            self.data_dict["size2_%d"%i] = data["size2"][:,i][emask]
            
        # Adding MC entries
        if self.simulation_data:
            self.data_dict["ENERGY_MC"] =  data["MCe0"][emask]
            self.data_dict["MCxoff"] =  data["MCxoff"][emask]
            self.data_dict["MCyoff"] =  data["MCyoff"][emask]
            self.data_dict["MCTheta"] =  np.sqrt(data["MCxoff"][emask]**2 + data["MCyoff"][emask]**2)
            self.data_dict["El"] =  90-data["Ze"][emask]
            self.data_dict["Az"] =  data["Az"][emask]
            self.data_dict["RA"] =  np.zeros(len(data["Yoff_derot"][emask])) # we dont care about ra and dec
            self.data_dict["DEC"] =  np.zeros(len(data["Yoff_derot"][emask])) 
            self.data_dict["TIME"] =  np.zeros(len(data["Yoff_derot"][emask])) # required colnames 
        


    # Name change needed Here we're just getting the RA/Dec
    def calculate_ra_dec(self):

        if not self.simulation_data:
            # convert Xoff_derot, Yoff_derot from current epoch into J2000 epoch
            derot = np.array(
                list(
                    map(
                        VSK.convert_derotated_coordinates_to_J2000,
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
                    VSK.get_horizontal_coordinates,
                    self.data_dict['MJD'],
                    self.data_dict['timeOfDay'],
                    self.data_dict['DEC'],
                    self.data_dict['RA']
                )
            )


            self.data_dict['El'] = np.array([elaz[0] for elaz in elaz])
            self.data_dict['Az'] = np.array([elaz[1] for elaz in elaz])

            # For interpolating IRFs over the run
            # ToDo: Look at event wise interpolation
            nslices = 10
            nsamp = len(self.data_dict['El']) // (nslices + 1)
            self.meta_data['El'] = np.zeros(nslices)
            self.meta_data['Az'] = np.zeros(nslices)
            self.meta_data['MeanPedvar'] = np.zeros(nslices)
            
            for i in range(nslices):
                # Using circular mean for angles
                self.meta_data['El'][i] = circmean(self.data_dict["El"][i*nsamp:(i+1)*nsamp] * u.deg).value
                self.meta_data['Az'][i] = circmean(self.data_dict["Az"][i*nsamp:(i+1)*nsamp] * u.deg).value
                # Standard mean for pedvar
                self.meta_data['MeanPedvar'][i] = np.mean(self.data_dict["MeanPedvar"][i*nsamp:(i+1)*nsamp])


    def to_dataframe(self):

        df = pd.DataFrame(self.data_dict)

        required_col = ['runNumber', 'EVENT_ID', 'timeOfDay', 'MJD', 'ENERGY',
                        'dES','EChi2S','SizeSecondMax', 'XCore', 'YCore', 'Core', 'Xoff_derot', 'Yoff_derot', 'NImages',
                        'ImgSel', 'MeanPedvar', 'MSCW', 'MSCL', 'RA',
                        'DEC', 'Az', 'El', 'EmissionHeight', 'Xoff', 'Yoff', 'TIME']


        # this is DL3 output file
        dls_data = df[required_col]
        dls_data.rename(columns = {'Xderot':'Xoff_derot', 'Yderot':'Yoff_derot'}, inplace = True)
        return dls_data
        

    def extract_simulated_spectrum(self,fname):
    
        fdata = uproot.open(fname)
        
        # Open up the histogram and the headers
        histo = fdata["MChistos"]
        header = fdata["MC_runheader"]
        
        # Find the simulated spectral index
        spectral_index = histo.member("fVSpectralIndex")
        specbins = np.array(spectral_index.tolist())
        sim_spectral_index = -header.member("spectral_index")
        indx = np.argmin(np.abs(specbins - sim_spectral_index))
        
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
    def get_meta_data(self, fname):
        pass

    def write_simulated_data(self, fname, outname, outdir = "./"):
        # Get the simulated spectrum
        bins, counts, theta2, simulated = self.extract_simulated_spectrum(fname)
        # Get the event-wise data
        self.load_data(fname)

    
        table = Table(
            self.data_dict,
        )
        
        phdu = fits.PrimaryHDU()
        tab_data = fits.BinTableHDU(Table(self.data_dict))
        tab_e_Bin = fits.BinTableHDU(Table({"ebin" : bins}))
        tab_theta_bin = fits.BinTableHDU(Table({"theta2" : theta2}))
        img_sim = fits.ImageHDU(data=simulated)

        hdul = fits.HDUList([phdu, tab_data, tab_e_Bin, tab_theta_bin, img_sim])
        hdul[1].name = "MSCW"
        hdul[2].name = "EBIN"
        hdul[3].name = "THETA2"
        hdul[4].name = "SIMULATED"

        for k in self.meta_data.keys():
            hdul[0].header[k] = self.meta_data[k]
        
        hdul.writeto(outdir + "/" + outname, overwrite = True)

    def write_data(self, fname, outname, outdir = "./"):

        # Get the event-wise data
        self.load_data(fname)
        self.calculate_ra_dec()

        phdu = fits.PrimaryHDU()
        tab_data = fits.BinTableHDU(Table(self.data_dict))

        tabs = [phdu, tab_data]
        hdul = fits.HDUList(tabs)

        # Add some useful headers
        hdul[0].header["RUNNUM"] = (self.data_dict["runNumber"][0], "VERITAS Run Number") 
        hdul[0].header["MJD"] = (np.mean(self.data_dict["MJD"]), "MJD of run") 
        hdul[0].header["PEDVAR"] = (np.mean(self.meta_data["MeanPedvar"]), "Mean PedVar")
        hdul[0].header["EL"] = (circmean(self.meta_data['El']*u.deg).value, "Mean Elevation")
        hdul[0].header["Az"] = (circmean(self.meta_data['Az']*u.deg).value, "Mean Azimuth")
        hdul[0].header["DEADTIME"] = (self.meta_data["DeadTime"], "Fractional Deadtime")
        hdul[0].header["DURATION"] = (self.meta_data["Duration"], "Duration (s)")
        hdul[0].header["RA_PNT"] = (self.meta_data["RA_PNT"], "Pointing Right Ascension (deg)")
        hdul[0].header["DEC_PNT"] = (self.meta_data["DEC_PNT"], "Pointing Declination (deg)")
        
        hdul[1].name = "MSCW"
        hdul.writeto(outdir + "/" + outname, overwrite = True)


    def write_fits(self, fname, outname, outdir = "./"):
        
        if (self.simulation_data):
            self.write_simulated_data(fname, outname, outdir)
        else:
            self.write_data(fname, outname, outdir)
