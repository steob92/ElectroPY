import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.table import Table
from astropy.io import fits

from joblib import dump, load
import yaml

from gammapy.irf.psf import PSF3D
from gammapy.irf import EnergyDispersion2D, EffectiveAreaTable2D


try:
    import cmasher as cms
    cmap = cms.ghostlight
except ImportError:
    cmap = "plasma"

class IRFMaker():

    def __init__(self):
        # Defaulting to 0 
        # 0 - electron
        # 1 - Proton
        # 2 - Helium
        self.eventClass = 0

        # Cut on the probability
        self.prob_cut = 0.5
        
        # Scaler, classifier and features of interest
        # self.scaler = None
        # self.classifier = None
        # self.featuresClass = None
        # self.featuresEnergy = None
        self.log_feat = None


        # For the effective areas
        self._rThrow = 750 # m
        self._aThrow = np.pi * self._rThrow**2   # m^2  

        # meta data for output
        self.metaData = {}

    def loadConfig(self, fname):
        with open(fname, "r") as f:
            self.config = yaml.safe_load(f)

        # Load in the classifier and scalers
        self.loadEnergyScaler()
        self.loadClassScaler()
        self.loadEnergyEstimator()
        self.loadClassifier()
        self.log_feat = self.config["LogFeat"]
        

    def loadEnergyScaler(self, fname = None):
        # Load in the energy scaler and extract the features
        if fname is None:
            self.scalerEnergy = load(self.config["EnergyScaler"])
            self.featuresEnergy = self.config["FeaturesEnergy"]
        else:
            self.scalerEnergy = load(fname)


    def loadClassScaler(self, fname = None):
        # Load in the classifier scaler and extract the features
        if fname is None:
            self.scalerClass = load(self.config["ClassifierScaler"])
            self.featuresClass = self.config["FeaturesClassifier"]
        else:
            self.scalerClass = load(fname)

    def loadEnergyEstimator(self, fname = None):
        # Load in the energy estimator
        if fname is None:
            self.energyEstimator = load(self.config["Energy"])
        else:
            self.energyEstimator = load(fname)
        

    def loadClassifier(self, fname = None):
        # Load in the Classifier
        if fname is None:
            self.classifier = load(self.config["Classifier"])
        else:
            self.classifier = load(fname)

    # def loadFeatures(self, fname):
    #     feat = load(fname)
    #     self.featuresClass = feat["FeaturesClass"]
    #     if "FeaturesEnergy" in feat:
    #         self.featuresEnergy = feat["FeaturesEnergy"]
    #     self.log_feat = feat["Log Features"]


    def readData(self, fname):

        # Open the fits files
        with fits.open(fname) as hdul:

            self.metaData = hdul[0].header
            # Convert event-wise data to a dataframe
            df = Table.read(hdul[1]).to_pandas()

            # Get the energy and theta2 binning
            ebin = Table.read(hdul[2])["ebin"]
            theta2bin = Table.read(hdul[3])["theta2"]
            ebinc = ebin[:-1] + 0.5*(ebin[1:] - ebin[:-1])
            theta2binc = theta2bin[:-1] + 0.5*(theta2bin[1:] - theta2bin[:-1])

            # Simulated spectrum
            spect = hdul[4].data

            # Convert to log10 of feature
            for feat in self.log_feat:
                df[feat] = np.log10(df[feat])

            # Drop nan's (they'll be poorly reconstructed events)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # Get the location in the FoV
            # This shouldn't be derotated. Assuming the derotation is a small effect for now
            df["Theta2"] = df["Xoff_derot"]**2 + df["Yoff_derot"]**2

        self.eventData =  {
            "data" : df, 
            "energy_binning" : ebin, 
            "theta2_binning" : theta2bin, 
            "simulated_spectrum" :spect,
            "energy_binning_cen": ebinc,
            "theta2_binning_cen": theta2binc
        }

    def estimateEnergy(self):
        x = self.eventData["data"][self.featuresEnergy].values
        x_scaled = self.scalerEnergy.transform(x)
        prediction = self.energyEstimator.predict(x_scaled)

        self.eventData["data"]["ENERGY_LUT"] = np.log10(self.eventData["data"]["ENERGY"])
        self.eventData["data"]["ENERGY_RF"] = prediction
        self.eventData["data"]["ENERGY"] = self.eventData["data"]["ENERGY_RF"] 



    def classifyEventData(self, eventClass = None):

        if eventClass is None:
            eventClass = self.eventClass
        
        # Extract features and apply Scaler transform
        x = self.eventData["data"][self.featuresClass].values
        x_scaled = self.scalerClass.transform(x)

        prediction = self.classifier.predict_proba(x_scaled)[:,eventClass] # Here we're only taking the electron probability
        self.eventData["data"]["Prob"] = prediction


    def makeEffectiveAreas(self, prob = None):
        if prob is None:
            prob = self.prob_cut

        passingEvents = self.eventData["data"][self.eventData["data"]["Prob"]>prob]

        # Note x/y bins are flipped for numpy.histogram2d
        self.eventData["reconstructed_spectrum"], _, _ = np.histogram2d(
                                            passingEvents["Theta2"],
                                            passingEvents["ENERGY"],
                                            bins = [self.eventData["theta2_binning"], self.eventData["energy_binning"]]
        )
        
        # Calculated effective areas
        eff = self._aThrow * self.eventData["reconstructed_spectrum"] / self.eventData["simulated_spectrum"]
        eff[np.isnan(eff)] = 0 # Set 0/0 = 0

        # Calculate the solid angle
        offsetAngle = np.sqrt(self.eventData["theta2_binning_cen"])
        offsetBinWidth = np.sqrt(np.diff(self.eventData["theta2_binning"]))
        solidAngle = 2 * np.pi * offsetAngle * offsetBinWidth

        # self.eventData["effective_area"] = eff  * solidAngle[:, None]
        self.eventData["effective_area"] = eff


    def makeEffectiveAreasPlots(self, prob = None):

        # Check if the effective areas have been calculated
        if "effective_area" not in self.eventData.keys():
            self.makeEffectiveAreas(prob)

        # 1st plot 
        # | Simulated events | Reconstructed Events | Effective Areas|

        fig1, axs = plt.subplots(1,3, figsize = (18,6))

        p0 = axs[0].pcolormesh(self.eventData["energy_binning_cen"],
                self.eventData["theta2_binning_cen"],
                np.log10(self.eventData["simulated_spectrum"]),
                cmap = cmap)
        axs[0].set_xlabel("Simulated Energy [TeV]")
        axs[0].set_ylabel("Sky Location [$\\theta^2$]")
        fig1.colorbar(p0, ax=axs[0]).set_label("Number of Simulated Events")



        p1 = axs[1].pcolormesh(self.eventData["energy_binning_cen"],
                self.eventData["theta2_binning_cen"],
                np.log10(self.eventData["reconstructed_spectrum"]),
                cmap = cmap)
        axs[1].set_xlabel("Simulated Energy [TeV]")
        axs[1].set_ylabel("Sky Location [$\\theta^2$]")
        fig1.colorbar(p1, ax=axs[1]).set_label("Number of Reconstructed Events")



        p2 = axs[2].pcolormesh(self.eventData["energy_binning_cen"],
                self.eventData["theta2_binning_cen"],
                np.log10(self.eventData["effective_area"]),
                cmap = cmap)
        axs[2].set_xlabel("Simulated Energy [TeV]")
        axs[2].set_ylabel("Sky Location [$\\theta^2$]")
        fig1.colorbar(p2, ax=axs[2]).set_label("Effective Area [m$^2$]")


        [ax.grid() for ax in axs]

        axs[0].set_title("Simulated Events")
        axs[1].set_title("Reconstructed Events")
        axs[2].set_title("Effective Area")



        # 2nd plot
        # Effective areas as a function of camera wobble
        wob = np.array([0.5, 1.0, 1.25, 1.5])
        wob_theta2 = wob**2

        fig2 = plt.figure(figsize = (11,6))
        for theta2 in wob_theta2:
            amin = np.argmin(np.abs(theta2 - self.eventData["theta2_binning_cen"]))
            print (amin)
            plt.plot(
                self.eventData["energy_binning_cen"], 
                self.eventData["effective_area"][amin, :],
                label = f'{np.sqrt(self.eventData["theta2_binning_cen"][amin]):0.1f} degrees wobble' 
                    )
        plt.legend()
        plt.yscale('log')
        plt.grid(which = 'both')
        plt.xlim(-1,2)
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Effective Area [m$^2$]")


        fig1.tight_layout()
        fig2.tight_layout()

        return fig1, fig2 


    def makeEnergyResponse(self, prob = None):
        if prob is None:
            prob = self.prob_cut



        # Define binning
        eng_bins = 10**np.arange(-1,2, 0.05)
        migra_bins = np.linspace(0.2,5, 20)
        wob_bins = np.linspace(0,5, 6)

        # Energy Dispersion
        energyResponse = np.zeros(
            (
                wob_bins.shape[0]-1,
                migra_bins.shape[0]-1,
                eng_bins.shape[0]-1,
            )
        )

        passingEvents = self.eventData["data"][self.eventData["data"]["Prob"]>prob]



        for i in range(energyResponse.shape[0]):

            # Create the wobble mask
            wob_mask =   ( np.sqrt(passingEvents["Theta2"]) > wob_bins[i]) &\
                    ( np.sqrt(passingEvents["Theta2"]) < wob_bins[i+1])

            # wob_mask = (passingEvents["Theta2"] > self.eventData["theta2_binning"][i]) & (passingEvents["Theta2"] <= self.eventData["theta2_binning"][i+1])
            
            for j in range(energyResponse.shape[2]):

                    eng_mask = (passingEvents["ENERGY_MC"] > np.log10(eng_bins[j])) & \
                               (passingEvents["ENERGY_MC"] < np.log10(eng_bins[j+1])) 

                    energyResponse[i,:,j], _ = np.histogram(
                        10**passingEvents["ENERGY_LUT"][eng_mask & wob_mask] / \
                        10**passingEvents["ENERGY_MC"][eng_mask & wob_mask],
                        bins = migra_bins
                    )
                    
                    energyResponse[i] += 1e-9 # Remove 0/0
                    energyResponse[i,:,j] /= np.sum(energyResponse[i,:,j])
            

        self.eventData["energy_response"] = energyResponse
        self.eventData["energy_response_ebins"] = eng_bins
        self.eventData["energy_response_migra"] = migra_bins
        self.eventData["energy_response_theta"] = wob_bins



    def makeEnergyResponsePlots(self, prob = None):

        # Check if the effective areas have been calculated
        if "energy_response" not in self.eventData.keys():
            self.makeEnergyResponse(prob)


        fig, axes = plt.subplots(3,3, figsize = (18,18))

        for i, ax in enumerate(axes.ravel()):
            p = ax.pcolormesh(
                self.eventData["energy_response_ebins"], 
                self.eventData["energy_response_ebins"], 
                self.eventData["energy_response"][i],
                cmap = cmap)

            ax.plot(self.eventData["energy_response_ebins"], self.eventData["energy_response_ebins"], "r-")
            ax.plot(self.eventData["energy_response_ebins"], self.eventData["energy_response_ebins"] + 0.2, "r:")
            ax.plot(self.eventData["energy_response_ebins"], self.eventData["energy_response_ebins"] - 0.2, "r:")

            ax.set_title(f'{np.sqrt(self.eventData["theta2_binning_cen"][i]):0.2f} Degrees Wobble')
            ax.grid(which = 'both')
            ax.set_xlabel("Reconstructed Energy [TeV]")
            ax.set_ylabel("Simulated Energy [TeV]")
            fig.colorbar(p, ax=ax).set_label('Prob')

        fig.tight_layout()
        return fig



    # Getting the spatial dispersion
    def makeSpatialDispersion(self, prob = None):

        if prob is None:
            prob = self.prob_cut


        passingEvents = self.eventData["data"][self.eventData["data"]["Prob"]>prob]

        # Define the energy range
        ebins = np.logspace(-1,1,16)
        ebins_c = 10**(np.log10(ebins[:-1])  + 0.5*(np.log10(ebins[1:]) - np.log10(ebins[:-1])))

        # Wobble angles 
        wob_bins = np.arange(0, 2, 0.1)

        # Prob range
        rad_bins = np.linspace(0,2,15)

        rad_binsC = rad_bins[:-1] + 0.5 *(rad_bins[1:] - rad_bins[:-1])
        rad_bins_data = np.linspace(0,2, 50)
        rad_bins_dataC = rad_bins_data[:-1] + 0.5 *(rad_bins_data[1:] - rad_bins_data[:-1])

        # PSF Data
        # Shape rad, wob, eng
        r_data = np.zeros(
            (
                len(rad_bins_data)-1, 
                len(wob_bins)-1, 
                len(ebins)-1
            )
        )
        for i in range(r_data.shape[1]):
            
            # Get the wobble mask
            wob_mask =   ( passingEvents["MCTheta"] > wob_bins[i]) &\
                        ( passingEvents["MCTheta"] < wob_bins[i+1])
            for j in range(r_data.shape[2]):
                    
                    eng_mask = ( passingEvents["ENERGY_MC"] > np.log10(ebins[j]) ) &\
                            ( passingEvents["ENERGY_MC"] < np.log10(ebins[j+1]) )
                    
                    
                    
                    counts, _ = np.histogram(
                        passingEvents["MCTheta"][wob_mask & eng_mask] -\
                        passingEvents["Theta"][wob_mask & eng_mask],
                        bins = rad_bins
                    )
                    inter = interp1d(rad_binsC, 
                                    counts, 
                                    fill_value="extrapolate",
                                    bounds_error=False, kind="quadratic")
                    r_data[:,i,j] = inter(rad_bins_dataC)
                    r_data[:,i,j][r_data[:,i,j]< 0] = 0 
        #             print (np.sum(r_data[:,i,j]))
        #             r_data[:,i,j] /= np.sum(r_data[:,i,j])
                    
                    r_data[:,i,j] *= 10000    # I don't know either...

        self.eventData["spatial_response"] = r_data
        self.eventData["spatial_response_ebins"] = ebins
        self.eventData["spatial_response_rad"] = rad_bins_data
        self.eventData["spatial_response_theta"] = wob_bins

    def writeToFile(self, fname):
        # if "joblib" not in fname:
        #     fname += ".joblib"
        # # Might as well keep using joblib
        # dump(self.eventData, fname)
        drop_keys = ["data"]
        tab_keys = [ 'energy_binning', 'theta2_binning', 
                     'energy_binning_cen', 'theta2_binning_cen', 
                      'energy_response_ebins']
        save_keys = [key for key in self.eventData.keys() if key not in drop_keys ]
        phdu = fits.PrimaryHDU()
        hduls = [phdu]
        for k in save_keys:
            # if k in tab_keys:
                # hduls.append(fits.BinTableHDU(Table({}))
            # else:
            hduls.append(fits.ImageHDU(self.eventData[k]))

        hdul_list = fits.HDUList(hduls)
        for i, k in enumerate(save_keys):
            hdul_list[i+1].name = k

        for k in self.metaData.keys():
            hdul_list[0].header[k] = self.metaData[k]
        hdul_list.writeto( fname, overwrite=True)



    def writeGammapyIRFs(self, fname, prob = None):
        

        if prob is None:
            prob = self.prob_cut

        # Check if the effective areas have been calculated
        if "effective_area" not in self.eventData.keys():
            self.makeEffectiveAreas(prob)

        # self.eventData["energy_response"] = energyResponse
        # self.eventData["energy_response_ebins"] = eng_bins
        # self.eventData["energy_response_migra"] = migra_bins
        # self.eventData["energy_response_theta"] = wob_bins

        tab = Table(
            {
                "ENERG_LO" : [10**self.eventData["energy_binning"][:-1] ]* u.TeV,
                "ENERG_HI" : [10**self.eventData["energy_binning"][1:] ]* u.TeV,
                "THETA_LO" : [np.sqrt(self.eventData["theta2_binning"])[:-1]]* u.deg,
                "THETA_HI" : [np.sqrt(self.eventData["theta2_binning"])[1:]]* u.deg,
                "EFFAREA" : [self.eventData["effective_area"]] / u.m / u.m
            }
        )
        aeff = EffectiveAreaTable2D.from_table(tab)

        # self.eventData["spatial_response"] = r_data
        # self.eventData["spatial_response_ebins"] = eng_bins
        # self.eventData["spatial_response_rad"] = rad_bins
        # self.eventData["spatial_response_theta"] = wob_bins


        if "spatial_response" not in self.eventData.keys():
            self.makeSpatialDispersion(prob)

        print ("Spatial Response: " ,self.eventData["spatial_response"].shape)
        print ("Spatial Response ebins: " ,self.eventData["spatial_response_ebins"].shape)
        print ("Spatial Response theta: " ,self.eventData["spatial_response_theta"].shape)
        print ("Spatial Response rad: " ,self.eventData["spatial_response_rad"].shape)
        tab = Table(
            {
                "ENERG_LO" : [self.eventData["spatial_response_ebins"][:-1] ]* u.TeV,
                "ENERG_HI" : [self.eventData["spatial_response_ebins"][1:] ]* u.TeV,
                "THETA_LO" : [self.eventData["spatial_response_theta"][:-1] ]* u.deg,
                "THETA_HI" : [self.eventData["spatial_response_theta"][1:] ]* u.deg,
                "RAD_LO" : [self.eventData["spatial_response_rad"][:-1]] *u.deg,
                "RAD_HI" : [self.eventData["spatial_response_rad"][1:]] *u.deg,
                "RPSF" : [self.eventData["spatial_response"]] / u.sr
            }
        )

        psf = PSF3D.from_table(tab)

        # self.eventData["energy_response"] = energyResponse
        # self.eventData["energy_response_ebins"] = eng_bins
        # self.eventData["energy_response_migra"] = migra_bins
        # self.eventData["energy_response_theta"] = wob_bins

        if "energy_response" not in self.eventData.keys():
            self.makeEnergyResponse(prob)

        tab = Table(
            {
                "ENERG_LO" : [self.eventData["energy_response_ebins"][:-1] ]* u.TeV,
                "ENERG_HI" : [self.eventData["energy_response_ebins"][1:] ]* u.TeV,
                "MIGRA_LO" : [self.eventData["energy_response_migra"][:-1]],
                "MIGRA_HI" : [self.eventData["energy_response_migra"][1:]],
                "THETA_LO" : [self.eventData["energy_response_theta"][:-1] ]* u.deg,
                "THETA_HI" : [self.eventData["energy_response_theta"][1:] ]* u.deg,
                "MATRIX" : [self.eventData["energy_response"]]
            }
        )
        edisp = EnergyDispersion2D.from_table(tab)


        hdul = fits.HDUList(aeff.to_hdulist() + edisp.to_hdulist()[1:] + psf.to_hdulist()[1:])
        hdul.writeto( fname, overwrite=True)


'''
    Handle IRF I/O and interpolation
'''
class IRFHandler():

    def __init__(self):
        pass
    

    def readIRFFiles(self, filename):
        pass