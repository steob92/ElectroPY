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
        self.event_class = 0

        # Cut on the probability
        self.prob_cut = 0.5
        
        # Scaler, classifier and features of interest
        # self.scaler = None
        # self.classifier = None
        # self.features_class = None
        # self.features_energy = None
        self.log_feat = None


        # For the effective areas
        self._r_throw = 750 # m
        self._a_throw = np.pi * self._r_throw**2   # m^2  

        # meta data for output
        self.meta_data = {}

    def load_config(self, fname):
        with open(fname, "r") as f:
            self.config = yaml.safe_load(f)

        # Load in the classifier and scalers
        self.load_energy_scaler()
        self.load_class_scaler()
        self.load_energy_estimator()
        self.load_classifier()
        self.log_feat = self.config["LogFeat"]
        

    def load_energy_scaler(self, fname = None):
        # Load in the energy scaler and extract the features
        if fname is None:
            self.scaler_energy = load(self.config["EnergyScaler"])
            self.features_energy = self.config["FeaturesEnergy"]
        else:
            self.scaler_energy = load(fname)


    def load_class_scaler(self, fname = None):
        # Load in the classifier scaler and extract the features
        if fname is None:
            self.scaler_class = load(self.config["ClassifierScaler"])
            self.features_class = self.config["FeaturesClassifier"]
        else:
            self.scaler_class = load(fname)

    def load_energy_estimator(self, fname = None):
        # Load in the energy estimator
        if fname is None:
            self.energy_estimator = load(self.config["Energy"])
        else:
            self.energy_estimator = load(fname)
        

    def load_classifier(self, fname = None):
        # Load in the Classifier
        if fname is None:
            self.classifier = load(self.config["Classifier"])
        else:
            self.classifier = load(fname)

    # def loadFeatures(self, fname):
    #     feat = load(fname)
    #     self.features_class = feat["Features_class"]
    #     if "Features_energy" in feat:
    #         self.features_energy = feat["Features_energy"]
    #     self.log_feat = feat["Log Features"]


    def read_data(self, fname):

        # Open the fits files
        with fits.open(fname) as hdul:

            self.meta_data = hdul[0].header
            # Convert event-wise data to a dataframe
            df = Table.read(hdul[1]).to_pandas()

            # Get the energy and theta2 binning
            ebin = Table.read(hdul[2])["ebin"]
            theta2_bin = Table.read(hdul[3])["theta2"]
            ebinc = ebin[:-1] + 0.5*(ebin[1:] - ebin[:-1])
            theta2_binc = theta2_bin[:-1] + 0.5*(theta2_bin[1:] - theta2_bin[:-1])

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

        self.event_data =  {
            "data" : df, 
            "energy_binning" : ebin, 
            "theta2_binning" : theta2_bin, 
            "simulated_spectrum" :spect,
            "energy_binning_cen": ebinc,
            "theta2_binning_cen": theta2_binc
        }

    def estimate_energy(self):
        x = self.event_data["data"][self.features_energy].values
        x_scaled = self.scaler_energy.transform(x)
        prediction = self.energy_estimator.predict(x_scaled)

        self.event_data["data"]["ENERGY_LUT"] = np.log10(self.event_data["data"]["ENERGY"])
        self.event_data["data"]["ENERGY_RF"] = prediction
        self.event_data["data"]["ENERGY"] = self.event_data["data"]["ENERGY_RF"] 



    def classify_event_data(self, event_class = None):

        if event_class is None:
            event_class = self.event_class
        
        # Extract features and apply Scaler transform
        x = self.event_data["data"][self.features_class].values
        x_scaled = self.scaler_class.transform(x)

        prediction = self.classifier.predict_proba(x_scaled)[:,event_class] # Here we're only taking the electron probability
        self.event_data["data"]["Prob"] = prediction


    def make_effective_areas(self, prob = None):
        if prob is None:
            prob = self.prob_cut

        passing_events = self.event_data["data"][self.event_data["data"]["Prob"]>prob]

        # Note x/y bins are flipped for numpy.histogram2d
        self.event_data["reconstructed_spectrum"], _, _ = np.histogram2d(
                                            passing_events["Theta2"],
                                            passing_events["ENERGY"],
                                            bins = [self.event_data["theta2_binning"], self.event_data["energy_binning"]]
        )
        
        # Calculated effective areas
        eff = self._a_throw * self.event_data["reconstructed_spectrum"] / self.event_data["simulated_spectrum"]
        eff[np.isnan(eff)] = 0 # Set 0/0 = 0

        # Calculate the solid angle
        offset_angle = np.sqrt(self.event_data["theta2_binning_cen"])
        offset_bin_width = np.sqrt(np.diff(self.event_data["theta2_binning"]))
        solidAngle = 2 * np.pi * offset_angle * offset_bin_width

        # self.event_data["effective_area"] = eff  * solidAngle[:, None]
        self.event_data["effective_area"] = eff


    def make_effective_areas_plots(self, prob = None):

        # Check if the effective areas have been calculated
        if "effective_area" not in self.event_data.keys():
            self.make_effective_areas(prob)

        # 1st plot 
        # | Simulated events | Reconstructed Events | Effective Areas|

        fig1, axs = plt.subplots(1,3, figsize = (18,6))

        p0 = axs[0].pcolormesh(self.event_data["energy_binning_cen"],
                self.event_data["theta2_binning_cen"],
                np.log10(self.event_data["simulated_spectrum"]),
                cmap = cmap)
        axs[0].set_xlabel("Simulated Energy [TeV]")
        axs[0].set_ylabel("Sky Location [$\\theta^2$]")
        fig1.colorbar(p0, ax=axs[0]).set_label("Number of Simulated Events")



        p1 = axs[1].pcolormesh(self.event_data["energy_binning_cen"],
                self.event_data["theta2_binning_cen"],
                np.log10(self.event_data["reconstructed_spectrum"]),
                cmap = cmap)
        axs[1].set_xlabel("Simulated Energy [TeV]")
        axs[1].set_ylabel("Sky Location [$\\theta^2$]")
        fig1.colorbar(p1, ax=axs[1]).set_label("Number of Reconstructed Events")



        p2 = axs[2].pcolormesh(self.event_data["energy_binning_cen"],
                self.event_data["theta2_binning_cen"],
                np.log10(self.event_data["effective_area"]),
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
            amin = np.argmin(np.abs(theta2 - self.event_data["theta2_binning_cen"]))
            print (amin)
            plt.plot(
                self.event_data["energy_binning_cen"], 
                self.event_data["effective_area"][amin, :],
                label = f'{np.sqrt(self.event_data["theta2_binning_cen"][amin]):0.1f} degrees wobble' 
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


    def make_energy_response(self, prob = None):
        if prob is None:
            prob = self.prob_cut



        # Define binning
        eng_bins = 10**np.arange(-1,2, 0.05)
        migra_bins = np.linspace(0.2,5, 20)
        wob_bins = np.linspace(0,5, 6)

        # Energy Dispersion
        energy_response = np.zeros(
            (
                wob_bins.shape[0]-1,
                migra_bins.shape[0]-1,
                eng_bins.shape[0]-1,
            )
        )

        passing_events = self.event_data["data"][self.event_data["data"]["Prob"]>prob]



        for i in range(energy_response.shape[0]):

            # Create the wobble mask
            wob_mask =   ( np.sqrt(passing_events["Theta2"]) > wob_bins[i]) &\
                    ( np.sqrt(passing_events["Theta2"]) < wob_bins[i+1])

            # wob_mask = (passing_events["Theta2"] > self.event_data["theta2_binning"][i]) & (passing_events["Theta2"] <= self.event_data["theta2_binning"][i+1])
            
            for j in range(energy_response.shape[2]):

                    eng_mask = (passing_events["ENERGY_MC"] > np.log10(eng_bins[j])) & \
                               (passing_events["ENERGY_MC"] < np.log10(eng_bins[j+1])) 

                    energy_response[i,:,j], _ = np.histogram(
                        10**passing_events["ENERGY_LUT"][eng_mask & wob_mask] / \
                        10**passing_events["ENERGY_MC"][eng_mask & wob_mask],
                        bins = migra_bins
                    )
                    
                    energy_response[i] += 1e-9 # Remove 0/0
                    energy_response[i,:,j] /= np.sum(energy_response[i,:,j])
            

        self.event_data["energy_response"] = energy_response
        self.event_data["energy_response_ebins"] = eng_bins
        self.event_data["energy_response_migra"] = migra_bins
        self.event_data["energy_response_theta"] = wob_bins



    def make_energy_response_plots(self, prob = None):

        # Check if the effective areas have been calculated
        if "energy_response" not in self.event_data.keys():
            self.make_energy_response(prob)


        fig, axes = plt.subplots(3,3, figsize = (18,18))

        for i, ax in enumerate(axes.ravel()):
            p = ax.pcolormesh(
                self.event_data["energy_response_ebins"], 
                self.event_data["energy_response_ebins"], 
                self.event_data["energy_response"][i],
                cmap = cmap)

            ax.plot(self.event_data["energy_response_ebins"], self.event_data["energy_response_ebins"], "r-")
            ax.plot(self.event_data["energy_response_ebins"], self.event_data["energy_response_ebins"] + 0.2, "r:")
            ax.plot(self.event_data["energy_response_ebins"], self.event_data["energy_response_ebins"] - 0.2, "r:")

            ax.set_title(f'{np.sqrt(self.event_data["theta2_binning_cen"][i]):0.2f} Degrees Wobble')
            ax.grid(which = 'both')
            ax.set_xlabel("Reconstructed Energy [TeV]")
            ax.set_ylabel("Simulated Energy [TeV]")
            fig.colorbar(p, ax=ax).set_label('Prob')

        fig.tight_layout()
        return fig



    # Getting the spatial dispersion
    def make_spatial_dispersion(self, prob = None):

        if prob is None:
            prob = self.prob_cut


        passing_events = self.event_data["data"][self.event_data["data"]["Prob"]>prob]

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
            wob_mask =   ( passing_events["MCTheta"] > wob_bins[i]) &\
                        ( passing_events["MCTheta"] < wob_bins[i+1])
            for j in range(r_data.shape[2]):
                    
                    eng_mask = ( passing_events["ENERGY_MC"] > np.log10(ebins[j]) ) &\
                            ( passing_events["ENERGY_MC"] < np.log10(ebins[j+1]) )
                    
                    
                    
                    counts, _ = np.histogram(
                        passing_events["MCTheta"][wob_mask & eng_mask] -\
                        passing_events["Theta"][wob_mask & eng_mask],
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

        self.event_data["spatial_response"] = r_data
        self.event_data["spatial_response_ebins"] = ebins
        self.event_data["spatial_response_rad"] = rad_bins_data
        self.event_data["spatial_response_theta"] = wob_bins

    def write_to_file(self, fname):
        # if "joblib" not in fname:
        #     fname += ".joblib"
        # # Might as well keep using joblib
        # dump(self.event_data, fname)
        drop_keys = ["data"]
        tab_keys = [ 'energy_binning', 'theta2_binning', 
                     'energy_binning_cen', 'theta2_binning_cen', 
                      'energy_response_ebins']
        save_keys = [key for key in self.event_data.keys() if key not in drop_keys ]
        phdu = fits.PrimaryHDU()
        hduls = [phdu]
        for k in save_keys:
            # if k in tab_keys:
                # hduls.append(fits.BinTableHDU(Table({}))
            # else:
            hduls.append(fits.ImageHDU(self.event_data[k]))

        hdul_list = fits.HDUList(hduls)
        for i, k in enumerate(save_keys):
            hdul_list[i+1].name = k

        for k in self.meta_data.keys():
            hdul_list[0].header[k] = self.meta_data[k]
        hdul_list.writeto( fname, overwrite=True)



    def write_gammapy_irfs(self, fname, prob = None):
        

        if prob is None:
            prob = self.prob_cut

        # Check if the effective areas have been calculated
        if "effective_area" not in self.event_data.keys():
            self.make_effective_areas(prob)

        # self.event_data["energy_response"] = energy_response
        # self.event_data["energy_response_ebins"] = eng_bins
        # self.event_data["energy_response_migra"] = migra_bins
        # self.event_data["energy_response_theta"] = wob_bins

        tab = Table(
            {
                "ENERG_LO" : [10**self.event_data["energy_binning"][:-1] ]* u.TeV,
                "ENERG_HI" : [10**self.event_data["energy_binning"][1:] ]* u.TeV,
                "THETA_LO" : [np.sqrt(self.event_data["theta2_binning"])[:-1]]* u.deg,
                "THETA_HI" : [np.sqrt(self.event_data["theta2_binning"])[1:]]* u.deg,
                "EFFAREA" : [self.event_data["effective_area"]] / u.m / u.m
            }
        )
        aeff = EffectiveAreaTable2D.from_table(tab)

        # self.event_data["spatial_response"] = r_data
        # self.event_data["spatial_response_ebins"] = eng_bins
        # self.event_data["spatial_response_rad"] = rad_bins
        # self.event_data["spatial_response_theta"] = wob_bins


        if "spatial_response" not in self.event_data.keys():
            self.make_spatial_dispersion(prob)

        print ("Spatial Response: " ,self.event_data["spatial_response"].shape)
        print ("Spatial Response ebins: " ,self.event_data["spatial_response_ebins"].shape)
        print ("Spatial Response theta: " ,self.event_data["spatial_response_theta"].shape)
        print ("Spatial Response rad: " ,self.event_data["spatial_response_rad"].shape)
        tab = Table(
            {
                "ENERG_LO" : [self.event_data["spatial_response_ebins"][:-1] ]* u.TeV,
                "ENERG_HI" : [self.event_data["spatial_response_ebins"][1:] ]* u.TeV,
                "THETA_LO" : [self.event_data["spatial_response_theta"][:-1] ]* u.deg,
                "THETA_HI" : [self.event_data["spatial_response_theta"][1:] ]* u.deg,
                "RAD_LO" : [self.event_data["spatial_response_rad"][:-1]] *u.deg,
                "RAD_HI" : [self.event_data["spatial_response_rad"][1:]] *u.deg,
                "RPSF" : [self.event_data["spatial_response"]] / u.sr
            }
        )

        psf = PSF3D.from_table(tab)

        # self.event_data["energy_response"] = energy_response
        # self.event_data["energy_response_ebins"] = eng_bins
        # self.event_data["energy_response_migra"] = migra_bins
        # self.event_data["energy_response_theta"] = wob_bins

        if "energy_response" not in self.event_data.keys():
            self.make_energy_response(prob)

        tab = Table(
            {
                "ENERG_LO" : [self.event_data["energy_response_ebins"][:-1] ]* u.TeV,
                "ENERG_HI" : [self.event_data["energy_response_ebins"][1:] ]* u.TeV,
                "MIGRA_LO" : [self.event_data["energy_response_migra"][:-1]],
                "MIGRA_HI" : [self.event_data["energy_response_migra"][1:]],
                "THETA_LO" : [self.event_data["energy_response_theta"][:-1] ]* u.deg,
                "THETA_HI" : [self.event_data["energy_response_theta"][1:] ]* u.deg,
                "MATRIX" : [self.event_data["energy_response"]]
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
    

    def read_irf_files(self, filename):
        pass