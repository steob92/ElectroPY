import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.table import Table
from astropy.io import fits

from joblib import dump, load

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
        
        # Scalar, classifier and features of interest
        self.scalar = None
        self.classifier = None
        self.features = None
        self.log_feat = None


        # For the effective areas
        self._rThrow = 750 # m
        self._aThrow = np.pi * self._rThrow**2   # m^2  


    def loadScalar(self, fname):
        self.scalar = load(fname)

    def loadClassifier(self, fname):
        self.classifier = load(fname)

    def loadFeatures(self, fname):
        feat = load(fname)
        self.features = feat["Features"]
        self.log_feat = feat["Log Features"]



    def readData(self, fname):

        # Open the fits files
        with fits.open(fname) as hdul:

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


    def classifyEventData(self, eventClass = None):

        if eventClass is None:
            eventClass = self.eventClass
        
        # Extract features and apply scalar transform
        x = self.eventData["data"][self.features].values
        x_scaled = self.scalar.transform(x)

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


    def writeToFile(self, fname):
        if "joblib" not in fname:
            fname += ".joblib"
        # Might as well keep using joblib
        dump(self.eventData, fname)
