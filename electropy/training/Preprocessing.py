import numpy as np

from astropy.table import Table
from astropy.io import fits
import pandas as pd
import logging

# logging.basicConfig(filename='feat.log', filemode='w', format='%(asctime)s - %(levelname)s- %(message)s', level=logging_level)

# Machine Learning Stuff
## Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Config file in yaml
import yaml

# electropy
from electropy.training.templates import configTemplate

# Saving to joblib
from joblib import dump, load




class Preprocessor():

    def __init__(self):

        # Create a default scaler and config file
        self.df = None 
        self.setScaler()
        self.makeConfig()


    def setScaler(self, type = "std"):
        if type.lower() == "std":
            self.scaler = StandardScaler()
        elif type.lower() == "minmax":
            self.scaler = MinMaxScaler()

    def loadScaler(self, fileName):
        self.scaler = load(fileName)

    def writeScaler(self, fileName):
        dump(self.scaler, fileName)

    # Make a new config from a template
    def makeConfig(self):
        self.config = yaml.safe_load(configTemplate)


    # Loading in a new config file
    def loadConfig(self, fileName):
        with open(fileName, 'r') as inFile:
            self.config = yaml.safe_load(inFile)

    # Writing the config file
    def writeConfig(self, fileName):
        with open(fileName, "w") as outFile:
            yaml.dump(self.config, outFile, default_flow_style=False)



    # readData should be internal to allow for labeling event types
    def _readData(self, fileName = None, label = 0):

        # Check if a file name is passed
        if fileName == None:
            fileName = self.config["FileName"]
        else:
            self.config["FileName"] = fileName

        with fits.open(self.config["FileName"]) as hdul:
            df = Table.read(hdul[self.config["Table"]]).to_pandas()

        # Label the datasets used in the config file
        if f"data_{label}_0" not in self.config:
            self.config[f"data_{label}_0"] = fileName
        else :
            keys = self.config.keys()
            # Get the last entry and increment by 1
            ent = np.array([ int(key.split("_")[-1]) for key in keys if f"data_{label}" in key])
            dataset = ent[np.argsort(ent)][-1] +1
            self.config[f"data_{label}_{dataset}"] = fileName

        return self.cleanData(df)

    def addData(self, fileName = None, label = 0):
        df = self._readData(fileName, label)
        df['label'] = label
        if self.df is None:
            self.df = df
        else :
            self.df = pd.concat([self.df, df])


    def cleanData(self, df, config=None):
        if config == None:
            config = self.config

        foi = df.keys().drop(config["DroppedFeatures"])
        df = df[foi]
        for feat in config["LogFeat"]:
            df.loc[df[feat] == 0, (feat)] = 0.1

            # Silence warning for now
            pd.options.mode.chained_assignment = None  
            df.loc[:, (feat)] = np.log10(df[feat])
            pd.options.mode.chained_assignment = "warn"  

        return df