import numpy as np
import pandas as pd
from .mscwReader import mscwReader
from .templates import reductionTemplate
from .IRFMaker import IRFHandler
import logging

class dataReduction():

    def __init__(self):
        self.irfs = IRFHandler()
        pass

    
    '''
        Read in a config file which will store the runlist
        and store the irf file names
    '''
    def readConfig(self, configFile):
        logging.debug(f"Reading config file: {configFile}")
        pass

    '''
        Loop over mscw files in the runlist and produce a DL3 file
    '''
    def reduceFiles(self):
        logging.debug(f"Reducing files")
        pass