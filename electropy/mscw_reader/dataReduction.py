import numpy as np
import pandas as pd
from .MscwReader import MscwReader
from .templates import reductionTemplate
from .IRFMaker import IRFHandler
import logging

class DataRetuction():

    def __init__(self):
        self.irfs = IRFHandler()
        pass

    
    '''
        Read in a config file which will store the runlist
        and store the irf file names
    '''
    def read_config(self, configFile):
        logging.debug(f"Reading config file: {configFile}")
        pass

    '''
        Loop over mscw files in the runlist and produce a DL3 file
    '''
    def reduce_files(self):
        logging.debug(f"Reducing files")
        pass