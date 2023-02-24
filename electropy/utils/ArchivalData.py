import numpy as np
from astropy.io import fits
from astropy.table import Table
# from astropy import units as u
from pathlib import Path

class Archive():

    def __init__(self):
        # List of archived datasets
        self._archiveFiles = ["electron_data.fits", "proton_data.fits"]
        self.readArchive()
        

    def readArchive(self):

        self.data = {}
        path = Path(__file__)

        for f in self._archiveFiles:
            part = f.split("_")[0]
            
            if part not in self.data:
                self.data[part] = {}
            with fits.open(path.parent.absolute().joinpath(f)) as hdul:
                for i in range(len(hdul) -1):
                    name = hdul[i+1].name
                    tab = Table.read(hdul[i+1])
                    self.data[part][name] = tab
    

    def __str__(self):
        ret = ""
        for part in self.data:
            ret+= f'Particle type: {part}\n'

            for pub in self.data[part]:
                datum = self.data[part][pub]
                ret+= f'\tMission: {datum.meta["MISSION"]}\n'
                ret+= f'\tYear: {datum.meta["YEAR"]}\n'
                ret+= f'\tReference: {datum.meta["REFER"]}\n'
                ret+= "\n"

        return ret 
