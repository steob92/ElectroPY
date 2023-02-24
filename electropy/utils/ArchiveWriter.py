''' 
    Script to read data downloaded from 
    https://tools.ssdc.asi.it/CosmicRays/
    and convert to fits table
'''

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import sys

if __name__ == "__main__":

    # Get the input dir and the out name
    indir = sys.argv[1]
    outname = sys.argv[2]
    
    tables = []

    # Loop over the entries in this file
    with open(indir + "/File_content_index.txt", "r") as f:

        lines = f.readlines()
    
        for line in lines:
            # skip comments
            if line[0] == "#":
                continue

            # ";" sepeareted file
            line = line.split(";")
            # Stores filenames as xml, reading in txt files instead
            filename = line[0].replace("__1__.xml", ".txt").replace(".xml", ".txt").strip()
            # extract some meta data for later
            mission = line[2].strip()
            year = line[3].split("(")[-1].split(")")[0]
            ref = line[3].strip()

            # Print progress
            print (filename, mission, ref, year)
            
            # Load in as numpy and apply units
            data = np.loadtxt(indir + "/" + filename)
            energy = np.array(data[:,0]) * u.GeV
            energy_errl = np.array(data[:,1]) * u.GeV
            energy_erru = np.array(data[:,2]) * u.GeV

            # Assuming one downloaded E^3...
            # (m^{2} s sr GeV)^{-1}#times(GeV/n)^{3.0}
            flux = np.array(data[:,3]) / (u.m**2 * u.s * u.sr *u.GeV) *u.GeV**3
            flux_errl = np.array(data[:,4]) * flux.unit
            flux_erru = np.array(data[:,5]) * flux.unit

            # Make table with the data
            tab = Table(
                data = {
                    "energy" : energy,
                    "energy_errl" :   energy_errl,
                    "energy_erru" : energy_erru,
                    "flux" : flux / energy**3,
                    "flux_errl" : flux_errl / energy**3,
                    "flux_erru" : flux_erru / energy**3
                },
                meta = {
                    "filename": filename,
                    "mission" : mission,
                    "refer" : ref,
                    "year" : year
                }
            )
            tables.append(tab)
    
    # empty primary for the moment
    phdu = fits.PrimaryHDU()
    hdus = [phdu]
    # Add the binary tables
    for tab in tables:
        hdus.append(fits.BinTableHDU(tab))
        
    hdul = fits.HDUList(hdus)

    # Rename and store meta data to header
    for i, tab in enumerate(tables):
        for k in tab.meta.keys():
            hdul[i+1].header[k] = tab.meta[k]
        hdul[i+1].name = tab.meta["mission"] + "_" + tab.meta["year"] 

    hdul.writeto(outname, overwrite=True)