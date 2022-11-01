import numpy as np
from astropy.coordinates import SkyCoord
import uproot
import pandas as pd
from pyslalib import slalib
from VSkyCoordinatesUtility import *



fileName = '../../ED_data/root-files/RecID0/97170.mscw.root'
f = uproot.open(fileName)
pointing = f["pointingDataReduced"].arrays(library="np")
data = f["data"].arrays(library="np")

tel_ra = np.rad2deg(np.mean(pointing["TelRAJ2000"]))
tel_dec = np.rad2deg(np.mean(pointing["TelDecJ2000"]))
pointing_position = SkyCoord(tel_ra, tel_dec, unit='deg', frame='icrs')
target_position = SkyCoord.from_name('Crab')

data_dict = {
    "runNumber": data["runNumber"],
    "eventNumber" : data["eventNumber"],
            "timeOfDay": data["Time"],
            "MJD": data["MJD"],
      "Erec" : data["ErecS"],
        "Erec_Err" : data["dES"],
    "NImages" : data["NImages"],
    "ImgSel": data["ImgSel"],
    "MeanPedvar": data["meanPedvar_Image"],
    "MSCW" : data["MSCW"],
        "MSCL" : data["MSCL"],
    "EmissionHeight" : data["EmissionHeight"],
    
    
        "Xoff_derot": data["Xoff_derot"],
        "Yoff_derot": data["Yoff_derot"],
        
        
        "EChi2S" : data["EChi2S"],
      
        "SizeSecondMax" : data["SizeSecondMax"],
        "XCore" : data["Xcore"],
        "YCore" : data["Ycore"],
        "Theta2": data["theta2"],
    "Xoff" : data["Xoff"],
    "Yoff": data["Yoff"]
        }

    # Use pandas df for use with classifier                                                                                                                                                            
df = pd.DataFrame(data_dict)
mask = df["Theta2"] <= 2.0

df = df[mask]


derot = np.array(list(map(convert_derotatedCoordinates_to_J2000, getUTC(df.MJD, df.timeOfDay),
                        np.repeat(target_position.ra.deg, len(df)),
                        np.repeat(target_position.dec.deg, len(df)),
                        df['Xoff_derot'], df['Yoff_derot'])))


df['Xderot'] = derot[:,0]
df['Yderot'] = derot[:,1]


radec = list(map(slalib.sla_dtp2s, np.deg2rad(df.Xderot), np.deg2rad(df.Yderot), 
   np.repeat(np.deg2rad(pointing_position.ra.deg), len(df)),
                        np.repeat(np.deg2rad(pointing_position.dec.deg), len(df)),
   ))


df['RA'] = np.rad2deg([radec[0] for radec in radec])
df['DEC'] = np.rad2deg([radec[1] for radec in radec])


elaz = list(map(getHorizontalCoordinates, df.MJD, df.timeOfDay, df.DEC, df.RA))


df['El'] = [elaz[0] for elaz in elaz]
df['Az'] = [elaz[1] for elaz in elaz]


required_col = ['runNumber', 'eventNumber', 'timeOfDay', 'MJD', 'Erec',
                'Erec_Err', 'XCore', 'YCore', 'Xderot', 'Yderot', 'NImages',
                'ImgSel', 'MeanPedvar', 'MSCW', 'MSCL', 'RA',
                'DEC', 'Az', 'El', 'EmissionHeight', 'Xoff', 'Yoff']


df_new = df[required_col]
print(df_new)
