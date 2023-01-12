configTemplate="""
DroppedFeatures:
- runNumber
- EVENT_ID
- MJD
- Time
- TargetElev
- TargetAz
- NTtype
- TargetDec
- TargetRA
- WobbleN
- WobbleE
- Ze
- Az
- ra
- dec
- Xoff
- Yoff
- Xoff_derot
- Yoff_derot
- El
- RA
- DEC
- ENERGY
- EChi2S
- dES
- TIME
- timeOfDay
FileName: ../fits_files/simulations/electron_30deg_50wob_NOISE120.mscw.fits
LogFeat:
- EmissionHeight
- SizeSecondMax
- Core
- ENERGY_MC
- size2_0
- size2_1
- size2_2
- size2_3
Model: RandomForestRegressor
Table: MSCW
param_grid:
  n_estimators: 100
"""