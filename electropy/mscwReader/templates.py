reductionTemplate="""
FileName: MSCWData
OutputName: Output
EnergyEstimator: energy.joblib
Classifier: class.joblib
"""

irfTemplate="""
EnergyScaler: "/home/obriens/DataAnalysis/Veritas/electron/EnergyEstimator/rndf_scaler.joblib"
Energy: "/home/obriens/DataAnalysis/Veritas/electron/EnergyEstimator/rndf_energy.joblib"
FeaturesEnergy: 
- SizeSecondMax
- Core
- size2_1
- size2_2
- theta2
- size2_3
- tgrad_x_1
- tgrad_x_2
- size2_0
- tgrad_x_3
- tgrad_x_0
modelEnergy: RandomForestRegressor
ClassifierScaler: "/home/obriens/DataAnalysis/Veritas/electron/EnergyEstimator/rndf_scaler.joblib"
Classifier: "/home/obriens/DataAnalysis/Veritas/electron/EnergyEstimator/rndf_scaler.joblib"
modelClassifier: RandomForestClassifier
FeaturesScaler: 
- SizeSecondMax
- Core
- size2_1
- size2_2
- theta2
- size2_3
- tgrad_x_1
- tgrad_x_2
- size2_0
- tgrad_x_3
- tgrad_x_0
LogFeat:
- EmissionHeight
- SizeSecondMax
- Core
- ENERGY_MC
- size2_0
- size2_1
- size2_2
- size2_3
"""

# Default header for EVENTS card
# Copied from https://gamma-astro-data-formats.readthedocs.io/en/v0.3/
events_hdul_header = {
    "HDUCLASS"  : "GADF",
    # type: string
        # Signal conformance with HEASARC/OGIP conventions (option: ‘OGIP’). See HDU classes.

    "HDUDOC"  : "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/",
    # type: string
        # Reference to documentation where data format is documented. See HDU classes.

    "HDUVERS" : '1.0.0',
    # type: string
        # Version of the format (e.g. ‘1.0.0’). See HDU classes.

    "HDUCLAS1" : 'EVENTS',
    # type: string
        # Primary extension class (option: ‘EVENTS’). See HDU classes.

    "OBS_ID" : None,
    # type: int
        # Unique observation identifier (Run number)

    "TSTART" : None,
    # type: float, unit: s
        # Start time of observation (relative to reference time, see Time)

    "TSTOP" : None,
    # type: float, unit: s
        # End time of observation (relative to reference time, see Time)
    
    # Additional stuff from V2DL3
    "MJDREFI" : 53402.0,
    "MJDREFF" : 0.0,
    "TIMEUNIT" : 's',
    "TIMESYS" : 'utc',
    "TIMEREF" : 'topocenter',

    "ONTIME" : None,
    # type: float, unit: s
        # Total good time (sum of length of all Good Time Intervals). If a Good Time Interval (GTI) table is provided, ONTIME should be calculated as the sum of those intervals. Corrections for instrumental dead time effects are NOT included.

    "LIVETIME" : None,
    # type: float, unit: s
        # Total time (in seconds) on source, corrected for the total instrumental dead time effect.

    "DEADC" : None,
    # type: float
        # Dead time correction, defined by LIVETIME/ONTIME. Is comprised in [0,1]. Defined to be 0 if ONTIME=0.

    "OBS_MODE" : "POINTING",
    # type: string
            # Observation mode. See notes on OBS_MODE below.
            
    "RA_PNT" : None,
    # type: float, unit: deg
        # Pointing Right Ascension (see RA / DEC). Not mandatory if OBS_MODE=DRIFT, but average values could optionally be provided.

    "DEC_PNT" : None,
    # type: float, unit: deg
        # Pointing declination (see RA / DEC). Not mandatory if OBS_MODE=DRIFT, but average values could optionally be provided.

    "ALT_PNT" : None,
    # type: float, unit: deg
        # Pointing Altitude (see Alt / Az). Only mandatory if OBS_MODE=DRIFT

    "AZ_PNT" : None,
    # type: float, unit: deg
        # Pointing azimuth (see Alt / Az). Only mandatory if OBS_MODE=DRIFT

    "EQUINOX" : 2000.0,
    # type: float
        # Equinox in years for the celestial coordinate system in which positions given in either the header or data are expressed (options: 2000.0). See also HFWG Recommendation R3 for the OGIP standard.

    "RADECSYS" : "ICRS", 
    # type: string
        # Stellar reference frame used for the celestial coordinate system in which positions given in either the header or data are expressed. (options: ‘ICRS’, ‘FK5’). See also HFWG Recommendation R3 for the OGIP standard.

    "ORIGIN" : "VERITAS",
    # type: string
            # Organisation that created the FITS file. This can be the same as TELESCOP (e.g. “HESS”), but it could also be different if an organisation has multiple telescopes (e.g. “NASA” or “ESO”).

    "TELESCOP" : "VERITAS",
    # type: string
        # Telescope (e.g. ‘CTA’, ‘HESS’, ‘VERITAS’, ‘MAGIC’, ‘FACT’)

    "INSTRUME" : "VERITAS",
    # type: string
        # Instrument used to aquire the data contained in the file. Each organisation and telescop has to define this. E.g. for CTA it could be ‘North’ and ‘South’, or sub-array configurations, this has not been defined yet.

    "CREATOR" : "ELECTROPY",
    # type: string
}