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