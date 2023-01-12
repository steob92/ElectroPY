import numpy as np

# Machine Learning Stuff
## Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

## Models to test
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, RandomForestClassifier
from xgboost.sklearn import XGBRegressor

## Evaluating
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

from electropy.training.Preprocessing import Preprocessor

import logging 
from joblib import dump, load
import yaml

class FeatureTuner(Preprocessor):

    def __init__(self):
        super().__init__()
        self._n_jobs = 1
        self.test_size = 0.33
        self.setModel()
        self.method = "energy"


    @property
    def n_jobs(self):
        return self._n_jobs
    
    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = n_jobs
        # model needs to be updated once n_jobs is changed
        self.model.n_jobs = self._n_jobs

    def setModel(self, mtype = None):
        
        if mtype is not None:
            self.modelName = mtype
        elif "Model" in self.config:
            self.modelName = self.config["Model"]
        else :
            self.modelName = "RandomForestRegressor"
        self._makeModel()
        
    def _makeModel(self):
        if self.modelName == "RandomForestRegressor":
            self.model = RandomForestRegressor()
            criterion = 'squared_error'
        
        elif self.modelName == "XGBRegressor":
            self.model = XGBRegressor()
            criterion = 'squared_error'
        
        elif self.modelName == "RandomForestClassifier":
            self.model = RandomForestClassifier()
            criterion = 'gini'

        # Default parameter for a RandomForrest/XGB Regressor
        self.model.n_jobs = self._n_jobs
        self.model.n_estimators=100
        # self.model.n_estimators=10
        self.model.criterion=criterion
        self.model.max_depth=None
        self.model.min_samples_split=2
        self.model.min_samples_leaf=10
        self.model.min_weight_fraction_leaf=0.0
        self.model.max_features='sqrt'


    def performEnergyEstimation(self, config_file, scaler, model):
        # Load in what we need
        with open(config_file, 'r') as inFile:
            tmp_config = yaml.safe_load(inFile)
        tmp_model = load(model)
        tmp_scaler = load(scaler)

        # Obtain the model prediction
        self.df["ENERGY"] = tmp_model.predict(
                tmp_scaler.transform(
                    self.df[tmp_config["Features"]].values
                )
            )

    def fitModel(self, features):
        # Extract features and values
        if self.method == "energy":
            y = self.df["ENERGY_MC"].values
        elif self.method == "classify":
            y = self.df["label"].values

        x = self.df[features].values
    
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size)

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Fit
        self.model.fit(x_train_scaled, y_train)
        # Score
        score = self.model.score(x_test_scaled, y_test)
        # Percision
        pre = 0
        if self.method == "classify":
            # Check the precision of classifying electrons
            # ToDo: Code in alternative optimization methods
            y_pred = self.model.predict(x_test_scaled)
            ele_mask = y_pred == 0
            
            pre = precision_recall_fscore_support(
                    y_test, 
                    y_pred,
                    pos_label = 0,
                    average = "binary")

        # Get Feature Importance
        imp = self.model.feature_importances_

        return score, imp, pre       

    def writeModel(self, fileName):
        dump(self.model, fileName)
        

    def tuneFeatures(self, delta = 0.05, refit = False):

        # Check that we have multiple labels
        if self.method == "classify":
            if len(np.unique(self.df["label"])) < 2:
                logging.error("Only one label type found, cannot run classifier")
                return -1

        max_score = -1
        new_score = -1
        if "Features" in self.config:
            logging.warning("Initial features loaded from config file")
            features_new = np.array(self.config["Features"])
        else:
            logging.warning("Initial features infered from data")
            features_new = self.df.keys().drop(["ENERGY_MC", "label"])

        # Require at least 4 parameters
        count_max = len(features_new) - 4

        # Things to store
        self.features_list = []
        self.nfeatures_list = []
        self.imp_list = []
        self.score_list = []

        converged = False
        # for i in range(count_max):
        while len(features_new) > 4:

            new_score, importance, prec = self.fitModel(features_new)
            asort = np.argsort(importance)[::-1]
            
            # For classifier use precission instead of f-score 
            if self.method == "classify":
                print (prec)
                new_score = prec[0]

            self.features_list.append([ feat for feat in features_new[asort] ])
            self.nfeatures_list.append(len(self.features_list[-1]))
            self.imp_list.append([ imp for imp in importance[asort] ])
            self.score_list.append(new_score)

            if (new_score > max_score):
                    logging.warning(f'New max found: {new_score:0.4f} (old: {max_score:0.4f})')
                    max_score = new_score

            for feat, impt in zip(features_new[asort], importance[asort]):
                logging.debug(f"\t{feat}: {100*impt:0.2f}")
        

            if ( np.abs(max_score - new_score) < delta ) :
                feat_tmp = features_new[asort][:-1]

                # if any of the telescope wise parameters are removed then remove all of them
                if ( "size2_" in features_new[asort][-1]):

                    logging.warning("Removing telescope level size2 parameters")

                    feat_mask = np.array([ True if "size2_" in feat  else False for feat in features_new ])
                    feat_tmp = features_new[~feat_mask]

                elif ( "tgrad_x_" in features_new[asort][-1]):

                    logging.warning("Removing telescope level tgrad parameters")

                    feat_mask = np.array([ True if "tgrad_x_" in feat  else False for feat in features_new ])
                    feat_tmp = features_new[~feat_mask]

                elif ( "loss_" in features_new[asort][-1]):

                    logging.warning("Removing telescope level loss parameters")

                    feat_mask = np.array([ True if "loss_" in feat  else False for feat in features_new ])
                    feat_tmp = features_new[~feat_mask]

                # Setting the new features for the next itteration
                features_new = feat_tmp
                logging.debug(f"Continuing... ({new_score:0.2f}, {len(features_new)})")
            else: 
                logging.warning(f"Stopping...")
                n_feat = len(self.features_list[-2])
                logging.warning(f"{n_feat} Parameters to be used:")
                ostr = ""
                for i in range (n_feat):
                    ostr += f"\n\t{self.features_list[-2][i]}"
                logging.warning(f"{ostr}")
                self.config["Method"] = self.method
                self.config["Features"] = self.features_list[-2]
                converged = True
                break
        
        if not converged:
            logging.warning(f"Tuning didn't converge... Consider changing the tolerance (delta = {delta:0.2f}) ")
            self.config["Features"] = self.features_list[-1]
            n_feat = len(self.features_list[-1])
            logging.warning(f"Reverting to last parameters...")
            logging.warning(f"{n_feat} Parameters to be used:")
            ostr = ""
            for i in range (n_feat):
                ostr += f"\n\t{self.features_list[-1][i]}"
            logging.warning(f"{ostr}")
            
        if refit == True:
            self.fitModel(self.config["Features"])