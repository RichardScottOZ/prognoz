# global modules
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
# local modules
from grid import Grid

class PredictionModel:
    """Class that creates two classification models: SVC and MLP. Fits them,
    and apply to make binary and probability predictions."""
    
    def __init__(self, model_form):
        """Function that makes X and Y train-test subsets,
        creates SVC and MLP model and fits them"""
        # input data
        data = model_form.deposits_to_use
        self.selected_rasters = model_form.selected_rasters
        
        self.xTrain = data.loc[data.Train == 'Train', self.selected_rasters]
        self.xTest = data.loc[data.Train == 'Test', self.selected_rasters]
        self.yTrain = data.loc[data.Train == 'Train', 'Commodity']
        self.yTest = data.loc[data.Train == 'Test', 'Commodity']
        
        self.Y_unique = np.unique(self.yTrain)
        
        # Fit classification models
        self.train_models()
        
    def train_models(self):
        """Function fits two models:
        SVC - Support Vector Classifier;
        MLP - Multi-layer Perceptron classifier.
        Output - dict with classification scores on train-test subsets,
        and fitted models"""
        
        # Create SVC model
        self.model_SVC = SVC(gamma=2, C=2, probability=True).fit(self.xTrain, self.yTrain)
        #  Matrices of classification (test subset)
        self.yPred_SVC = self.model_SVC.predict(self.xTest)
        
        # Create MLP model
        self.model_MLP = MLPClassifier(solver = "sgd", 
                                       learning_rate = "constant", 
                                       momentum = 0.9, 
                                       nesterovs_momentum = True, 
                                       learning_rate_init = 0.2,
                                       validation_fraction = 0.2
                                       ).fit(self.xTrain, self.yTrain)
        #  Matrices of classification (test subset)
        self.yPred_MLP = self.model_MLP.predict(self.xTest)
        
        # Show overal result
        self.scores = {}
        self.scores['train_SVC'] = self.model_SVC.score(self.xTrain, self.yTrain)
        self.scores['test_SVC'] = self.model_SVC.score(self.xTest, self.yTest)
        self.scores['train_MLP'] = self.model_MLP.score(self.xTrain, self.yTrain)
        self.scores['test_MLP'] = self.model_MLP.score(self.xTest, self.yTest)
        
    def make_prediction(self, multi_img, multiband_names, meta, site_key, **kwargs):
        """
        Function applies a classification model to input multiband raster
        and returns pediction rasters for each class."""
        
        # unpack **kwargs - they should contain bool for saving params
        svc_bool = kwargs.get('svc_bool', False)
        mlp_bool = kwargs.get('mlp_bool', True)
        binary_bool = kwargs.get('binary_bool', False)
        
        # Index of selected rasters in multiband_names
        selected_rasters_idx = [multiband_names.index(item) for item 
                                                in self.selected_rasters]
        
        # Create subset of selected layers in multiband raster
        # and reshape it to 2D array, where feature equals column
        multi_img_selected = multi_img[selected_rasters_idx,::]
        multi_img_1d = multi_img_selected.ravel().reshape(-1, len(self.selected_rasters), order = 'F')
        
        # Check if NaN values exist and create a mask for Nan rows
        nan_exist = False
        true_shape = multi_img.shape[1:3] # raster size
        if np.isnan(multi_img_1d).any(): # check NaN presence
            nan_exist = True
            # create indice vector of NaN values
            idx_nan = np.isnan(multi_img_1d).any(axis=1)
            # Create subse 'not a NaN'. It will be an input to a model 
            multi_img_1d = multi_img_1d[~np.isnan(multi_img_1d).any(axis=1)]
        else:
            idx_nan = None
        
        # Apply a classification model
        imgs_pred = {}
        predictions_structure = {}
        if svc_bool:
            pred_1d_SVC = self.model_SVC.predict_proba(multi_img_1d)
            predictions_structure['SVC'] = []
        if mlp_bool:
            pred_1d_MLP = self.model_MLP.predict_proba(multi_img_1d)
            predictions_structure['MLP'] = []
        
        # Reshape 1D to 2D raster
        def get_prediction_2d(nan_exist, true_shape, idx_nan, pred_1d):
            """ Function reshapes 1D prediction to 2D vectors. 
            It assigns NaN values if they existed in an initial raster."""
            if nan_exist:
                # vector with NaN-values
                true_array = np.empty(true_shape[0]*true_shape[1])
                true_array.fill(np.nan)
                # input predictions
                true_array[~idx_nan] = pred_1d
                # reshape into 2D
                z_array = np.reshape(true_array, true_shape)
            else:
                z_array = np.reshape(pred_1d, true_shape)
            return z_array
        
        # get predictions for selected class
        for i, item in enumerate(self.Y_unique): # loop by classes names
            #  SVC
            if svc_bool:
                # calc probabilities
                z_SVC = get_prediction_2d(nan_exist, true_shape, 
                                          idx_nan, pred_1d_SVC[:,i])
                # write it into dict-storage
                name = site_key + ' SVC '+item
                imgs_pred['SVC '+ item] = Grid(load = False, grid_name = name, 
                                               parent = '', img = z_SVC, 
                                               meta = meta)
                # write it name into structure
                predictions_structure['SVC'].append(item)
            
            # MLP
            if mlp_bool:
                # get probabilities
                z_MLP = get_prediction_2d(nan_exist, true_shape, 
                                          idx_nan, pred_1d_MLP[:,i])
                # write it name into dict-storage
                name = site_key + ' MLP '+item
                imgs_pred['MLP ' + item] = Grid(load = False, grid_name = name, 
                                                parent = '', img = z_MLP, 
                                                meta = meta)
                # write it name into structure
                predictions_structure['MLP'].append(item)
        
        # get binary predictions of selected class
        if binary_bool:
            # SVC-binary
            if svc_bool:
                groups_SVC = self.model_SVC.predict(multi_img_1d)
                for group in self.Y_unique:
                    array = np.array(groups_SVC == group).astype(int)
                    z_binary = get_prediction_2d(nan_exist, true_shape, 
                                                             idx_nan, array)
                    # write it name into dict-storage
                    name = site_key + ' SVC ' + group + ' binary'
                    imgs_pred['SVC '+ group + ' binary'] = Grid(load = False, 
                                                grid_name = name, parent = '', 
                                                img = z_binary, meta = meta)
                    # write it name into structure
                    predictions_structure['SVC'].append(group + ' binary')
            
            # MLP-binary
            if mlp_bool:
                groups_MLP = self.model_MLP.predict(multi_img_1d)
                for group in self.Y_unique:
                    array = np.array(groups_MLP == group).astype(int)
                    z_binary = get_prediction_2d(nan_exist, true_shape, 
                                                             idx_nan, array)
                    # write it name into dict-storage
                    name = site_key + ' MLP ' + group + ' binary'
                    imgs_pred['MLP '+ group + ' binary'] = Grid(load = False, 
                                                grid_name = name, parent = '', 
                                                img = z_binary, meta = meta)
                    # write it name into structure
                    predictions_structure['MLP'].append(group + ' binary')
        
        return imgs_pred, predictions_structure
    