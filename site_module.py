# Global modules
import os
import numpy as np
import pandas as pd
from osgeo import ogr, gdal
import json
from sklearn.preprocessing import StandardScaler
import warnings
import pywt
from scipy import ndimage, interpolate
from skimage.transform import resize
from PyQt5.QtWidgets import QMessageBox, QWidget

# local modules
from grid import Grid

class Site(object):
    """
    Class stores all data about an area: gridsd, points, boundaries.
    It also has methods of data processing.
    """
    
    def __init__(self, path, folder):
        """
        During __init__ class imports shapefiles: "Boundaries", "Deposits",
        '*.tif' rasters. Multiband rasters are allowed. 
        Also it compares rasters dimension with the first loaded. 
        All rasters must have equal dimension.
        """
        
        self.site = folder
        self.path = path
        
        # Import boundaries shapefile
        boundaries_filepath = os.path.join(self.path, self.site, 'Boundaries.shp')
        self.boundaries = self.download_boundaries(boundaries_filepath)
        
        # Import deposits shapefile
        deposits_filepath = os.path.join(self.path, self.site, 'Deposits.shp')
        self.deposits, self.labels = self.download_deposits(deposits_filepath)
        
        # Import tif files
        filenames = self.get_filenames('.tif')
        self.grids = {}
        for filename in filenames:
            # filepath
            folderpath = os.path.join(self.path, self.site)
            filepath = os.path.join(folderpath, filename)
            # START import session
            src_ds = gdal.Open(filepath)
            # count bands
            n_bands = src_ds.RasterCount
            # CLOSE import session. MUST BE DONE
            src_ds = None
            
            # get bands names. They could be stored in a separate txt-file
            bands_names_lst = self.get_multiband_names(folderpath, filename, n_bands)
            
            # Import band after band
            for i in range(n_bands):
                grid_name = bands_names_lst[i]
                self.grids[grid_name] = Grid(load = True, 
                                               path = folderpath, 
                                               filename = filename,
                                               grid_name = grid_name,
                                               band_id = i+1
                                               ) #i+1, because rasters ids'
                                                # start from 1
        # check rasters dimensions
        for i, key in enumerate( list( self.grids.keys() ) ):
            shape = self.grids[key].meta['shape']
            if i == 0:
                self.grid_shape = shape
                self.first_key = key
            elif self.grid_shape != shape:
                QMessageBox.about(QWidget(), 'Error', f'<p align="left">Grid shape: <br>\
                                      {key} : {shape}<br>\
                                      is not equal the first loaded grid: <br>\
                                      {self.first_key} : {self.grid_shape}.<br>\
                                    Use grids with similar shapes.</p>')
                return
        # create a structure with rasters
        self.structure = {grid:list() for grid in self.grids.keys()}
    
    def download_boundaries(self, boundaries_filepath):
        """Functions imports Boundaries.shp
        """
        # storage
        boundaries = []
        
        # If file exists
        if os.path.exists(boundaries_filepath):
            file = ogr.Open(boundaries_filepath)
            shape = file.GetLayer(0)
        else:
            return boundaries
        
        # Get coordinates
        for item in shape:
            item_dict = json.loads(item.ExportToJson())    
            geometry_type = item_dict['geometry']['type']
            if geometry_type == 'LineString':
                x = [i[0] for i in item_dict['geometry']['coordinates']]
                y = [i[1] for i in item_dict['geometry']['coordinates']]
                boundaries.append([x, y])
            elif geometry_type == 'MultiLineString':
                for sub_lst in item_dict['geometry']['coordinates']:
                    x = [i[0] for i in sub_lst]
                    y = [i[1] for i in sub_lst]
                    boundaries.append([x, y])
        return boundaries
    
    def download_deposits(self, deposits_filepath):
        """Function imports Deposits.shp
        Output: two tables - deposits, labels
        """
        # storage
        deposits = []
        labels = []
        
        # If file exists
        if os.path.exists(deposits_filepath):
            file = ogr.Open(deposits_filepath)
            shape = file.GetLayer(0)
        else:
            return deposits, labels
        
        # If column Commodity exists. OBLIGATORY column
        properties = [*json.loads(shape[0].ExportToJson())['properties']]
        properties = [item.lower().capitalize() for item in properties]
        if 'Commodity' not in properties:
            QMessageBox.about(QWidget(), 'Error', '"Deposits.shp" does not \
                    have column: "Commodity". The file has not been imported')
            return deposits, labels
        
        # Extract coordinates
        for item in shape:
            item_dict = json.loads(item.ExportToJson())    
            # attributes
            attributes_lst = list(item_dict['properties'].values())
            group = item_dict['id']
            # points coordinates
            # first of all check type: Point vs MultiPoint
            if item_dict['geometry']['type'] == 'Point':
                x = item_dict['geometry']['coordinates'][0]
                y = item_dict['geometry']['coordinates'][1]
                deposits.append(attributes_lst+[group, x, y])
            elif item_dict['geometry']['type'] == 'MultiPoint':
                # if MultiPoint, check number of points inside: one or more
                item_dict_len = len(item_dict['geometry']['coordinates'])
                if item_dict_len == 1:
                    x = item_dict['geometry']['coordinates'][0][0]
                    y = item_dict['geometry']['coordinates'][0][1]                
                    deposits.append(attributes_lst+[group, x, y])
                elif item_dict_len > 1:
                    for point in item_dict['geometry']['coordinates']:
                        x, y = point
                        deposits.append(attributes_lst+[group, x, y])
        
        # Redefine deposits table for output
        features_names = properties + ['INNERGROUP', 'x', 'y'] # add INNERGROUP, x, y
        deposits_out = pd.DataFrame(deposits, columns = features_names) # create table
        
        # Check if Train column exists, and if it was correctly populated
        if 'Train' in deposits_out.columns:
            # Capitalize Train column
            deposits_out['Train'] = deposits_out['Train'].str.capitalize()
            # Check only two possible values: Train, Test
            if set(deposits_out['Train']) != set(['Train', 'Test']):
                deposits_out['Train'] = "" # set to empty
                # warn if incorrectly filled
                QMessageBox.about(QWidget(), 'Error', f'Incorrectly filled\
                                      the column Train in Deposits.shp \
                                      withing a site {self.site}. \
                                      Train will be generated automatically.')
        else: # if no column Train, we make new one empty
            deposits_out['Train'] = ""
        
        # Create labels table if they exist in Deposits.shp
        if 'Label' in properties:
            labels_df = deposits_out.loc[deposits_out['Label'].notnull(), 
                                ['Label', 'Commodity', 'INNERGROUP', 'x', 'y']]
            labels_groupped = labels_df.groupby('INNERGROUP').first()
            labels_out = labels_groupped.sort_values(by = ['y'], 
                                        ascending=False, ignore_index = True)
        else:
            labels_out = labels
        
        return deposits_out, labels_out
    
    def get_multiband_names(self, folderpath, name, n_bands):
        """Function reads txt file with names for multiband tif raster"""
        raster_name = os.path.join(folderpath, name)
        fullname = raster_name.replace('.tif', '.txt')
        # if file exist
        if os.path.exists(fullname):
            with open(fullname) as f:
                bands_names_lst = [line.rstrip() for line in f]
                # if number of rows in txt is less than number of bands,
                # create new names: band 1, band 2, etc
                if len(bands_names_lst) != n_bands:
                    bands_names_lst = [f'{name} layer {i}' for i in range(1, n_bands+1)]
        else: # if no such a file
            if n_bands == 1:
                bands_names_lst = [name.replace('.tif', '')]
            else:
                bands_names_lst = [f'{name} layer {i}' for i in range(1, n_bands+1)]
        
        # output
        return bands_names_lst
            
    def get_filenames(self, endings):
        """Function gets filenames of specific ending"""
        path = os.path.join(self.path, self.site)
        filenames = [f for f in os.listdir(path) if f.endswith(endings)]
        return filenames
    
    def _nan_mask(self, img):
        """Function checks if there are nan values in raster and creates a mask"""
        # If NaN exist, create a mask
        nan_exist = np.isnan(img).any()
        mask = np.logical_not(np.isnan(img))
        return nan_exist, mask
    
    def _nan_interp(self, img):
        """Function extrapolate values to NaN cells"""
        if self.nan_exist:
            # 2D raster coordinates
            xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            xym = np.vstack( (np.ravel(xx[self.mask]), np.ravel(yy[self.mask])) ).T
            # valid values to 1D вектор (XYZ)
            data = np.ravel( img[self.mask] )
            # create interpolator
            interp = interpolate.NearestNDInterpolator( xym, data )
            # interpolate all
            result = interp(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
            # rewrite NaN values
            img[~self.mask] = result[~self.mask]
        return img
    
    def _save_to_dict(self, img, short_name, parent_name):
        """
        Function resize wavelet derivatives to initial raster size.
        Assign back NaN values.
        Finally, write new raster to storage.
        
        Input:
            img - 2D array, wavelet derivative in it initial dimension
            meta - dict, metadata about raster
            parent_name - parent raster name
            interpolation_method - int, interpolation type # 0 - Nearest 
                    Neighbor, 1 - Bilinear, 2 - Cubic
            short_name - short grid name by mask: *wavelet* + 
                *a[approximation] or d[detailes]* + *level* + 
                *moving window size [optional]* + *specific product* 
            nan_exist - bool - if any NaN cell exists in an initial raster
            mask - 2D bool, True - Real number
        Output:
            new Grid written to self.grids dict
            record in self.structure dict to consequent raster
        """
        # WITH EACH OUTPUT WRITE NAN VALUES BACK
        def nan_return(img):
            if self.nan_exist:
                img[~self.mask] = np.nan
            return img
        
        # rescale wavelet derivative to initial raster
        img_rescaled = resize(img, (self.meta['nrows'], self.meta['ncols']), 
                                            order = self.interpolation_method)
        # reset NaN
        img_rescaled = nan_return(img_rescaled)
        # write Grid into dict self.grids
        new_grid_name = f'{parent_name} {short_name}'
        self.grids[new_grid_name] = Grid(load = False, 
                                         grid_name = new_grid_name, 
                                         parent = parent_name, 
                                         img = img_rescaled,
                                         meta = self.meta)
        # write short_name in structure dict
        # if wavelet derivative will be calculated again, it will be rewritten,
        # but without duplicate in a dict
        if not short_name in self.structure[parent_name]:
            self.structure[parent_name].append(short_name)
    
    def wavelet_preparation(self, grid_name, wave_form, products):
        """
        Function makes wavelet decomposition of a selected raster        
        Input:
            grid_name - str, grid name to process
            wave_form - wavelet parameters
            products - dict, checkboxes of wavelet products
        """
        # read parameters of wavelet decomposition
        wave_name = str(wave_form.w_box.currentText())
        lvl_max = wave_form.w_spin.value()
        window_size = wave_form.window_spin.value()
        self.interpolation_method = wave_form.interpolation_box.currentIndex()
        
        # Input data
        # copy, because there will be NaN check and assignment
        img = np.array(self.grids[grid_name].img)
        self.meta = self.grids[grid_name].meta
        
        # check NaN values and create a mask
        self.nan_exist, self.mask = self._nan_mask(img)
        # interpolate NaN values
        img = self._nan_interp(img)
            
        # Wavelet decomposition and restore by level and type (approx., detailes)
        coeffs = pywt.wavedec2(img, wave_name) # wavelet decomposition
        def reconstruct(coeffs):
            # coeffs separately
            cA = coeffs[0]
            cD = coeffs[1:]
            # storage
            app = {}
            det = {}
            # levels
            max_lvl = len(cD)
            lvls = list(range(max_lvl,0,-1))
            # loop by all levels
            for i in range(max_lvl):
                # approximations
                coef = (cA, cD[i])
                app[lvls[i]] = pywt.waverec2(coef,'haar')
                # detailes
                coef = (np.zeros_like(cA), cD[i])
                det[lvls[i]] = pywt.waverec2(coef,'haar')
                # rewrite approximation coefficients
                cA = app[lvls[i]]    
            return app, det
        app, det = reconstruct(coeffs)
        
        # Loop by levels
        for lvl in app:            
            # check lvl
            if lvl > lvl_max: 
                continue
            
            # Approximations
            img_a = app[lvl] # extract from a dict
            # save to storage
            if 'product_wa' in products and products['product_wa']:
                short_name = f'{wave_name} a lvl-{lvl}'
                self._save_to_dict(img_a, short_name, grid_name)
            
            # Wavelet-detailes
            img_d = det[lvl] # extract from storage
            # save to storage
            if 'product_wd' in products and products['product_wd']:
                short_name = f'{wave_name} d lvl-{lvl}'
                # write Grid into self.grids dict
                self._save_to_dict(img_d, short_name, grid_name)
                
            # Moving STD (standard deviation) for wavelet detailes
            if 'product_wd_std' in products and products['product_wd_std']:
                img_mean = ndimage.uniform_filter(img_d, size = window_size) # mean value
                img_subtr = np.subtract(img_d, img_mean) # subtract from each cell
                img_mult = np.multiply(img_subtr,img_subtr) # squre the difference
                img_sqmean = ndimage.uniform_filter(img_mult, size = window_size) # mean of squared values
                # finally calc STD - aquire root of variance, but
                # the variance in a point could be equal to zero, so numpy
                # send a warning and save with a nan,
                # therefore we apply nan = 0
                with warnings.catch_warnings(record=True) as w: # # catch a warning
                    img_d_std = np.sqrt(img_sqmean) # final STD calc
                    if len(w) > 0: # if warning happend
                        img_d_std = np.nan_to_num(img_d_std)
                # save to dict
                short_name = f'{wave_name} d lvl-{lvl} sz-{window_size} STD'
                self._save_to_dict(img_d_std, short_name, grid_name)
            
            # Relative absolute wavelet detailes (abs Z-scores)
            if 'product_wd_rel' in products and products['product_wd_rel']:
                img_d_rel = (np.abs(img_d) - np.mean(img_d)) / np.std(img_d)
                # save to dict
                short_name = f'{wave_name} d lvl-{lvl} sz-{window_size} abs Z'
                self._save_to_dict(img_d_rel, short_name, grid_name)
            
            # Moving range (max-min) of wavelet detailes
            if 'product_wd_range' in products and products['product_wd_range']:
                img_max = ndimage.maximum_filter(img_d, size = window_size)
                img_min = ndimage.minimum_filter(img_d, size = window_size)
                img_range = img_max - img_min
                
                # Save to dict
                short_name = f'{wave_name} d lvl-{lvl} sz-{window_size} MinMax'
                self._save_to_dict(img_range, short_name, grid_name)

    def make_multiband_img(self):
        """ Function creates multiband raster from all rasters within an area.
        Rasters are standardized."""
        
        # Rasters names
        self.multiband_names = [*self.grids]
        self.multiband_names.sort()
        
        # loop by rasters names
        for i, item in enumerate(self.multiband_names):
            # Standardize raster
            # if grid is binomial [0,1], do not standardize
            if not np.array_equal( np.unique(self.grids[item].img), np.array([0,1]) ):
                grid_add_stdz = StandardScaler().fit_transform(self.grids[item].img)
            else:
                grid_add_stdz = self.grids[item].img
           
            # Check sequence number
            if i == 0:
                self.multi_img = grid_add_stdz
            elif i == 1:
                self.multi_img = np.stack((self.multi_img, grid_add_stdz))
            else:
                self.multi_img = np.concatenate((self.multi_img, [grid_add_stdz]))
    
    def get_relative_points_coords(self):
        """Function get XY transformation coordinates. From real to raster"""
        # Get params of a first raster
        grids_names = [*self.grids]
        grid_xx = self.grids[grids_names[0]].meta['xx']
        grid_yy = self.grids[grids_names[0]].meta['yy']
        
        self.deposits['Col'] = self.deposits['x'].apply(lambda x, 
                            grid_xx = grid_xx: (np.abs(grid_xx - x)).argmin())
        self.deposits['Row'] = self.deposits['y'].apply(lambda y, 
                            grid_yy = grid_yy: (np.abs(grid_yy - y)).argmin())
    
    def get_point_values(self):
        """Function extract multiband raster values at points"""
        # Check if table has no points
        if len(self.deposits) == 0:
            return
        
        # Get point coordinates in raster cells
        self.get_relative_points_coords()
        
        # extract values at points 
        if self.multi_img.ndim == 3:
            point_values = self.multi_img[:,self.deposits.Row, self.deposits.Col]
        elif self.multi_img.ndim == 2:
            point_values = self.multi_img[self.deposits.Row, self.deposits.Col]
        
        # Create table of extracted values and group it by INNERGROUP,
        # just in case of multipoints
        point_values_reshaped = np.array(point_values).T
        point_values_df = pd.DataFrame(point_values_reshaped, columns = self.multiband_names)
        # Group table of names and classes
        self.deposits_values = self.deposits[
                    ['Name', 'Commodity', 'Train', 'INNERGROUP', 'Row', 'Col']
                                            ].groupby('INNERGROUP').first()
        # in a loop check features
        # if a feature is binomial ([0,1]), take mode of a group
        # if not - mean value
        for i in self.multiband_names:
            if np.array_equal(np.unique(point_values_df[i]), np.array([0,1])):
                col_groupped = point_values_df.groupby(
                    self.deposits['INNERGROUP'])[i].agg(
                        lambda x: pd.Series.mode(x)[0])
                        # lambda in bimodal distribution get first value
            else:
                col_groupped = point_values_df.groupby(self.deposits['INNERGROUP'])[i].mean()
            self.deposits_values[i] = col_groupped
        
        self.deposits_values['Site'] = self.site
        
    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            return None
