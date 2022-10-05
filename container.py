import os
import pandas as pd
import numpy as np
import pywt

from site_module import Site

class Container:
    "Main storage of data"

    def __init__(self, path):
        
        # List of subfolders. Each subfolder - unique area/tenement
        self.folders = next(os.walk(path), ([],[],[]))[1]
        
        # Create Sites - classes that store the data of an area
        self.sites = {}
        for folder in self.folders:
            self.sites[folder] = Site(path, folder)
        self.sites_count = len(self.sites) # количество участков
        
        # Check all imported rasters
        # Because user is allowed to import multiband rasters, that are
        # harder to check before loading, we do check at the end
        rasters_all = []
        for key in self.sites.keys():
            folder_rasters = list(self.sites[key].grids.keys())
            rasters_all.extend(folder_rasters)
        self.common_major_grids = [item for item in set(rasters_all) if 
                                   rasters_all.count(item) == self.sites_count]
        self.common_major_grids.sort()
        # delete rasters out of a list
        for site_key in self.sites.keys():
            for grid_key in self.sites[site_key].grids.keys():
                if grid_key not in self.common_major_grids:
                    del self.sites[site_key].grids[grid_key]
        
        # Get max wavelet decomposition level
        self.get_max_wavelet_lvl()
           
    def get_max_wavelet_lvl(self):
        """
        Function checks all rasters in all areas and gets max wavelet decompo-
        sition level
        """
        self.lvl_max = {'haar':100, 'sym7':100}
        for key in self.sites.keys():
            min_shape = np.min(self.sites[key].grid_shape)
            for wave_name in self.lvl_max:
                current_lvl = pywt.dwt_max_level(min_shape, wave_name)
                if self.lvl_max[wave_name] > current_lvl:
                    self.lvl_max[wave_name] = current_lvl

    def get_wavelet_derivatives(self, wave_form):
        """Function to calculate wavelet derivatives"""
        # Get list of selected rasters
        selected_grids = [item.text() for item in wave_form.w_list.selectedItems()]
        if len(selected_grids) == 0:
            return
        
        # get dict of derivatives
        products = wave_form.get_products_status()
        if not any(products.values()): # if no selected, return
            return
        
        # get list of selected rasters, and wavelet decomposition params 
        for site in self.sites:
            for grid in selected_grids:
                self.sites[site].wavelet_preparation(grid, wave_form, products)
        
        wave_form.close() # close the window
    
    def get_all_deposits_values(self):
        """Function extract rasters values at points"""
        for site_key in self.sites: 
            site = self.sites[site_key]
            # Create multiband raster for each area
            site.make_multiband_img()
            # Extract rasters values at points
            site.get_point_values()
            # Join tables of point values and deposits into one
            if not hasattr(self, 'all_deposits_values') and hasattr(site, 'deposits_values'):
                self.all_deposits_values = site.deposits_values
            elif hasattr(self, 'all_deposits_values') and hasattr(site, 'deposits_values'):
                self.all_deposits_values = pd.concat([self.all_deposits_values, 
                                                      site.deposits_values],
                                                     axis = 0,
                                                     ignore_index = True)
        
        self.multiband_names = site.multiband_names
        self.all_deposits_values = self.all_deposits_values.dropna(subset = self.multiband_names)
    
    def __getitem__(self, item):
        return getattr(self, item)