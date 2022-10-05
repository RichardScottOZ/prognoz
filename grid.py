import os
from osgeo import gdal, osr
import numpy as np

from PyQt5.QtWidgets import QFileDialog, QWidget, QMessageBox


class Grid:
    """
    Class that stores data about a raster and has methods for manipulation.
    """
    
    def __init__(self, load = True, **kwargs):
        """
        Function creates instance by one of two ways:
        1. import from an absolute path
        2. reading from memory (copy of a derivative)
        
        Input:
            load - bool, method for creating instance.
        if load:
            filename = str, short filename, e.g.: 'WGM2012.tif'
            
        """
        
        if load == True: # if import by absolute path
            self.filename = kwargs['filename'] # short filename with extension
            self.name =  kwargs['grid_name'] # without extension
            self.parent = ''
            self.fullpath = os.path.join(kwargs['path'], self.filename)
            self.open_geotiff(kwargs['band_id']) # call import function
        else:
            self.name = kwargs['grid_name']
            self.parent = kwargs['parent']
            self.img = kwargs['img']
            self.meta = kwargs['meta']
            
        
    def open_geotiff(self, band_id):
        """
        Function imports raster from absolute path. A specific band is imported.
        Input:
            band_id - int, layer id. Starts from 1
            
        Output:
            self.img
            self.meta
        """
        
        # Input data
        fullpath = self.fullpath
        
        # Start import session
        ds = gdal.Open(fullpath)
        band = ds.GetRasterBand(band_id)
               
        # Read a raster
        nodata = band.GetNoDataValue()
        self.img = band.ReadAsArray() #.astype(np.float32)
        
        # Replace raster values that should be NaN as NaN
        # Check, if nodata attribute exists and raster is not binary [0,1]
        if nodata is not None and not all([ self.img.dtype == 'uint8', 
                                           np.array_equal(np.unique(self.img), np.array([0,1]))
                                           ]):
            self.img[self.img == nodata] = np.nan
        
        # Raster params
        shape = self.img.shape # dimension (nrows, ncols)
        nrows = ds.RasterYSize 
        ncols = ds.RasterXSize 
        # geographical params
        ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
        # lower right corner
        lrx = ulx + (ncols * xres)
        lry = uly + (nrows * yres)
        # CRS
        proj_str = ds.GetProjection()
        proj = osr.SpatialReference(wkt=ds.GetProjection())
        epsg_str = proj.GetAttrValue('AUTHORITY',1)
        epsg = int(epsg_str)
        
        # Session close. MUST BE DONE
        ds = None
        
        # Coordinates of corners
        def get_xy(ulx, uly, lrx, lry, ncols, nrows):
            """Function calculates rasters coordinates"""
            xres = (lrx-ulx)/ncols # xres - positive num
            yres = (lry-uly)/nrows # yres - negative num
            yy = np.linspace(uly + yres*0.5, lry - yres*0.5, nrows)
            xx = np.linspace(ulx + xres*0.5, lrx - xres*0.5, ncols)
            return yy, xx
        # XY vectors
        yy, xx = get_xy(ulx, uly, lrx, lry, ncols, nrows)
        
        # Write metadata dict
        self.meta = {'nodata':nodata, 'shape':shape, 'ulx':ulx, 'xres':xres,
                     'uly':uly, 'yres':yres, 'proj_str':proj_str, 'proj':proj,
                     'epsg':epsg, 'ncols':ncols, 'nrows':nrows, 'lrx':lrx, 'lry':lry,
                     'yy':yy, 'xx':xx}
    
    def export(self, **kwargs):
        """Function exports a raster"""
        
        # default params
        default_path = kwargs.get('default_path', None)
        
        # Data preparation
        img = self.img
        ulx = self.meta['ulx']
        xres = self.meta['xres']
        uly = self.meta['uly']
        yres = self.meta['yres']
        ncols = self.meta['ncols']
        nrows = self.meta['nrows']
        proj_str = self.meta['proj_str']
        name = self.name
        
        # function to get fullpath
        def getSaveFileName(default_filename):
            """
            SaveFileName dialog window
            """
            
            file_filter = 'Raster Files (*.tif)'
            filename, _ = QFileDialog.getSaveFileName(
                caption = "Export selected raster",
                directory = default_filename, 
                filter = file_filter,
                initialFilter = file_filter
                )
            return filename
        
        # Full raster name (path + name + extension)
        if default_path is None:
            fullname = getSaveFileName(name)
            if not fullname:
                return # if Cancel
        else:
            fullname = default_path
        
        try:
            # raster_bands - number of bands
            raster_bands = 1
            
            # Export itself
            geotransform=(ulx, xres, 0, uly, 0, yres)
            output_raster = gdal.GetDriverByName('GTiff').Create(fullname, 
                                        ncols, nrows, raster_bands, gdal.GDT_Float32)
            output_raster.SetGeoTransform(geotransform)
            output_raster.SetProjection(proj_str)
            output_raster.GetRasterBand(1).WriteArray(img)
            QMessageBox.about(QWidget(),'Succeful', 'The raster has been exported')
        except Exception:
            QMessageBox.about(QWidget(),'Error', 'Could not export the raster')
        finally:
            del output_raster # close session. MUST BE DONE
            
            
    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            return None
