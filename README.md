# prognoz
main is prognoz.py

**ABOUT**
**A program for prediction new perspective mineral deposits areas**. You could use one or more unique areas. At least one should have deposits points. The program makes preprocessing on rasters, extracts rasters values at points, fits SVC (Support Vector Classification) and MLP (Multi-Layer Perceptron) classification models. Finally applies models to selected areas. If there are multiple areas with points - the latter will be consolidated to one table.
**Input**:
- folder with subfolders. Each subfolder - unique area with files inside (rasters, shapefiles).
- rasters must be *.tif extension. 
    - Withing each area they must have equal size (width x height). 
    - They should be georeferenced. 
    - They could contain NaN values, nodata values (substitute values for nan). 
    - They could be multiband. In this case just add the same name file with *.txt extension. Inside each line equals name of a band. If error, band names will be generated automatically (band 1, band 2, etc.)
    - rasters could be binary [0, 1].
-   Deposits.shp - file containing deposits points or multipoints
    -  it must has column "Commodity", str. There you assign classes that should be discriminated.
    -  it is optional to have additional columns: 
        -  Label - str, names of points to show; 
        -  Train - str, only two values: Train, Test - specify which points will be in Train subset, which in Test. You could generate new split anytime; 
    -  should be the same CRS as rasters, or errors will occur;
    -  If it has multipoints (assume several pits for one deposit): in case of binary feature mode value will be taken for them, in case of real number - mean value;
-  Boundaries.shp - lines to show on a plot. Just for picture.

**The unique feature** of program is the wavelet decomposition preprocessing method. The main idea is in integral nature of geological fields: signals of different scales from most regional provinces to local orebodies are overlayed in geological field. Therefore, wavelets are used as an attempt to separate these signals and find the most valuable for prediction.

**Output**: geotifs with predictions for each class, or binary classifications. Plots.

**HOW TO LAUNCH**
The program is written in Spyder on Python 3.9.10.
All requirements are listed in a file of the same name. But in short you need:
pip libraries: pandas numpy scikit-learn scikit-image PyWavelets PyQt5 pyqt5-tools spyder-kernels matplotlib seaborn pyqtgraph openpyxl
wheel libraries (https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal):
  - GDAL-3.4.3-cp39-cp39-win_amd64.whl
  - pyproj-3.3.1-cp39-cp39-win_amd64.whl
  - Shapely-1.8.2-cp39-cp39-win_amd64.whl
  - Fiona-1.8.21-cp39-cp39-win_amd64.whl

If installing Fiona there will be errors, try to clear hash:
pipenv --clear
pipenv lock
pipenv install <Fiona wheel-file>
  
launch main file - prognoz.py
