# Global modules
from PyQt5.QtWidgets import (
                            QMainWindow, QApplication, QFileDialog, 
                            QTreeWidgetItem
                            )

from PyQt5 import QtCore, QtWidgets
import sys
import os

# Local modules
from apply_advanced import ApplyAdvancedWindow
from area_deposits import AreaDeposits
from container import Container
from export_multiple import ExportMultipleWindow
from main_canvas import MainCanvas
from matrices_canvas import MatricesTabWidget
from model_form import ModelForm
from prediction_model import PredictionModel
from text_browser import TextBrowser
from wavelet_dialog import WaveletDialog

class PredictWidget(QMainWindow):
    """
    Main window
    """
    
    def __init__(self):
        QMainWindow.__init__(self)
        
        self.setupUi()
        
        ## Events
        # clicked button
        # Load data
        self.load_data.clicked.connect(self.get_data)
        # Delete
        self.delete_btn.clicked.connect(self.delete_raster)
        # Export
        self.export_btn.clicked.connect(self.export_raster)
        # Export mult
        self.export_multiple_btn.clicked.connect(self.export_multiple_window)
        # Calc wavelet
        self.calc_wavelet.clicked.connect(self.wavelet_dialog)
        # Create model
        self.create_model_btn.clicked.connect(self.create_model_dialog)
        # Simple Apply MLP
        self.apply_model_btn.clicked.connect(self.apply_model)
        # Apply advanced
        self.apply_advanced_btn.clicked.connect(self.apply_advanced_window)
        # About
        self.about_btn.clicked.connect(self.about_dialog)
        
    
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1280, 800)
        self.setMinimumSize(QtCore.QSize(1280, 800))
        self.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")
        
        # layouts
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(250, 700))
        self.groupBox.setMaximumSize(QtCore.QSize(250, 16777215))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        
        # Button Load data
        self.load_data = QtWidgets.QPushButton(self.groupBox)
        self.load_data.setObjectName("load_data")
        self.verticalLayout.addWidget(self.load_data)
        
        # rasters tree
        self.tree_rasters = QtWidgets.QTreeWidget(self.groupBox)
        self.tree_rasters.setHeaderHidden(True)
        self.tree_rasters.setObjectName("tree_rasters")
        self.tree_rasters.header().setVisible(False)
        self.verticalLayout.addWidget(self.tree_rasters)
        
        # predictions tree
        self.tree_predictions = QtWidgets.QTreeWidget(self.groupBox)
        self.tree_predictions.setHeaderHidden(True)
        self.tree_predictions.setObjectName("tree_predictions")
        self.tree_predictions.header().setVisible(False)
        self.tree_predictions.setVisible(False) # первоначально скрываем
        self.verticalLayout.addWidget(self.tree_predictions)
        
        # Delete raster
        self.delete_btn = QtWidgets.QPushButton()
        self.delete_btn.setObjectName("deleteRaster")
        self.delete_btn.setEnabled(False)
        
        # Export raster
        self.export_btn = QtWidgets.QPushButton()
        self.export_btn.setObjectName("exportRaster")
        self.export_btn.setEnabled(False)
        
        # Export multi
        self.export_multiple_btn = QtWidgets.QPushButton()
        self.export_multiple_btn.setObjectName("exportMultiple")
        self.export_multiple_btn.setEnabled(False)
        
        # Layouts for buttons delete and export
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.addWidget(self.delete_btn)
        self.horizontalLayout.addWidget(self.export_btn)
        self.horizontalLayout.addWidget(self.export_multiple_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        
        # Buttoon Calc wavelet
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.calc_wavelet = QtWidgets.QPushButton(self.groupBox)
        self.calc_wavelet.setEnabled(False)
        self.calc_wavelet.setObjectName("calc_wavelet")
        self.verticalLayout.addWidget(self.calc_wavelet)
        
        # Button create model
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.create_model_btn = QtWidgets.QPushButton(self.groupBox)
        self.create_model_btn.setEnabled(False)
        self.create_model_btn.setObjectName("create_model_btn")
        self.verticalLayout.addWidget(self.create_model_btn)
        
        # Button Apply simple MLP model
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.apply_model_btn = QtWidgets.QPushButton(self.groupBox)
        self.apply_model_btn.setEnabled(False)
        self.apply_model_btn.setObjectName("apply_model_btn")
        self.verticalLayout.addWidget(self.apply_model_btn)
        
        # Button Apply advanced
        self.apply_advanced_btn = QtWidgets.QPushButton(self.groupBox)
        self.apply_advanced_btn.setEnabled(False)
        self.apply_advanced_btn.setObjectName("apply_advanced_btn")
        self.verticalLayout.addWidget(self.apply_advanced_btn)
        
        # Button create plot Area vs threashold
        self.area_deposits_btn = QtWidgets.QPushButton(self.groupBox)
        self.area_deposits_btn.setVisible(False)
        self.area_deposits_btn.setObjectName("area_deposits_btn")
        self.verticalLayout.addWidget(self.area_deposits_btn)
        
        # Button About
        spacerItem = QtWidgets.QSpacerItem(20, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.about_btn = QtWidgets.QPushButton(self.groupBox)
        self.about_btn.setObjectName("about_btn")
        self.verticalLayout.addWidget(self.about_btn)
        
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        
        # Main tab widget
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setObjectName("tabWidget")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar()
        self.statusbar.setEnabled(True)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Prognoz 1.02"))
        self.load_data.setText(_translate("MainWindow", "Load data"))
        self.tree_rasters.headerItem().setText(0, _translate("MainWindow", "Grids"))
        self.delete_btn.setText(_translate("MainWindow", "Delete"))
        self.export_btn.setText(_translate("MainWindow", "Export"))
        self.export_multiple_btn.setText(_translate("MainWindow", "Export Mult."))
        
        self.label.setText(_translate("MainWindow", "Data processing"))
        self.calc_wavelet.setText(_translate("MainWindow", "Calculate wavelet derivatives"))
        self.label_2.setText(_translate("MainWindow", "Create prediction model"))
        self.create_model_btn.setText(_translate("MainWindow", "Create model"))
        self.label_3.setText(_translate("MainWindow", "Apply prediction model"))
        self.apply_model_btn.setText(_translate("MainWindow", "Fase apply MLP model to selected"))
        self.apply_advanced_btn.setText(_translate("MainWindow", "Apply advanced"))
        self.area_deposits_btn.setText(_translate("MainWindow", "Plot Area/Deposits vs. prob."))
        self.about_btn.setText(_translate("MainWindow", "About"))
  
    def get_data(self):
        """
        Function to load dataset into program.
        Dialog offers to choose a main folder, that contains sub folders with
        data. Each subfolder - unique area/tenement.
        Use a test dataset to try.
        """
        
        # User dialog to choose a folder
        path = QFileDialog.getExistingDirectory(self, 'Select Folder')   
        if not path:
            return # if "Cancel"
        else:
            self.path = path
        
        # check if has been loaded before
        # if yes, delete old
        if hasattr(self, 'container'):
            delattr(self, 'container')
            self.tree_rasters.selectionModel().clearSelection()
            self.tree_rasters.clear()
            
        if hasattr(self, 'tab_main_plot'):
            self.tabWidget.removeTab(self.tabWidget.indexOf(self.tab_main_plot))
            delattr(self, 'tab_main_plot')
            
        if hasattr(self, 'tab_predict'):
            self.tabWidget.removeTab(self.tabWidget.indexOf(self.tab_predict))
            delattr(self, 'tab_predict')
            
        if hasattr(self, 'tab_matrices'):
            self.tabWidget.removeTab(self.tabWidget.indexOf(self.tab_matrices))
            delattr(self, 'tab_matrices')
        
        # Continue to load data
        self.container = Container(self.path)
        self.build_tree(self.container.sites, 'structure', self.tree_rasters)
        
        # Create Plots tab
        if not hasattr(self, 'tab_main_plot'):
            self.add_tab_main_plot()
        
        # Select first area and raster
        self.tree_rasters.setCurrentItem(self.tree_rasters.topLevelItem(0).child(0))
        
        # Enable buttons
        self.delete_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.export_multiple_btn.setEnabled(True)
        self.calc_wavelet.setEnabled(True)
        self.create_model_btn.setEnabled(True)
    
    def add_tab_main_plot(self):
        """Function adds tab of main canvas after loading data"""
        # Create tab
        self.tab_main_plot = MainCanvas(self.tree_rasters, self.container, 'grids')
        self.tab_main_plot.setEnabled(True)
        self.tab_main_plot.setObjectName("tab_main_plot")
        self.tabWidget.addTab(self.tab_main_plot, "")
        # Rename tab
        _translate = QtCore.QCoreApplication.translate
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_main_plot), 
                                  _translate("MainWindow", "Plots"))
    
    def build_tree(self, sites, structure_name, tree):
        """Function populates selected tree with names of areas and rasters 
        from input structure"""
        tree.clear()
        # Populate with areas names
        for site_key in sites:
            site = sites[site_key]
            if not hasattr(site, structure_name):
                continue
            item_site = QTreeWidgetItem(tree)
            item_site.setText(0, site_key)
            # populate with first order rasters names
            for grid_key in site[structure_name]:
                item_grid = QTreeWidgetItem(item_site)
                item_grid.setText(0, grid_key)
                # populate with deivatives names
                if not isinstance(site[structure_name][grid_key], list):
                    continue
                if len(site[structure_name][grid_key]) == 0:
                    continue
                for deriv_key in site[structure_name][grid_key]:
                    item_derivative = QTreeWidgetItem(item_grid)
                    item_derivative.setText(0, deriv_key)
        
        tree.expandToDepth(0) # expand selected tree
    
    def get_active_tree(self): 
        """Function defines an active tree and structure"""
        if self.tree_rasters.isVisible():
            self.active_tree = self.tree_rasters
            self.active_grids_group = 'grids'
            self.active_structure = 'structure'
        else:
            self.active_tree = self.tree_predictions
            self.active_grids_group = 'predictions'
            self.active_structure = 'predictions_structure'
            
    def get_selected_item(self):
        """Function gets selected item from an active tree"""
        # get active tree
        self.get_active_tree()
        # create list with strings as a path to selected item
        if self.active_tree.selectionModel().hasSelection():
            item = self.active_tree.selectedItems()[0]
            texts = []
            while item is not None:
                texts.append(item.text(0))
                item = item.parent()
            self.selected_item = texts[::-1]
    
    def delete_raster(self):
        """Function deletes selected item from tree, storage and structure"""
        self.get_selected_item()
        lst = self.selected_item
        # check if not an area selected
        if len(lst) == 1:
            return
        
        # loop by name
        for site_key in self.container.sites:
            site = self.container.sites[site_key]
            if not hasattr(site, self.active_grids_group):
                continue
            for grid_key in site[self.active_grids_group].copy():
                if len(lst) == 2 and grid_key == lst[1]: # if raster selected
                    # delete derivatives
                    for item in site[self.active_structure][grid_key].copy():
                        # delete raster from grids dict
                        del site[self.active_grids_group][ grid_key + ' ' + item ]
                        # delete raster from structure
                        site[self.active_structure][grid_key].remove(item) 
                    # delete raster from grids
                    del site[self.active_grids_group][grid_key]
                    # delete raster from structure
                    del site[self.active_structure][grid_key]
                # if selected derivative
                elif len(lst) == 3 and grid_key == lst[1]+' '+lst[2]: 
                    # delete from grids dict
                    del site[self.active_grids_group][ lst[1] + ' ' + lst[2] ]
                    # delete from structure
                    site[self.active_structure][lst[1]].remove(lst[2]) 
        
        self.active_tree.clear()
        # rebuild tree
        self.build_tree(self.container.sites, self.active_structure, self.active_tree)
        # select firts raster
        self.active_tree.setCurrentItem(self.active_tree.topLevelItem(0).child(0))
        
    def export_raster(self):
        """Function exports selected raster"""
        self.get_selected_item()
        lst = self.selected_item
        # check if selected area name, but a rsater or derivative
        if len(lst) == 1:
            return
        # if selected derivative, get fullname
        if len(lst) == 3: # if derivative
            lst = [lst[0]] + [lst[1] + ' ' + lst[2]]
        # select grid
        grid = self.container.sites[lst[0]][self.active_grids_group][lst[1]]
        grid.export() # call export function
    
    def export_multiple_window(self):
        """Function creates window to select multiple rasters and export them"""
        self.get_active_tree()
        self.export_multiple_window = ExportMultipleWindow(self.active_tree)
        self.export_multiple_window.execute.clicked.connect(self.export_multiple)
    
    def export_multiple(self):
        """Function to export multiple selected rastes in a loop"""
        sites = self.export_multiple_window.selected_sites
        rasters = self.export_multiple_window.selected_rasters
        derivatives = self.export_multiple_window.selected_derivatives
        
        # get directory to write files
        folderpath = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folderpath:
            return # if "Cancel"
        
        # Close the window
        self.export_multiple_window.close()
        
        # export individual rsters in a loop
        for i in sites:
            site = self.container.sites.get(i, None)
            if site is None:
                continue
            for j in rasters:    
                for k in derivatives:
                    grid_name = str(j) + ' ' + str(k)
                    grid = site[self.active_grids_group].__getitem__(grid_name)
                    if grid is None:
                        continue
                    fullpath = os.path.join(folderpath, str(i) +' '+ grid_name + '.tif')
                    grid.export(default_path = fullpath)
            
    def wavelet_dialog(self):
        """Function creates window to select rasters and parameters for
        wavelet decomposition"""
        
        self.wave_form = WaveletDialog(self.container.common_major_grids, self.container.lvl_max)
        # Event clicked button "Execute"
        self.wave_form.execute_btn.clicked.connect(self.update_wavelet_derivatives)
    
    def update_wavelet_derivatives(self):
        """Function creates wavelet-derivatives grids"""
        self.container.get_wavelet_derivatives(self.wave_form)
        self.tree_rasters.clear() # clear rasters tree
        # update tree
        self.build_tree(self.container.sites, 'structure', self.tree_rasters)
    
    def create_model_dialog(self):
        """Function creates window to select rasters and teach a classification
        model"""
        # Check if there is at least 1 point in Deposits
        count = 0
        for site_key in self.container.sites:
            if len(self.container.sites[site_key].deposits) > 0:
                   count += 1
        if count == 0:
            return
                
        # Check if summary points table aready exists
        if hasattr(self.container, 'all_deposits_values'):
            delattr(self.container, 'all_deposits_values')
        
        # Create multilayer rasters and extract values at points for every area
        self.container.get_all_deposits_values()
        
        # Create window
        self.model_form = ModelForm(self.container)
        
        # Even clicked "Execute"
        self.model_form.execute_btn.clicked.connect(self.create_model)
    
    def create_model(self):
        """Function create classification models: SVC and MLP"""
        # Classification model creation
        self.container.pred_model = PredictionModel(self.model_form)
        
        # # Classification matrices plots
        # check if matrices have been created before
        if hasattr(self, 'tab_matrices'):
            self.tab_matrices.remove_tabs()
        else:
            self.tab_matrices = MatricesTabWidget()
            self.tab_matrices_idx = self.tabWidget.addTab(self.tab_matrices, 'Matrices')
        
        # Create plots
        self.tab_matrices.make_matrixes(self.container.pred_model)
        
        # Show window with classification scores
        scores = self.container.pred_model.scores
        title = 'Models created'
        text = f'<p><span style=\" font-size:12pt;\">Classification accuracy: <br></br>\
                SVC train: {scores["train_SVC"]:.2f} <br></br>\
                SVC test: {scores["test_SVC"]:.2f} <br></br>\
                MLP train: {scores["train_MLP"]:.2f} <br></br>\
                MLP test: {scores["test_MLP"]:.2f}</span></p>'
        self.results = TextBrowser(title, text)
        
        # Enable buttons apply model
        self.apply_model_btn.setEnabled(True)
        self.apply_advanced_btn.setEnabled(True)
        
        # Close window
        self.model_form.close()
        
        # Event active tab has changed
        self.tabWidget.currentChanged.connect(self.onChange)  
    
    def _apply_model_core_func(self, site, site_key, **kwargs):
        """Function that applies selected model to selected area"""
        
        # Apply models SVC и MLP
        site.predictions = {} # storage of predictions
        meta = site.grids[list(site.grids)[0]]['meta'] # get metadata from the first raster of an area 
        site.predictions, site.predictions_structure = self.container.pred_model.make_prediction(
                                            site.multi_img, 
                                            site.multiband_names,
                                            meta,
                                            site_key,
                                            **kwargs
                                            )
        
        # Create a plot
        if not hasattr(self, 'tab_predict'):
            self.add_prediction_tab()
        else:
            self.tree_predictions.clear()
            self.build_tree(self.container.sites, 'predictions_structure', self.tree_predictions)
    
    def apply_model(self):
        """Function wrapper to apply seelcted model to selected area"""        
        # Get selected area
        self.get_selected_item()
        site_key = self.selected_item[0]
        site = self.container.sites[site_key]
        self._apply_model_core_func(site, site_key)
    
    def apply_advanced_window(self):
        """Function creates window to select areas and parameters to apply
        cllasification models"""
        self.apply_advanced_window = ApplyAdvancedWindow(self.container.sites.keys())
        self.apply_advanced_window.execute.clicked.connect(self.apply_advanced)
    
    def apply_advanced(self):
        """Function makes advanced apply of classification models"""
        # get selected areas
        sites_lst = self.apply_advanced_window.selected_sites
        
        for site_key in sites_lst:
            site = self.container.sites[site_key]
            self._apply_model_core_func(site, site_key, 
                            svc_bool = self.apply_advanced_window.svc_bool, 
                            mlp_bool = self.apply_advanced_window.mlp_bool, 
                            binary_bool = self.apply_advanced_window.binary_bool)
        
        # Check if the window is active and close
        if hasattr(self, 'apply_advanced_window'):
            if not self.apply_advanced_window.isHidden():
                self.apply_advanced_window.close()
        
    def add_prediction_tab(self):
        """Function creates tab Prediction"""
        # populate predicctions tree
        self.build_tree(self.container.sites, 'predictions_structure', self.tree_predictions)
        
        # create tab for predictions plot
        self.tab_predict = MainCanvas(self.tree_predictions, self.container, 'predictions')
        self.tabWidget.addTab(self.tab_predict, 'Prediction')
        
        # Select first predictions grid and plot
        if not hasattr(self, 'active_tree'):
            self.get_active_tree()
        self.tree_predictions.setCurrentItem(self.active_tree.topLevelItem(0).child(0))
        
        # Event area deposits button clicked
        self.area_deposits_btn.clicked.connect(self.make_area_deposits_window)
    
    def onChange(self):
        """Function changes buttons for tabs: Plots-Matrices-Prediction"""
        idx = self.tabWidget.currentIndex()
        current_tab_name = self.tabWidget.tabText(idx)
        if current_tab_name in ['Plots', 'Matrices']:
            self.tree_rasters.setVisible(True)
            self.tree_predictions.setVisible(False)
            self.delete_btn.setVisible(True)
            self.export_btn.setVisible(True)
            self.label.setVisible(True)
            self.calc_wavelet.setVisible(True)
            self.label_2.setVisible(True)
            self.create_model_btn.setVisible(True)
            self.label_3.setVisible(True)
            self.apply_model_btn.setVisible(True)
            self.apply_advanced_btn.setVisible(True)
            self.area_deposits_btn.setVisible(False)
        elif current_tab_name == 'Prediction':
            self.tree_rasters.setVisible(False)
            self.tree_predictions.setVisible(True)
            self.delete_btn.setVisible(True)
            self.export_btn.setVisible(True)
            self.label.setVisible(False)
            self.calc_wavelet.setVisible(False)
            self.label_2.setVisible(False)
            self.create_model_btn.setVisible(False)
            self.label_3.setVisible(False)
            self.apply_model_btn.setVisible(False)
            self.apply_advanced_btn.setVisible(False)
            self.area_deposits_btn.setVisible(True)
    
    def make_area_deposits_window(self):
        # if window already has been created
        if hasattr(self, 'area_deposits_window'):
            if self.area_deposits_window.isVisible():
                return
        
        # create a window
        self.area_deposits_window = AreaDeposits(self.container.sites, self.model_form.deposits_values)
        
    
    def about_dialog(self):
        """Function calls a window with About info"""
        title = 'About'
        text = '<p><span style=\" font-size:12pt;\">Program for prediction new areas perspective for mineral deposits</span></p>\n\
                    <p><span style=\" font-size:12pt;\">Author: Evgenii Sosnin <br></br>\
                    Senior Exploration Geologist</span></p>\n\
                    <p><span style=\" font-size:12pt;\">Please feel free to contact me:  <a href='"'mailto:sosnin.ep@gmail.com'"'>sosnin.ep@gmail.com</a></span></p>'
        self.about = TextBrowser(title, text)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Make text in all QMessageBoxes selectable
    app.setStyleSheet("QMessageBox { messagebox-text-interaction-flags: 5; }")
    
    window = PredictWidget()
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint), 
    window.show()
    window.setWindowFlags(window.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
    window.show()
    
    sys.exit(app.exec_())
