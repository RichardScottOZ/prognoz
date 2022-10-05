# Global modules
from PyQt5.QtWidgets import (
                QWidget, QLabel, QTreeWidget, QTreeWidgetItem, QSlider,
                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy,
                QSplitter, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.artist import setp

# Local modules
from feature_selection import FeatureSelection

class ModelForm(QWidget):
    """
    Class that creates a window, where user chooses:
        - classes of deposits (depended variable)
        - independed variables.
    Features could be rated on a box-plot/bar plot. Either, selected
    with a FeatureSelection tool.
    
    Step 1.
    During __init__ table with points and feature values is imported.
    
    Step 2.
    Check if Train column exists and if it filled correctly (only Train and Test
    values). Otherwise new Train-Test split is made. Finally, new DataFrame 
    self.deposits_to_use is created and used for: boxplot, 
    feature_selection, prediction_model
    
    Step 3. (optional)
    It to change deposits classes selection, self.deposits_to_use is also 
    will be changed, and a box-plot will be updated.
    If make new split, in self.deposits_values will be changed Train column,
    then self.deposits_to_use  will be changed and a plot will be updated.
    
    If to close a Model_Form create a new one from Main,
    random split will be dropped (new split is created).
    
    Step 4. (optional)
    The tool FeatureSelection gets x, y col from a table 
    
    Step 5. (optional)
    If to click button Execute, Main will create instance of PredictionModel 
    class and pass to it ModelForm instance to read parameters.
    """
    
    def __init__(self, container):
        super().__init__()
        
        # Input data
        self.features_lst = container.multiband_names
        self.deposits_values = container.all_deposits_values
        
        # Create dialog window
        self.initUI()
        
        # Check if Train column was correctly populated (it is created anyway) 
        if set(self.deposits_values['Train']) == set(['Train', 'Test']):
            # if correctly, we update self.deposits_to_use
            self.get_deposits_to_use()
        else: # if not, we make new random split
            self.split_subsets()
        
        # Change window size
        self.resize(900,800)
        # Events
        # change slider
        self.slider.valueChanged.connect(self.set_current_slider_value)
        # change selection of feature
        self.tree_grids.itemSelectionChanged.connect(self.select_and_boxplot)
        # change selection of classes
        self.tree_deposits.itemChanged.connect(self.get_deposits_to_use)
        # clicked 'Feature Selection'
        self.feature_selection_btn.clicked.connect(self.feature_selection_window)
        # clicked 'Split Train-Test'
        self.split_subsets_btn.clicked.connect(self.split_subsets)
        # clicked 'To Excel'
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        # clicked 'Execute'
        self.execute_btn.clicked.connect(self.execute)
    
    def initUI(self):
        self.setWindowTitle('Prediction model creation')
        
        # list of rasters
        self.tree_grids = QTreeWidget()
        self.tree_grids.setSelectionMode(3) # model ExtendedSelection
        self.tree_grids.setHeaderHidden(True)
        for item in self.features_lst:
            self.tree_grids.addTopLevelItem(QTreeWidgetItem([item]))
        self.tree_grids.expandToDepth(0)
        self.tree_grids.setMinimumWidth(400)
        self.tree_grids.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                                 QSizePolicy.MinimumExpanding))
        
        # Feature selection tool
        self.feature_selection_btn = QPushButton('Feature Selection')
        # layout for Feature Selection
        select_layout = QHBoxLayout()
        select_layout.addWidget(self.feature_selection_btn)
        
        # Slider for size of Train-Test split
        self.slider = QSlider(tickPosition = QSlider.TicksBelow, orientation = Qt.Horizontal)
        # Set slider params
        self.slider.setMaximum(19)
        self.slider.setMinimum(10)
        self.slider.setValue(12)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        # Slider size
        self.slider.setMinimumWidth(500)
        # Splitter for slider labels
        splitter = QSplitter()
        splitter.setMinimumWidth(500)
        splitter.setMaximumHeight(10)
        splitter.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                           QSizePolicy.Fixed))
        
        # Slider labels
        label_minimum = QLabel('50 %', splitter)
        label_minimum.setAlignment(Qt.AlignLeft)
        self.label_current = QLabel('Selected 60 %', splitter)
        self.label_current.setAlignment(Qt.AlignHCenter)
        label_maximum = QLabel('95 %', splitter)
        label_maximum.setAlignment(Qt.AlignRight)
        
        # list of deposits classes
        self.tree_deposits = QTreeWidget()
        self.tree_deposits.setMinimumWidth(500)
        self.tree_deposits.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                                 QSizePolicy.MinimumExpanding))
        # add columns
        header = QTreeWidgetItem(["Group","N total","N train", "N test"])
        self.tree_deposits.setHeaderItem(header)
        # get groups size
        train_ratio_initial = 0.6
        # populate a tree
        self.populate_tree_deposits(train_ratio_initial)
        
        # Button to split on Train-Test subset
        self.split_subsets_btn = QPushButton('Split to train-test subsets')
        # Button 'To Excel'
        self.export_excel_btn = QPushButton('To Excel')
        self.export_excel_btn.setMaximumWidth(50)
        # aux layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.split_subsets_btn)
        h_layout.addWidget(self.export_excel_btn)
        
        # Plot
        label3 = QLabel('Box-plot (median + IQR)')
        self.boxplot_canvas = BoxplotCanvas()
        self.boxplot_canvas.setMinimumWidth(900)
        self.boxplot_canvas.setMinimumHeight(300)
        self.tree_deposits.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                                 QSizePolicy.MinimumExpanding))
        
        # Button execute
        self.execute_btn = QPushButton('CREATE PREDICTION MODEL')
        
        # Create layout and add widget    
        gridbox = QGridLayout()
        # Top part, left column
        gridbox.addWidget(QLabel('Select rasters'), 0, 0) # tree title
        gridbox.addWidget(self.tree_grids, 1, 0, 4, 1) # rasters tree
        gridbox.addLayout(select_layout, 5, 0)# FeatureSelection btn
        # Top part. Right column
        gridbox.addWidget(QLabel('Deposits groups'), 0, 1) # deposits title
        gridbox.addWidget(self.tree_deposits, 1, 1) # list of deposits
        gridbox.addWidget(QLabel('Select the ratio of training subset'), 2, 1)
        gridbox.addWidget(self.slider, 3, 1) 
        gridbox.addWidget(splitter, 4, 1)
        # 
        gridbox.addLayout(h_layout, 5, 1)
        # Lower part, both columns
        gridbox.addWidget(label3, 6, 0, 1, 2) # Title
        gridbox.addWidget(self.boxplot_canvas, 7, 0, 1, 2) # box plot
        gridbox.addWidget(self.execute_btn, 8, 0, 1, 2) # execute
        
        # Apply layout and show
        self.setLayout(gridbox)
        self.show()
    
    def set_current_slider_value(self):
        """Function updates slider label and Train column"""
        self.slider_value = self.slider.value()*5 # step 5
        self.label_current.setText(f'Selected {self.slider_value} %')
        # Update deposits classes
        train_ratio = self.slider_value/100
        self.populate_tree_deposits(train_ratio)
        self.split_subsets() # split Train-Test
    
    def get_selected_items(self):
        """ Function gets selected rasters. If rasters were selected function
        outputs list of names, if not - list of all names.
        """
        selected_items = []
        if self.tree_grids.selectionModel().hasSelection():
            self.items = self.tree_grids.selectedItems()
            # loop child-parent
            for item in self.items:
                selected_items.append(item.text(0))
        else: # if user hasn't selected any raster
            # select zero item
            root = self.tree_grids.invisibleRootItem()
            # loop all rasters
            raster_count = root.childCount()
            if raster_count>0:
                for i in range(raster_count):
                    raster = root.child(i)
                    selected_items.append([raster.text(0)])
            
        # Output
        self.selected_items = selected_items
        return selected_items
    
    def count_deposits_groups(self, deposits_values, train_ratio):
        """Function counts number of deposits in Train-Test subsets
        in each class"""
        deposits_groups = deposits_values.groupby(['Commodity']).size().\
                                                reset_index(name='Counts_all')
        deposits_groups['Counts_train'] = round(deposits_groups['Counts_all']\
                                                *train_ratio, 0).astype(int)
        deposits_groups['Counts_test'] = deposits_groups['Counts_all'] - \
                                                deposits_groups['Counts_train']
        return deposits_groups
    
    def populate_tree_deposits(self, train_ratio):
        """Function populates deposits tree"""
        self.deposits_groups = self.count_deposits_groups(self.deposits_values, 
                                                          train_ratio)
        self.tree_deposits.clear()
        for i in range(len(self.deposits_groups)):
            fill_row = [str(i) for i in 
                        self.deposits_groups.loc[i,:].values.tolist()
                        ]
            parent = QTreeWidgetItem(self.tree_deposits, fill_row)
            # add checkbox
            if self.deposits_groups.Counts_all[i] <10:
                parent.setCheckState(0, Qt.Unchecked)
            else:
                parent.setCheckState(0, Qt.Checked)
       
    def get_checked_deposits(self):
        """Function gets checked deposits"""
        checked_deposits = []
        for item in self.tree_deposits.findItems("", Qt.MatchContains \
                                                 | Qt.MatchRecursive):
            if item.checkState(0)>0:
                checked_deposits.append( item.text(0) )
        return checked_deposits
    
    def get_deposits_to_use(self):
        """Function creates a subset only of selected deposits classes from 
        self.deposits_values.
        """
        # table subset (select only selected deposits classes)
        rows_idx = self.deposits_values['Commodity'].isin(
                                                self.get_checked_deposits()
                                                        )
        table = self.deposits_values.loc[rows_idx, :]
        self.deposits_to_use = table.sort_values(by=['Commodity'])
        
        # update plot
        self.select_and_boxplot()
    
    def split_subsets(self):
        """Function creates split to Train-Test"""
        # get ratio to split
        test_ratio = 1 - self.slider.value()/20 # step 5%
        
        # split Train-Test
        train, test = train_test_split(self.deposits_values, 
                                test_size = test_ratio, 
                                stratify = self.deposits_values['Commodity'])
        
        # create column Train
        train['Train'] = 'Train' 
        test['Train'] = 'Test'
        
        # create subset with a Train column
        # that subset will be used for: boxplot, feature_selection, 
        # prediction_model
        self.deposits_values = pd.concat([train, test], axis=0)
        self.deposits_values = self.deposits_values.sort_values(by=['Commodity'])
        
        # update deposits table
        self.get_deposits_to_use()
        
    def select_and_boxplot(self):
        """ Function that updates a plot if selection is changed. It plots
        only if selected one raster."""
        selected_raster = self.get_selected_items()
        if len(selected_raster) == 1:
            self.boxplot_canvas.boxplot(self.deposits_to_use, selected_raster)
        
    def check_conditions(self):
        """Function checks if oly one deposits class was selected
        and if number of features <= number of classes. Both are not good
        for classification"""
        # get list of selected deposits classes
        self.checked_deposits = self.get_checked_deposits()
        first_condition = len(self.checked_deposits) < 2
        if first_condition:
            QMessageBox.about(self, 'Warning', 
                              'Selecte less than two deposits groups')
            return first_condition
        
        # check if number of features less that number of deposits classes
        second_condition = len(self.features_lst) <= len(self.checked_deposits)
        if second_condition:
            QMessageBox.about(self, 'Warning', 
                              'Number of variables <= number of deposits groups')
            return second_condition
    
    def feature_selection_window(self):
        """ Function creates FeatureSelection tool window"""
        if not self.check_conditions(): 
            # Check if a window is already opened
            if hasattr(self, 'feature_selection'):
                if not self.feature_selection.isHidden():
                    return
            
            # Prepare input data
            x = self.deposits_to_use.loc[:, self.features_lst] #
            y = self.deposits_to_use.loc[:, ['Commodity']] #
            
            # Create window
            self.feature_selection = FeatureSelection(x, y)
            
            # Event clicked "Execute" inside "feature_selection" window
            self.feature_selection.execute_btn.clicked.connect(self.get_auto_selection)
    
    def get_auto_selection(self):
        """Function gets selected features in FeatureSelection window
        and select only them in rasters tree"""
        
        # get index of selected features from FeatureSelection window
        self.features_auto_selected_df = self.feature_selection.get_selected_idx()
        
        # Make new selection in rasters tree
        # in a loop search tree elements that exist in list of features from
        # FeatureSelection window
        self.tree_grids.selectionModel().clearSelection()
        root = self.tree_grids.invisibleRootItem()
        raster_count = root.childCount()
        for i in range(raster_count):
            raster = root.child(i)
            if raster.text(0) in self.features_auto_selected_df.Name.values:
                raster.setSelected(True)
                
    def getSaveFileName(self, default_filename):
        """Function opens FileDialog to get path to write Excel table"""
        file_filter = 'Excel Files (*.xlsx)'
        filename, _ = QFileDialog.getSaveFileName(
            parent = self, 
            caption = "Export deposits table",
            directory = default_filename,
            filter = file_filter,
            initialFilter = file_filter
            )
        return filename
    
    def export_to_excel(self):
        """Function to export padas DataFrame to Excel table"""
        if hasattr(self, 'deposits_to_use'):
            # Вызываем окно
            default_filename = 'Deposits_table'
            full_filename = self.getSaveFileName(default_filename)
            try:
                self.deposits_to_use.to_excel(full_filename)
            except Exception:
                QMessageBox.about(self,'Error', 'Could not explort the file')
            finally:
                return
    
    def execute(self):
        """Function saves status of selected features and closes 
        feature_selection window"""
        self.selected_rasters = self.get_selected_items()
        
        # Check if window exists and is opened
        if hasattr(self, 'feature_selection'):
            if not self.feature_selection.isHidden():
                self.feature_selection.close()
                        
# =============================================================================
# Box Plot
# =============================================================================
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import seaborn as sns

class BoxplotCanvas(QWidget):
    """Class that creates box plot or barplot of selected feature and
    deposits classes"""
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        # Create figure
        self.figure = plt.figure()
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Delete unnecessary buttons from a toolbar
        unwanted_buttons = ['Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        # Create subplot
        self.ax = self.figure.add_subplot(111)
        
        # Main Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
    
    def boxplot(self, data, selected_raster):
        """Function creates boxplot"""
        # Clear axes
        self.ax.clear()
        self.canvas.draw()
        
        # Plot
        # if feature is not binary [0,1] create box plot
        if not np.array_equal( np.unique(data[selected_raster[0]]), np.array([0,1]) ):
            sns.boxplot(x = 'Commodity', y = selected_raster[0], hue = 'Train', 
                    data = data, width=.6, ax = self.ax
                    ).set(xlabel='Group')
        else: # if feature is binary, create bar plot
            # prepare data
            group_param = ['Commodity', 'Train', selected_raster[0]]
            data_groupped = data.groupby(group_param)[selected_raster[0]].\
                                            size().reset_index(name = 'Count')
            group_param = ['Commodity', 'Train']
            data_groups = data.groupby(group_param)['Тектон_ВП'].\
                                    size().reset_index(name = 'Groups_count')
            for row in data_groupped.iterrows():
                idx = (data_groups['Commodity'] == row[1]['Commodity']) & \
                                    (data_groups['Train'] == row[1]['Train'])
                val = int(data_groups.loc[idx, 'Groups_count'])
                data_groupped.loc[row[0], 'Relative'] = \
                                    data_groupped.loc[row[0], 'Count'] / val
            data_groupped['hue'] = data_groupped['Train'] + '-' + \
                    data_groupped[selected_raster[0]].astype(int).astype(str)
            # bar plot
            sns.barplot(x = 'Commodity', y = 'Relative', hue = 'hue',
                    data = data_groupped, ax = self.ax).\
                    set(xlabel='Group', ylabel = f'Pct. {selected_raster[0]}')
        
        # plot params
        self.ax.legend(fontsize = 16)
        self.ax.xaxis.label.set_size(16)
        self.ax.yaxis.label.set_size(16)
        setp(self.ax.get_xticklabels(), ha = 'right', rotation=45)
        xticks_lst = list(set(data.Commodity))
        xticks_lst.sort()
        self.ax.set_xticklabels(xticks_lst, size = 16)
        
        # show
        self.canvas.draw()


        
        
        
