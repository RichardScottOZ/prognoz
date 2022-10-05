from PyQt5.QtWidgets import (QWidget, QListWidget, QListWidgetItem, QLabel, 
                             QPushButton, QVBoxLayout, QGridLayout, QSlider,
                             QSplitter, QSizePolicy)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import numpy as np

class AreaDeposits(QWidget):
    """ Class creates window for a plot 'Area/Deposits vs threashold probability'
    Steps:    
        1. During init lists of areas with done predictions (non binary), 
        classes of deposits, models, train-test;
        2. User chooses as many areas as wants, but only one class of deposits;
        3. Consequent rasters are extracted from storage, staked into 1D array,
        at the same time points values are extracted at rasters.
        4. A plot is created.
            Axis X - threashold probability 100 to 0%,
            Axis Y - ratio of deposits/areas above threashold value.
    """
    def __init__(self, sites, deposits_values):
        super().__init__()
        self.setWindowTitle('Percentage of area/deposits to threshold probability')
        
        self.sites = sites
        self.deposits_values = deposits_values
        
        # list of areas
        self.sites_list = QListWidget()
        self.sites_list.setSelectionMode(3) # модель ExtendedSelection
        # populate list with areas names, if they contain predictions
        for site_key in sites:
            if not hasattr(sites[site_key], 'predictions'):
                continue
            item = QListWidgetItem(site_key)
            self.sites_list.addItem(item)
        
        # list of grids
        self.grids_list = QListWidget() # only one item could be selected
        # populate list with grids names
        site_key = self.sites_list.item(0).text()
        for key in sites[site_key].predictions_structure['MLP']:
            if key.endswith(' binary'):
                continue
            item = QListWidgetItem(key)
            self.grids_list.addItem(item)
        
        # list of models
        self.models_list = QListWidget()
        self.models_list.addItem(QListWidgetItem('SVC '))
        self.models_list.addItem(QListWidgetItem('MLP '))
        
        # list train-test
        self.train_list = QListWidget()
        self.train_list.setSelectionMode(3) # модель ExtendedSelection (можно выбрать несколько строк)
        self.train_list.addItem(QListWidgetItem('Train'))
        self.train_list.addItem(QListWidgetItem('Test'))
        self.train_list.addItem(QListWidgetItem('Combined'))
        
        # Layout
        hbox = QGridLayout()
        hbox.addWidget(QLabel('Areas'), 0, 0)
        hbox.addWidget(QLabel('Deposits'), 0, 1)
        hbox.addWidget(QLabel('Models'), 0, 2)
        hbox.addWidget(QLabel('Train/Test'), 0, 3)
        hbox.addWidget(self.sites_list, 1, 0)
        hbox.addWidget(self.grids_list, 1, 1)
        hbox.addWidget(self.models_list, 1, 2)
        hbox.addWidget(self.train_list, 1, 3)
        
        # Button Plot
        self.execute_btn = QPushButton('PLOT')
        
        # Create figure, canvas and toolbar
        self.figure = plt.figure()
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Delete exessive buttons from toolbar
        unwanted_buttons = ['Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        # Create subplots
        self.ax1 = self.figure.add_subplot(111)
        
        # Slider for threashold
        self.slider = QSlider(tickPosition = QSlider.TicksBelow, orientation = Qt.Horizontal)
        # Set mix-max, current val and a step.
        n_min = 1
        n_max = 100
        self.slider.setMaximum(n_max)
        self.slider.setMinimum(n_min)
        self.slider.setValue(50)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        # Slider status
        self.slider.setEnabled(False)
        # Create splitter
        splitter = QSplitter()
        splitter.setMaximumHeight(10)
        splitter.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                           QSizePolicy.Fixed))
        # Slider labels 
        label_minimum = QLabel(str(0.5), splitter)
        label_minimum.setAlignment(Qt.AlignLeft)
        self.label_current = QLabel(f'Selected {0.5}', splitter)
        self.label_current.setAlignment(Qt.AlignHCenter)
        self.label_maximum = QLabel(str(1.0), splitter)
        self.label_maximum.setAlignment(Qt.AlignRight)
        
        # Main layout
        vbox = QVBoxLayout()
        vbox.addLayout(hbox, 1)
        vbox.addWidget(self.execute_btn, 1)
        vbox.addWidget(self.canvas, 3)
        vbox.addWidget(self.toolbar, 1)
        vbox.addWidget(self.slider)
        vbox.addWidget(splitter)
        
        # Show window
        self.setLayout(vbox)
        self.show()
        
        # Events
        self.slider.valueChanged.connect(self.set_slider_value)
        self.execute_btn.clicked.connect(self.plot)
    
    def get_vectors(self, sites, deposits):
        """Function prepares vactors for plotting"""
        # get selected areas
        items = self.sites_list.selectedItems()
        selected_sites = [i.text() for i in items]
        
        # get selected grid name
        self.selected_grid = self.grids_list.selectedItems()[0].text()
        
        # get selected model
        selected_model = self.models_list.selectedItems()[0].text()
        
        # make subset of deposits
        data = deposits.loc[ 
            (deposits['Site'].isin(selected_sites)) & 
            (deposits['Commodity'] == self.selected_grid), 
            ['Site', 'Train', 'Row', 'Col']
            ]
        
        # select prediction grids and concatenate into 1D array
        img_1d = np.array([])
        for site_key in selected_sites:
            # select grid
            img = sites[site_key].predictions[selected_model + self.selected_grid].img
            
            # add grid to 1D vector
            img_1d = np.append(img_1d, img[~np.isnan(img)].ravel())
            
            # get point values at ratsers
            rows_idx = data.loc[data['Site'] == site_key, 'Row']
            cols_idx = data.loc[data['Site'] == site_key, 'Col']
            probabilities = np.array(img[rows_idx, cols_idx]).T
            data.loc[data['Site'] == site_key, 'Probability'] = probabilities
        
        # create base vector for threashold value p
        ratios = np.arange(0, np.max(img_1d), 0.01)
        # calculate ratio of area above threashold values p
        tot = len(img_1d)
        area_vec = [np.sum(img_1d >= i)/tot for i in ratios]
        
        # create table of points with probabilities
        deposits_df = data.sort_values(by=['Probability'], ignore_index=True)
        
        return ratios, area_vec, deposits_df
    
    def set_slider_value(self):
        """Function sets the slider value"""
        self.label_current.setText(f'Selected {self.slider.value()/100}')
        self.update_slider_line()
        
    def update_slider_line(self):
        """Function updates the vertical line and labels"""
        slider_val = self.slider.value()/100
        idx = np.where(self.ratios == slider_val)[0]
        if idx.size > 0:
            idx = idx[0]
            x = self.ratios[idx]
            y = self.area_vec[idx]
            self.pline.set_xdata(x)
            self.ptext.set_position((x+0.01, y-0.02))
            self.ptext.set_text(f'p = {slider_val}')
            self.area_text.set_position((x+0.01, y+0.02))
            self.area_text.set_text(f'{y:.1%}')
            # update labels
            for item in ['deposits_ratio', 'deposits_ratio_train', 'deposits_ratio_test']:
                if not item in self.labels_dict.keys():
                    continue
                vec = getattr(self, item)
                y = vec[idx]
                label = self.labels_dict[item]
                label.set_position((x+0.01, y+0.02))
                label.set_text(f'{y:.1%}')
        
        # Show plot
        self.canvas.draw()
    
    def plot(self):
        """Function creates a plot"""
        # Check if every list has a selection
        if not all([self.sites_list.selectionModel().hasSelection(),
                    self.grids_list.selectionModel().hasSelection(),
                    self.models_list.selectionModel().hasSelection(),
                    self.train_list.selectionModel().hasSelection()
                    ]):
            return
        
        # Enable slider
        self.slider.setEnabled(True)
        
        # get vectors for plotting
        sites = self.sites
        deposits = self.deposits_values
        self.ratios, self.area_vec, deposits_df = self.get_vectors(sites, deposits)
        
        # get Train subsets params
        train_selection = [i.text() for i in self.train_list.selectedItems()]
        
        # function to calculate ratio of deposits above threashold
        def calc_deposits_ratio(data, ratios):
            result = []
            tot = len(data)
            for i in ratios:
                ratio = np.sum(data >= i) / tot
                result.append(ratio)
            return result
        
        # plot
        # Clear axes
        self.ax1.clear()
        self.canvas.draw()
        
        # Plot area line
        self.ax1.plot(self.ratios, self.area_vec, color='tab:blue', label = 'Area')
        
        # plot vertical line
        slider_val = self.slider.value()/100
        idx = np.where(self.ratios == slider_val)[0]
        if idx.size > 0:
            idx = idx[0]
            x = self.ratios[idx]
            y = self.area_vec[idx]
            self.pline = self.ax1.axvline(x, ls = '--', lw = 3, c = 'black')
            self.ptext = self.ax1.text(x+0.01, y-0.02, f'p = {slider_val}', fontsize = 'x-large')
            self.area_text = self.ax1.text(x+0.01, y+0.02, f'p = {y:.1%}', c = 'blue', fontsize = 'x-large')
        
        # dict for labels
        self.labels_dict = {}
        
        # Plot deposits
        # both train and test subsets combined
        if 'Combined' in train_selection:
            deposits_data = deposits_df['Probability']
            self.deposits_ratio = calc_deposits_ratio(deposits_data, self.ratios)
            self.ax1.plot(self.ratios, self.deposits_ratio,
                     color='tab:orange', 
                     label = f'Deposits {self.selected_grid}')
            # label
            if idx.size > 0:
                y = self.deposits_ratio[idx]
                self.deposits_text = self.ax1.text(x+0.01, y+0.01, f'{y:.1%}', c = 'orange', fontsize = 'x-large')
                self.labels_dict['deposits_ratio'] = self.deposits_text
        else:
            # train subset
            train_idx = deposits_df['Train'] == 'Train'
            deposits_data = deposits_df.loc[train_idx, 'Probability']
            if 'Train' in train_selection and len(deposits_data) > 0:
                self.deposits_ratio_train = calc_deposits_ratio(deposits_data, self.ratios)
                self.ax1.plot(self.ratios, self.deposits_ratio_train,
                         color='tab:green',
                         linewidth = 3,
                         label = f'Deposits {self.selected_grid} (train)')
                # label
                if idx.size > 0:
                    y = self.deposits_ratio_train[idx]
                    self.train_text = self.ax1.text(x+0.01, y+0.01, f'{y:.1%}', c = 'green', fontsize = 'x-large')
                    self.labels_dict['deposits_ratio_train'] = self.train_text
            
            # test subset 
            test_idx = deposits_df['Train'] == 'Test'
            deposits_data = deposits_df.loc[test_idx, 'Probability']
            if 'Test' in train_selection and len(deposits_data) > 0:
                self.deposits_ratio_test = calc_deposits_ratio(deposits_data, self.ratios)
                self.ax1.plot(self.ratios, self.deposits_ratio_test,
                         color='tab:red', 
                         label = f'Deposits {self.selected_grid} (test)')
                # label
                if idx.size > 0:
                    y = self.deposits_ratio_test[idx]
                    self.test_text = self.ax1.text(x-0.04, y+0.01, f'{y:.1%}', c = 'red', fontsize = 'x-large')
                    self.labels_dict['deposits_ratio_test'] = self.test_text
                
        # axis labels
        self.ax1.set_xlabel('Threshold probability (p)')
        self.ax1.set_ylabel('Percentage of [Area / Deposits] ≥ p')
        self.ax1.xaxis.label.set_size(16)
        self.ax1.yaxis.label.set_size(16)
        
        # grid
        major_ticks = np.arange(0, 1.01, 0.2)
        minor_ticks = np.arange(0, 1.01, 0.05)
        self.ax1.set_xticks(major_ticks)
        self.ax1.set_xticks(minor_ticks, minor=True)
        self.ax1.set_yticks(major_ticks)
        self.ax1.set_yticks(minor_ticks, minor=True)
        self.ax1.grid(which='major', color = 'dimgray', alpha = 0.8)
        self.ax1.grid(which='minor', color='grey', linestyle=':', alpha = 0.8)
        self.ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # legend
        self.ax1.legend(loc = 'upper center', fontsize=16)
        
        # Show plot
        self.canvas.draw()
        
        