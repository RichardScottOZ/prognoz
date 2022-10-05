# global modules
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, 
                                                     QPushButton, QCheckBox)
from PyQt5.QtGui import QPainterPath, QFont, QTransform
from PyQt5.QtCore import Qt 
import pyqtgraph as pg
import numpy as np
from collections import namedtuple
import pandas as pd
from numpy import nanmin, nanmax
from itertools import cycle, islice
import warnings

# local modules
from pandas_widget import PandasView

class MainCanvas(QWidget):
    """Class that plot rasters (input or predictions)"""
    def __init__(self, treeWidget, container, grids_group, parent=None):
        QWidget.__init__(self, parent)
        
        # input data
        self.treeWidget = treeWidget
        self.container = container
        self.grids_group = grids_group
        
        # variable to define current area
        # if no change, we only update title and data
        self.current_site_plot = ''
        
        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Base configs
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.setConfigOption('background', (249,249,249))
        pg.setConfigOption('foreground', 'k')
        
        # Title
        self.title = QLabel('Plot')
        self.title.setAlignment(Qt.AlignCenter)
        
        # Create button to show points labels as a list
        self.show_labels_btn = QPushButton('Labels table')
        self.show_labels_btn.setEnabled(False)
        self.show_labels_btn.setChecked(False)
        
        # Create button to show grid lines
        self.show_grid_lines_btn = QCheckBox('Grid lines')
        self.show_grid_lines_btn.setChecked(False)
        
        # Aux layout
        self.hbox = QHBoxLayout()
        self.hbox.setAlignment(Qt.AlignLeft)
        self.hbox.addWidget(self.show_labels_btn)
        self.hbox.addWidget(self.show_grid_lines_btn)
        
        # Plot layout
        self.layout_plot = QVBoxLayout()
        
        # add widgets to Main layout
        self.layout.addLayout(self.hbox)
        self.layout.addWidget(self.title)
        self.layout.addLayout(self.layout_plot)
        
        # filter warnings (actually there is a warning when you hover over
        # a colorbar)
        warnings.filterwarnings("ignore")
        
        # Events
        # connect treeWidget to MainCanvas
        self.treeWidget.itemSelectionChanged.connect(self.update_plot)
        # show points labels
        self.show_labels_btn.clicked.connect(self.show_labels_window)
        # show grid lines
        self.show_grid_lines_btn.clicked.connect(self.plot_grid_lines)
             
    def get_selected_item(self):
        """Function gets selected item from a tree"""
        if self.treeWidget.selectionModel().hasSelection():
            item = self.treeWidget.selectedItems()[0]
            texts = []
            while item is not None:
                texts.append(item.text(0))
                item = item.parent()
            self.selected_item = texts[::-1]
            
            # if it has a derivative
            if len(self.selected_item) == 3:
                self.selected_item = [self.selected_item[0]] +\
                        [self.selected_item[1] + ' ' + self.selected_item[2]]
    
    def update_plot(self):
        """Function that create or updates a plot"""
        self.get_selected_item()
        # check if only an area has been selected
        if len(self.selected_item) < 2:
            return
        # Title
        self.title_str = str.join(" ", self.selected_item)
        self.title.setText(self.title_str)
        # Create a plot
        # if selected area doesn't equals active area (already plotted),
        # we plot a new one, if does - just update data and title
        if self.current_site_plot != self.selected_item[0]:
            # clean axes
            for i in range(self.layout_plot.count()): 
                self.layout_plot.itemAt(i).widget().close()
            # canvas
            self.win = pg.GraphicsLayoutWidget()
            self.layout_plot.addWidget(self.win)
            # plot objects
            self.plot_grid()
            self.plot_boundaries()
            self.plot_points()
            # labels
            site = self.container.sites[self.selected_item[0]]
            labels = site.labels
            self.plot_labels(labels)
            # grid lines
            self.plot_grid_lines()
            # show plot
            self.win.show()
            # update active area
            self.current_site_plot = self.selected_item[0]
            # enable buttons
            self.show_labels_btn.setEnabled(True)
            # Event - legend object is clicked
            self.win.scene().sigMouseClicked.connect(self.onClick)
        else:
            self.get_img() # extract img from storage
            self.bar.close() # close a colorbar
            self.item.setImage(self.img, autoRange=False)
            self.update_grid_colorbar()

    def onClick(self, event):
        """Function update plot if objects/labes in a legend have been clicked.
        """
        items = self.win.scene().items(event.scenePos())
        group = [x.text for x in items if isinstance(x, pg.LabelItem)]
        # check if item clicked
        if len(group) == 0:
            return
        # check if selected group exists in table
        if group[0] not in set(self.scatter_labels):
            return
        # change visibility of selected element
        element = self.scatter_labels[group[0]]
        if element.isVisible():
            element.setVisible(False)
        else:
            element.setVisible(True)
    
    def get_img(self):
        """Function gets selected img from storage"""
        # Short link on an area
        site = self.container.sites[self.selected_item[0]]
        if not hasattr(site, self.grids_group):
            return
        
        # Select grid and check if it hasn't been deleted
        try:
            grid = site[self.grids_group][self.selected_item[1]]
            self.img = np.flip(grid.img, axis=0)
            self.meta = grid.meta
        except KeyError:
            # object has been deleted
            return
        
    def update_grid_colorbar(self):
        """Function creates colorbar and insert it into a plot canvas"""
        # Colorbar
        # pos, rgba_colors = zip(*cmapToColormap(cm.jet))
        pos = (0.0, 0.11, 0.125, 0.34, 0.35, 0.375, 0.64, 0.65, 0.66, 0.89, 0.91, 1.0)
        rgba_colors = (
            (0, 0, 127, 255),
            (0, 0, 255, 255),
            (0, 0, 255, 255),
            (0, 219, 255, 255),
            (0, 229, 246, 255),
            (20, 255, 226, 255),
            (238, 255, 8, 255),
            (246, 245, 0, 255),
            (255, 236, 0, 255),
            (255, 18, 0, 255),
            (231, 0, 0, 255),
            (127, 0, 0, 255)
                        )
        pgColormap =  pg.ColorMap(pos, rgba_colors)
        
        val1 = nanmin(self.img) # modules nanmin, nanmax from numpy
        val2 = nanmax(self.img)
        if val2 == self.meta['nodata']: # check Nodata attribute presence
            val2 = np.unique(self.img)[-2] # select the second max value
        self.values = (val1, val2)
            
        # cm = pg.colormap.get('plasma')
        self.bar = pg.ColorBarItem( values= (val1, val2), colorMap = pgColormap)
        self.bar.setImageItem( self.item, insert_in=self.plot1)
        
    def plot_grid(self):
        """Function plots a grid"""
        # Extract a grid
        self.get_img()
        # Canvas (ViewBox + axes)
        self.plot1 = self.win.addPlot()
        # Insert image
        if not hasattr(self, 'img'): # check if image exists
            return
        self.item = pg.ImageItem(image = self.img, autoRange=False)
        self.plot1.addItem(self.item)
        # Create a colorbar
        self.update_grid_colorbar()
        
    def get_transform_params(self):    
        """Function gets XY transformation parameters"""
        self.zero_x = self.meta['xx'].min()
        self.zero_y = self.meta['yy'].min()
        self.x_res = self.meta['xres']
        self.y_res = self.meta['yres']*-1
        
    def plot_boundaries(self):
        """Function plots boundaries lines"""
        site = self.container.sites[self.selected_item[0]]
        boundaries = site.boundaries
        
        # get XY transformation params
        self.get_transform_params()
        
        # plot lines
        if len(boundaries)  == 0:
            return
        
        for line in boundaries:
            line_x_img = (np.array(line[0]) - self.zero_x) / self.x_res
            line_y_img = (np.array(line[1]) - self.zero_y) / self.y_res
            self.plot1.addItem(
                pg.PlotCurveItem(line_x_img, line_y_img, pen=pg.mkPen('k', width=1))
                               )  
    
    def plot_points(self):
        """Function plots points"""
        site = self.container.sites[self.selected_item[0]]
        deposits = site.deposits
        
        if len(deposits) == 0:
            return
        
        # Add legend
        self.legend = self.plot1.addLegend(verSpacing=-1)
        self.legend.setPen(pg.mkPen('k',width=1.5))
        self.legend.setBrush(pg.mkBrush(255,255,255,200))
        
        commodities = deposits['Commodity'].unique()
        commodities.sort()
        for i, commodity in enumerate(commodities):
            subset = deposits.loc[deposits['Commodity'] == commodity,:]
            points_x = (subset.x.to_numpy() - self.zero_x) / self.x_res
            points_y = (subset.y.to_numpy() - self.zero_y) / self.y_res
            scatter = pg.ScatterPlotItem(
                points_x, 
                points_y, 
                size = 8, 
                brush=pg.mkBrush(pg.intColor(i))
                                            )
            self.plot1.addItem(scatter)
            self.legend.addItem(scatter, name = commodity)
    
    def plot_labels(self, labels):    
        """Function plots labels of points"""
        if len(labels) == 0:
            return
        
        # Points labels
        # text symbols
        TextSymbol = namedtuple("TextSymbol", "label symbol commodity")
        # method for creating label
        def createSymbol(df_row):
            """Function creates namedtuple with three keys:
                label - str, index of point in a Dataframe
                symbol - QPainterPath - low-level graph object
                commodity - class, if visibility should be set to zero.
                
                Imput - str DataFrame site.labels"""
            # crate point label (as a string) from its index
            label = str(df_row.name)
            # QPainterPath
            symbol = QPainterPath() 
            # creating QFont object
            f = QFont()
            # setting font
            f.setFamily('Arial')
            f.setBold(True)
            # adding text
            symbol.addText(0, 0, f, label)
            # returning text symbol
            # getting bounding rectangle
            br = symbol.boundingRect()
            # getting scale
            scale = min(1. / br.width(), 1. / br.height())
            # getting transform object
            tr = QTransform()
            # setting scale to transform object
            tr.scale(scale/2.5, scale/2.5)
            # translating
            # tr.translate(-br.x() - br.width() / 2., -br.y() - br.height() / 2.)
            tr.translate(br.width() * 0.2, - br.height() * 0.2)
            return TextSymbol(label, tr.map(symbol), df_row.Commodity)
        
        def createSpots(labels, zero_x, zero_y, x_res, y_res):
            """Function creates named tuple"""
            # transform XY real to raster
            labels_x = (labels.x.to_numpy() - zero_x) / x_res
            labels_y = (labels.y.to_numpy() - zero_y) / y_res
            pos = np.array([labels_x, labels_y])
            # annotations
            annos = [createSymbol(labels.loc[i]) for i in labels.index]
            # spots
            spots = [
                        {
                        'pos': pos[:,i], 
                        'data': anno.commodity, 
                        'brush': pg.mkBrush('k'),
                        'pen': pg.mkPen('w', width=1),
                        'symbol': anno[1],
                        'size': 35
                        } for i, anno in enumerate(annos)
                    ]
            return spots
 
        # adding points to the scatter plot
        self.scatter_labels = {}
        for group in set(labels.Commodity):
            spots = createSpots(labels.loc[labels['Commodity'] == group, :], 
                                self.zero_x, self.zero_y, self.x_res, self.y_res)
            self.scatter_labels[group] = pg.ScatterPlotItem(spots)
            self.plot1.addItem(self.scatter_labels[group])
    
    def plot_grid_lines(self):
        """Function plots grid lines"""
        if self.show_grid_lines_btn.isChecked() and hasattr(self, 'meta'):
            # Transform XY real to raster
            xx = (self.meta['xx']- self.zero_x) / self.x_res
            yy = (self.meta['yy'] - self.zero_y) / self.y_res
            # vertical lines
            ymin = np.min(yy)
            ymax = np.max(yy)
            v_xs = list(np.repeat(xx,2,axis=0))
            v_y = [ymin, ymax, ymax, ymin]
            v_ys = list(islice(cycle(v_y), len(v_xs)))
            vline = [v_xs, v_ys]
            
            # horizontal lines
            xmin = np.min(xx)
            xmax = np.max(xx)
            h_ys = list(np.repeat(yy,2,axis=0))
            h_x = [xmin, xmax, xmax, xmin]
            h_xs = list(islice(cycle(h_x), len(h_ys)))
            hline = [h_xs, h_ys]
            
            # all lines
            self.vline = pg.PlotCurveItem(vline[0], vline[1], 
                                          pen = pg.mkPen('k', width=0.2))
            self.hline = pg.PlotCurveItem(hline[0], hline[1], 
                                          pen = pg.mkPen('k', width=0.2))
            # plot lines
            self.plot1.addItem(self.vline)
            self.plot1.addItem(self.hline)
        # update lines
        elif not self.show_grid_lines_btn.isChecked() and hasattr(self, 'vline'):
            self.plot1.removeItem(self.vline)
            self.plot1.removeItem(self.hline)
            delattr(self, 'vline')
            delattr(self, 'hline')
            
    def show_labels_window(self):
        """Function shows list of points labels"""
        # close the window if it is opened
        if hasattr(self, 'labels_window'):
            self.labels_window.close()
        
        # prepare a table
        site = self.container.sites[self.selected_item[0]]
        labels = site.labels
        if len(labels) == 0:
            return
        data = pd.DataFrame({'Index':labels.index, 'Name':labels.Label, 
                             'Commodity':labels.Commodity})
        title = 'Abbreviations of deposits labels'
        # show new window
        self.labels_window = PandasView(title, data)