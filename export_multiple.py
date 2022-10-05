from PyQt5.QtWidgets import (QWidget, QListWidget, QListWidgetItem, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout)

class ExportMultipleWindow(QWidget):
    """ Class creates window to select which areas and rasters to export."""
    
    def __init__(self, tree):
        super().__init__()
        self.setWindowTitle('Export multiple rasters')
        
        # create structure of an active tree
        def get_structure(tree):
            root = tree.invisibleRootItem()
            root_dict = {}
            
            for i in range(root.childCount()):
                item_first = root.child(i)
                child_name_first = item_first.text(0)
                
                if item_first.childCount() == 0:
                    root_dict[child_name_first] = {}
                    continue
                second_dict = {}
                for j in range(item_first.childCount()):
                    item_second = item_first.child(j)
                    child_name_second = item_second.text(0)
                    
                    if item_second.childCount() == 0:
                        second_dict[child_name_second] = {}
                        continue
                    third_lst = []
                    for k in range(item_second.childCount()):
                        item_third = item_second.child(k)
                        third_lst.append(item_third.text(0))
                    second_dict[child_name_second] = third_lst
                root_dict[child_name_first] = second_dict
                        
            return root_dict
        self.structure = get_structure(tree)
        
        # list of areas
        self.sites_list = QListWidget()
        self.sites_list.setSelectionMode(3) # model ExtendedSelection
        # poplulate list with areas names
        for site_key in self.structure:
            item = QListWidgetItem(site_key)
            self.sites_list.addItem(item)
        
        # list of first-order grids
        self.rasters_list = QListWidget()
        self.rasters_list.setSelectionMode(3) # model ExtendedSelection
        
        # list of derivatives (second-order grids)
        self.derivatives_list = QListWidget()
        self.derivatives_list.setSelectionMode(3) # model ExtendedSelection
        
        # layers
        hbox = QHBoxLayout()
        hbox.addWidget(self.sites_list)
        hbox.addWidget(self.rasters_list)
        hbox.addWidget(self.derivatives_list)
        
        # button Execute
        self.execute = QPushButton('EXPORT')
        
        # main layout
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel('Select areas to apply'), 0)
        vbox.addLayout(hbox, 10)
        vbox.addWidget(self.execute, 0)
        
        # Show window
        self.setLayout(vbox)
        self.show()
        
        # Events
        self.sites_list.selectionModel().selectionChanged.connect(self.populate_lists)
        self.execute.clicked.connect(self.make_apply)
    
    def make_apply(self):
        """Function reads selected sites and rasters"""
        self.selected_sites = [i.text() for i in self.sites_list.selectedItems()]
        self.selected_rasters = [i.text() for i in self.rasters_list.selectedItems()]
        self.selected_derivatives = [i.text() for i in self.derivatives_list.selectedItems()]
    
    def populate_lists(self):
        """Function populates lists of rasters and derivatives"""
        # get selected sites
        selected_sites = [i.text() for i in self.sites_list.selectedItems()]
        
        self.rasters_list.clear()
        self.derivatives_list.clear()
        
        if len(selected_sites) == 0:
            return
        
        rasters = []
        derivatives = []
        for i in selected_sites:
            rasters.extend([*self.structure[i]])
            for j in self.structure[i]:
                derivatives.extend(self.structure[i][j])
        derivatives = set(derivatives)
        rasters = set(rasters)
        
        for raster_key in rasters:
            item = QListWidgetItem(raster_key)
            self.rasters_list.addItem(item)
        
        for derivative_key in derivatives:
            item = QListWidgetItem(derivative_key)
            self.derivatives_list.addItem(item)