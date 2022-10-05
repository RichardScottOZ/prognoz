from PyQt5.QtWidgets import (QWidget, QListWidget, QListWidgetItem, QLabel, 
                             QPushButton, QVBoxLayout, QCheckBox)

class ApplyAdvancedWindow(QWidget):
    """Class creates a window, where user selects areas, and choses parameters:
        1. Which model to use: SVC, MLP, or both;
        2. Output or not a binary classification.
    """
        
    def __init__(self, sites):
        super().__init__()
        self.setWindowTitle('Apply model advanced')
        
        # list of areas
        self.sites_list = QListWidget()
        self.sites_list.setSelectionMode(3) # модель ExtendedSelection
        # populate list with areas names
        for site_key in sites:
            item = QListWidgetItem(site_key)
            self.sites_list.addItem(item)
        
        # Parameters
        self.svc_box = QCheckBox('SVC')
        self.svc_box.setChecked(False)
        self.mlp_box = QCheckBox('MLP')
        self.mlp_box.setChecked(True)
        self.binary_out_box = QCheckBox('Include Binary classification')
        self.binary_out_box.setChecked(False)
        
        # Execute button
        self.execute = QPushButton('APPLY')
        
        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel('Select areas to apply'), 0)
        vbox.addWidget(self.sites_list, 10)
        vbox.addWidget(QLabel('Select models to apply'), 0)
        vbox.addWidget(self.svc_box, 0)
        vbox.addWidget(self.mlp_box, 0)
        vbox.addWidget(self.binary_out_box, 0)
        vbox.addWidget(self.execute, 0)
        
        # Show window
        self.setLayout(vbox)
        self.show()
        
        # Event buttton clicked Execute
        self.execute.clicked.connect(self.make_apply)
        
    def make_apply(self):
        """Function get selected areas as a list and reads parameters"""
        items = self.sites_list.selectedItems()
        self.selected_sites = [i.text() for i in items]
        
        # получим параметры предсказания
        self.svc_bool = self.svc_box.isChecked()
        self.mlp_bool = self.mlp_box.isChecked()
        
        if self.svc_bool == False and self.mlp_bool == False:
            self.binary_bool = False
        else:
            self.binary_bool = self.binary_out_box.isChecked()
        
        
        