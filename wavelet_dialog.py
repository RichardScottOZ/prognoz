from PyQt5.QtWidgets import (QWidget, QLabel, QListWidget, QComboBox, QSpinBox, 
                             QPushButton, QCheckBox, QVBoxLayout)

class WaveletDialog(QWidget):
    """ Class creates window where user selects rasters to decompose
    by wavelets, and choses wavelet parameters"""
    
    def __init__(self, common_major_grids, lvl_max):
        super().__init__()
        # Initial params
        self.common_major_grids = common_major_grids
        self.lvl_max = lvl_max
        
        self.initUI() # Create window
        self.resize(500, 500)
        
        # Event wavelet type change - change max wavelet decomposition
        self.w_box.currentIndexChanged.connect(self.update_w_spin)
        
    def initUI(self):
        """Function creates window"""
        
        # Title
        self.setWindowTitle('Settings for the wavelet transform')
        
        # Rasters list
        label1 = QLabel('Select rasters')
        self.w_list = QListWidget()
        self.w_list.setSelectionMode(3) #model ExtendedSelection
        self.w_list.addItems(self.common_major_grids)
        
        # Selection of wavelet type. By default, 'haar'
        label2 = QLabel('Wavelet type')
        self.w_box = QComboBox()
        self.w_box.addItems(['haar', 'sym7'])
        self.w_box.setCurrentText('haar')
        
        # Set current and max wavelet decomposition levels by default
        label3 = QLabel('Maximum level of the wavelet decomposition')
        self.w_spin = QSpinBox()
        if self.lvl_max['haar'] >=3:
            # reccomendation -2 to max lvl, so there will be significant amount
            # of pixels in a wavelet derivative
            self.w_spin.setValue(self.lvl_max['haar']-2)
        else:
            self.w_spin.setValue(self.lvl_max['haar'])
        self.w_spin.setMaximum(self.lvl_max['haar'])
        self.w_spin.setMinimum(1)
        
        # Moving window size
        label4 = QLabel('Moving window size')
        self.window_spin = QSpinBox()
        self.window_spin.setValue(3)
        self.window_spin.setMaximum(11)
        self.window_spin.setSingleStep(2)
        self.window_spin.setMinimum(3)
        
        # Interpolation type
        label5 = QLabel('Interplolation type')
        self.interpolation_box = QComboBox()
        self.interpolation_box.addItems(['Nearest neighbour', 'Bilinear', 'Cubic'])
        self.interpolation_box.setCurrentIndex(1)
        
        # Types of wavelet products
        # OBLIGATORY to assign ObjectName, because products are found through
        # findChildren(QCheckBox) and send to wavelet decomposition to Site obj. 
        label6 = QLabel('Output products')
        
        self.product_wa = QCheckBox('Approximation')
        self.product_wa.setObjectName('product_wa')
        self.product_wa.setChecked(True)
        
        self.product_wd = QCheckBox('Detailes')
        self.product_wd.setObjectName('product_wd')
        self.product_wd.setChecked(False)
        
        self.product_wd_std = QCheckBox('Moving STD of detailes')
        self.product_wd_std.setObjectName('product_wd_std')
        self.product_wd_std.setChecked(True)
        
        self.product_wd_rel = QCheckBox('Detailes (abs. Z)')
        self.product_wd_rel.setObjectName('product_wd_rel')
        self.product_wd_rel.setChecked(False)
        
        self.product_wd_range = QCheckBox('Min-Max')
        self.product_wd_range.setObjectName('product_wd_range')
        self.product_wd_range.setChecked(False)
        
        # Button "EXECUTE"
        self.execute_btn = QPushButton('EXECUTE')
        
        # Layout
        vbox = QVBoxLayout()
        
        vbox.addWidget(label1)
        vbox.addWidget(self.w_list)
        
        vbox.addWidget(label2)
        vbox.addWidget(self.w_box)
        
        vbox.addWidget(label3)
        vbox.addWidget(self.w_spin)
        
        vbox.addWidget(label4)
        vbox.addWidget(self.window_spin)
        
        vbox.addWidget(label5)
        vbox.addWidget(self.interpolation_box)
        
        vbox.addWidget(label6)
        vbox.addWidget(self.product_wa)
        vbox.addWidget(self.product_wd)
        vbox.addWidget(self.product_wd_std)
        vbox.addWidget(self.product_wd_rel)
        vbox.addWidget(self.product_wd_range)
        
        vbox.addWidget(self.execute_btn)
        
        # Show
        self.setLayout(vbox)
        self.show()
    
    def update_w_spin(self):
        """Function updates current and max wavelet level in a spin box"""
        
        current_key = str(self.w_box.currentText())
        if self.lvl_max[current_key] >=3:
            # -2 reccomended to have singificant amount of pixels in
            # a product
            self.w_spin.setValue(self.lvl_max[current_key]-2)
        else:
            self.w_spin.setValue(self.lvl_max[current_key])
        self.w_spin.setMaximum(self.lvl_max[current_key])
    
    def get_products_status(self):
        """Function gets status of wavelet products: selected or not"""
        products = {i.objectName() : i.isChecked() for i in self.findChildren(QCheckBox)}
        return products