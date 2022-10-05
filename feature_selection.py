# Global modules
from PyQt5.QtWidgets import (
                QWidget, QListWidget, QListWidgetItem, QLabel, QPushButton, 
                QHBoxLayout, QVBoxLayout, QSizePolicy, QSlider, QSplitter,
                QSpinBox, QDoubleSpinBox, QSpacerItem, QMessageBox, QFileDialog
                )
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# local modules
from text_edit import TextEdit
from pandas_widget import PandasView

class FeatureSelection(QWidget):
    """
    Class that creates window where user selects features for training a
    classification model.
    """
    
    def __init__(self, x, y):
        super().__init__()
        self.setWindowTitle('Semi-automatical feature selection')
        
        # Input data
        # sort cols X for a beatiful view of classification matrix 
        self.cols = x.columns.to_list()
        self.cols.sort()
        self.x = x.loc[:,self.cols]
        # write Y
        self.y = y
        
        # create window
        self.initUI()
        
        # populate list
        self.reset_list() # заполняем список
        
        # Events
        # layout for features list
        self.reset_btn.clicked.connect(self.reset_list)
        self.show_features_lst.clicked.connect(self.show_current_list)
        self.drop_selected_btn.clicked.connect(self.drop_selected)
        # layout for features exclusion
        self.multicol_btn.clicked.connect(self.exclude_high_correlation)
        self.multicol_show.clicked.connect(self.show_correlated_features)
        self.multicol_show_corr.clicked.connect(self.export_correlation_matrix)
        self.select_F_btn.clicked.connect(self.select_f)
        self.select_RFE_btn.clicked.connect(self.select_rfe)
        self.select_active_scores_btn.clicked.connect(self.show_scores)
        # plot layout
        self.slider.valueChanged.connect(self.update_slider_plot)
        self.drop_unnecessary_btn.clicked.connect(self.drop_unnecessary)
        
    def initUI(self):
        """Function creates initial window"""
        
        # Layout for features list
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(3)
        self.reset_btn = QPushButton('Reset list')
        self.show_features_lst = QPushButton('List included/excluded')
        self.drop_selected_btn = QPushButton('Remove selected')
        features_layout = QVBoxLayout()
        features_layout.addWidget(QLabel('Current features'))
        features_layout.addWidget(self.features_list)
        features_layout.addWidget(self.reset_btn)
        features_layout.addWidget(self.drop_selected_btn)
        
        # layout for methods of exclusion features
        # button for multicollinearity
        self.multicol_btn = QPushButton('Exclude r-Pearson >')
        self.r_pearson_spin = QDoubleSpinBox()
        self.r_pearson_spin.setDecimals(1)
        self.r_pearson_spin.setMaximum(0.9)
        self.r_pearson_spin.setMinimum(0.1)
        self.r_pearson_spin.setSingleStep(0.1)
        self.r_pearson_spin.setValue(0.7)
        # auxilary layout for multicolliearity buttons
        multicol_layout = QHBoxLayout()
        multicol_layout.addWidget(self.multicol_btn)
        multicol_layout.addWidget(self.r_pearson_spin)
        # buttons to show included features, corr. matrix
        self.multicol_show = QPushButton('List included/excluded')
        self.multicol_show.setEnabled(False)
        self.multicol_show_corr = QPushButton('corr. matrix to Excel')
        self.multicol_show_corr.setEnabled(False)
        # layout for RFE
        self.RFE_max_features_spin = QSpinBox()
        self.RFE_max_features_spin.setMinimum(1)
        rfe_max_layout = QHBoxLayout()
        rfe_max_layout.addWidget(self.RFE_max_features_spin)
        rfe_max_layout.addWidget(QLabel('max features RFE'))
        # buttons to select model
        self.select_F_btn = QPushButton('F-selection')
        self.select_RFE_btn = QPushButton('RFE-selection')
        # spacer to push buttons up
        spacerItem = QSpacerItem(20, 200, QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        # populate methods layout
        methods_layout = QVBoxLayout()
        methods_layout.addWidget(QLabel('Exclude highly correlated features'))
        methods_layout.addLayout(multicol_layout)
        methods_layout.addWidget(self.multicol_show)
        methods_layout.addWidget(self.multicol_show_corr)
        methods_layout.addWidget(QLabel('F-value'))
        methods_layout.addWidget(self.select_F_btn)
        methods_layout.addWidget(QLabel('RFE'))
        methods_layout.addLayout(rfe_max_layout)
        methods_layout.addWidget(self.select_RFE_btn)
        methods_layout.addItem(spacerItem)
        methods_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create plot objects
        self.figure = plt.figure()
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumWidth(300)
        self.canvas.setMinimumHeight(100)
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                           QSizePolicy.MinimumExpanding))
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Delete excessive buttons from toolbar
        unwanted_buttons = ['Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        # create subplots
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()
        # create slider
        self.slider = QSlider(tickPosition = QSlider.TicksBelow, orientation = Qt.Horizontal)
        n_min = 1
        n_max = len(self.cols)
        self.slider.setMaximum(n_max)
        self.slider.setMinimum(n_min)
        self.slider.setValue(n_min)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        # Enable slider
        self.slider.setEnabled(False)
        # Create splitter for slider labels
        splitter = QSplitter()
        splitter.setMaximumHeight(10)
        splitter.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,
                                           QSizePolicy.Fixed))
        # Slider labels
        label_minimum = QLabel(str(n_min), splitter)
        label_minimum.setAlignment(Qt.AlignLeft)
        self.label_current = QLabel(f'Selected {n_min}', splitter)
        self.label_current.setAlignment(Qt.AlignHCenter)
        self.label_maximum = QLabel(str(n_max), splitter)
        self.label_maximum.setAlignment(Qt.AlignRight)
        
        # Button "Exclude unnecessary"
        self.drop_unnecessary_btn = QPushButton('Exclude unnecessary')
        self.drop_unnecessary_btn.setEnabled(False)
        # Button "Scores"
        self.select_active_scores_btn = QPushButton('Scores')
        # Button "EXECUTE"
        self.execute_btn = QPushButton('SELECT')
        self.execute_btn.setEnabled(False)
        
        # Plot layout
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(QLabel('Select n-best features')) # title
        plot_layout.addWidget(self.slider)
        plot_layout.addWidget(splitter)
        
        # Buttons "Exclude unnecessary" and "Scores" into aux layout
        plot_btns_layout = QHBoxLayout()
        plot_btns_layout.addWidget(self.drop_unnecessary_btn)
        plot_btns_layout.addWidget(self.select_active_scores_btn)
        plot_btns_layout.setContentsMargins(0, 5, 0, 5)
        # aux layout into plot layout
        plot_layout.addLayout(plot_btns_layout)
        
        # add button "EXECUTE"
        plot_layout.addWidget(self.execute_btn)
        
        # Create main layout and populate
        main_layout = QHBoxLayout()
        main_layout.addLayout(features_layout, 2)
        main_layout.addLayout(methods_layout, 0)
        main_layout.addLayout(plot_layout, 5)
        main_layout.setContentsMargins(5,5,5,5)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Show window
        self.show()
    
    def reset_list(self):
        """Function clears current list and populate it with features names.
        """
        self.features_list.clear()
        for i in self.cols:
            self.features_list.addItem(QListWidgetItem(i))
        # update max RFE_max_features
        self.refresh_rfe_max_features_spin()
        # Enable multicol button
        self.multicol_btn.setEnabled(True)
    
    def drop_selected(self):
        """Function erase feature from list"""
        listItems = self.features_list.selectedItems()
        if not listItems: return        
        for item in listItems:
            self.features_list.takeItem(self.features_list.row(item))
    
    def get_items_names(self):
        current_items =  [str(self.features_list.item(i).text()) for i in 
                          range(self.features_list.count())]
        return current_items
    
    def select_from_list(self, index):
        """Function makes selection from idex - list of features names"""
        for i in index:
            matching_items = self.features_list.findItems(i, Qt.MatchExactly)
            for item in matching_items:
                item.setSelected(True)
    
    def show_current_list(self):
        included_items = self.get_items_names()
        excluded_items = [i for i in self.cols if i not in included_items]
        text = ['<span style=\" font-size:12pt; font-weight:600;\">Included features:</span>'] + included_items +\
            ['<span style=\" font-size:12pt; font-weight:600;\">Excluded features:</span>'] +\
                excluded_items
        text = ''.join(['<p>' + item + '</p>' for item in text])
        title = 'List of current features'
        self.correlated_window = TextEdit(title, text)
    
    def refresh_rfe_max_features_spin(self):
        nmax = self.features_list.count()
        self.RFE_max_features_spin.setMaximum(nmax)
        self.RFE_max_features_spin.setValue(nmax)
    
    def exclude_high_correlation(self):
        """
        Function standardize features and exclude highly correlated features.
        It creates a list of features above threashold correlation value.
        First higly correlated feature is left included, others excluded.
        """
        # get current features and make a subset
        current_items = self.get_items_names()
        data = self.x.loc[:, current_items]
        
        # Exclude hugly correlated features
        self.corr_matrix = data.corr()
        self.drop_features = set()
        for i in range(len(self.corr_matrix.columns)):
            for j in range(i):
                if abs(self.corr_matrix.iloc[i, j]) > round(self.r_pearson_spin.value(),1):
                    colname = self.corr_matrix.columns[i]
                    self.drop_features.add(colname)
        # bock multicol button, as it has been done
        self.multicol_btn.setEnabled(False)
        # enable buttons of correlation matrix and list
        self.multicol_show.setEnabled(True)
        self.multicol_show_corr.setEnabled(True)
        # if highly correlated buttons have been found, we drop them
        if len(self.drop_features) > 0:
            self.select_from_list(self.drop_features)
            self.drop_selected()
        # refresh max RFE spin
        self.refresh_rfe_max_features_spin()
    
    def show_correlated_features(self):
        dropped_lst = list(self.drop_features)
        dropped_lst.sort()
        included_lst = self.get_items_names()
        r_pearson = round(self.r_pearson_spin.value(),1)
        text = [f'<span style=\" font-size:12pt; font-weight:600;\">Included features (r-Pearson ≤ {r_pearson}):</span>'] + included_lst +\
            [f'<span style=\" font-size:12pt; font-weight:600;\">Excluded features (r-Pearson > {r_pearson}):</span>'] +\
                dropped_lst
        text = ''.join(['<p>' + item + '</p>' for item in text])
        title = 'List of features'
        self.correlated_window = TextEdit(title, text)
    
    def getSaveFileName(self, default_filename):
        file_filter = 'Excel Files (*.xlsx)'
        filename, _ = QFileDialog.getSaveFileName(
            parent = self, 
            caption = "Export correlation matrix",
            directory = default_filename,
            filter = file_filter,
            initialFilter = file_filter
            )
        return filename
    
    def export_correlation_matrix(self):
        data = self.corr_matrix.round(2)
        # call dialog
        default_filename = 'Correlation matrix'
        full_filename = self.getSaveFileName(default_filename)
        try:
            data.to_excel(full_filename)
        except Exception:
            QMessageBox.about(self,'Error', 'Could not explort the file')
        finally:
            return
    
    def select_f(self):
        """Univariate statistical method. F-value"""
        # get current features and create a subset
        current_items = self.get_items_names()
        data = self.x.loc[:, current_items]
        
        # Enable slider and scores button
        self.slider.setEnabled(True)
        self.select_active_scores_btn.setEnabled(True)
        
        # Perform analysis
        selector = SelectKBest(f_classif, k='all') # create model
        selector.fit(data, self.y['Commodity'].tolist()) # fit model
        scores = selector.scores_ # F-values
        pvalues = selector.pvalues_ #p-values
        
        # Table of results
        self.data_f = pd.DataFrame({'Score':scores, 'Pvalues':pvalues, 
                                    'Name':data.columns})
        self.data_f['Feature_id'] = np.arange(len(scores)) # index of features
        self.data_f = self.data_f.sort_values(by=['Score'], 
                                    ascending=False).reset_index(drop=True)
        self.data_f['N'] = np.arange(len(scores))+1
        
        # define active method and data
        self.active_selection_method = 'F-scores'
        self.active_data = self.data_f
        
        # update max slider value
        self.slider.setMaximum(len(current_items))
        self.label_maximum.setText(str(len(current_items)))
        
        # Show plot
        self.plot()
        
        # Enable buttons
        self.drop_unnecessary_btn.setEnabled(True)
        self.execute_btn.setEnabled(True)
    
    def select_rfe(self):
        """
        Recursuve selection with Logistic Regression.
        """
        
        # Enable slider and scores
        self.slider.setEnabled(True)
        self.select_active_scores_btn.setEnabled(True)
        
        # get a list of models to evaluate
        def get_models(n_min, n_max, n_jobs):
        	models = dict()
        	for i in range(n_min, n_max+1):
        		rfe = RFE(estimator=LogisticRegression(n_jobs=n_jobs, solver='lbfgs', 
                                        max_iter=1000), n_features_to_select=i)
        		model = LogisticRegression(n_jobs=n_jobs, solver='lbfgs', 
                                     max_iter=1000)
        		models[str(i)] = Pipeline(
                    steps=[
                        ('standardscaler', StandardScaler()),
                        ('s',rfe),
                        ('m',model)
                        ]
                                        )
                
        	return models
         
        # evaluate a give model using cross-validation
        def evaluate_model(model, x, y, n_jobs):
         	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
         	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, 
                                  n_jobs = n_jobs, error_score='raise')
         	return scores
        
        # get current features and create a subset
        current_items = self.get_items_names()
        self.x_rfe = self.x.loc[:, current_items]
        self.y_rfe = self.y.values.ravel()
        
        n_min = 1
        n_max = self.RFE_max_features_spin.value()
        n_jobs = -1
        
        # get the models to evaluate
        self.models = get_models(n_min, n_max, n_jobs)
        models = self.models
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
         	scores = evaluate_model(model, self.x_rfe, self.y_rfe, n_jobs)
         	results.append(scores)
         	names.append(name)
         	
        self.data_rfe = {'results':results, 'labels':names}
        
        # define ative method and data
        self.active_selection_method = 'RFE'
        self.active_data = self.data_rfe
        
        # update slider and label
        self.slider.setMaximum(n_max)
        self.label_maximum.setText(str(n_max))
        
        # Show plot
        self.plot()
        
        # Enable buttons
        self.drop_unnecessary_btn.setEnabled(True)
        self.execute_btn.setEnabled(True)
    
    def show_scores(self):
        title = self.active_selection_method
        if self.active_selection_method == 'F-scores':
            data = self.active_data
        elif self.active_selection_method == 'RFE':
            data = self.get_selected_idx()
        self.active_scores_window = PandasView(title, data)
    
    def plot(self):
        """
        Function create a plot depending on active method.
        """
        
        # Clear axes
        self.ax.clear()
        self.ax2.clear()
        self.canvas.draw()
        
        # Plot
        if self.active_selection_method == 'F-scores':
            x = np.array([0.5, self.slider.value() + 0.5])
            # don't call new var as 'y' - we already have that variable
            self.y_top = self.active_data['Score'].max()
            self.fill = self.ax.fill_between(x = x, y1 = self.y_top, 
                                             facecolor = 'red', alpha=0.25)
            # bar plot
            self.ax.bar(self.active_data['N'], self.active_data['Score'], 
                        width=0.2, color = 'blue')
            self.ax.set_ylabel('F-values', fontsize=12)
            self.ax.set_xticks(list(range(1,len(self.active_data)+1,2)))
            # p-values
            self.ax2.get_yaxis().set_visible(True)
            self.ax2.bar(self.active_data['N'], self.active_data['Pvalues'], 
                         width = 0.1, color = 'red')
            self.ax2.set_ylabel('p-values', fontsize=12)
            self.ax2.spines['right'].set_color('red')
            self.ax2.yaxis.label.set_color('red')
            self.ax2.tick_params(axis='y', color='red')
            # title
            self.ax.set_title('F-scores')
            
            # Show plot
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax.set_xlim(left = 0.5)
            self.canvas.draw()
            
        elif self.active_selection_method == 'RFE':
            # hide the second y-axis for F-value 
            self.ax2.spines['right'].set_color('black')
            self.ax2.get_yaxis().set_visible(False)
            
            # area of selection
            x = np.array([0, self.slider.value() + 0.5])
            self.y_top = np.std(self.active_data['results'])*3 +\
                                        np.mean(self.active_data['results'])
            self.fill = self.ax.fill_between(x = x, y1 = self.y_top, 
                                             facecolor = 'red', alpha=0.25)
            # box plot
            self.ax.boxplot(self.active_data['results'], 
                            labels = self.active_data['labels'], showmeans = True)
            self.ax.set_ylabel('Prediction score', fontsize=12)
            # Title
            self.ax.set_title('RFE')
            
            # Show plot
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax.set_xlim(left = 0.5)
            self.canvas.draw()
    
    def update_slider_plot(self):
        self.label_current.setText(f'Selected {self.slider.value()}')
        x = np.array([0.5, self.slider.value() + 0.5])
        self.fill.remove()
        self.fill = self.ax.fill_between(x = x, y1 = self.y_top, facecolor = 'red', alpha=0.25)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax.set_xlim(left = 0.5)
        self.canvas.draw()
        
    
    def drop_unnecessary(self):
        """
        Function get selected on a plot features as a list and drops features 
        out of that list.
        """
        # get selected features
        data = self.get_selected_idx()
        included_items = data.loc[:, 'Name'].to_list()
        included_items.sort()
        
        # create a list for exclusion
        current_items = self.get_items_names()
        exclude_items = [i for i in current_items if i not in included_items]
        
        # select and drop
        self.select_from_list(exclude_items) # select features in a list
        self.drop_selected()
        
        # update RFE spin
        self.refresh_rfe_max_features_spin()
    
    def get_selected_idx(self):
        """Function gets index of selected features depending on an active method"""
        if self.active_selection_method == 'F-scores':
            n = self.slider.value()
            df_sub = self.active_data.loc[:n-1, :]
            return df_sub
        elif self.active_selection_method == 'RFE':
            # get number of features to select
            n = self.slider.value()
            # select RFE model, fit
            rfe = self.models[str(n)].named_steps['s']
            xdata = self.x_rfe
            ydata = self.y_rfe
            rfe.fit(xdata, np.ravel(ydata))
            # get list of features (Score, Name, Id, Selected)
            results = []
            for i in range(xdata.shape[1]):
                results.append([rfe.ranking_[i], list(xdata.columns)[i], i, rfe.support_[i]])
            # get subset
            df = pd.DataFrame(results, columns = ['Score', 'Name', 'Feature_id', 'Selected'])
            df_sub = df[df.Selected == True]
            return df_sub