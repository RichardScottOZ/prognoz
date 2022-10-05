from PyQt5.QtWidgets import (QWidget, QTabWidget, QVBoxLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

class MatrixCanvas(QWidget):
    """Класс который создаёт график матрицу-классификации. Нужен для вывода 
    отдельных вкладок-графиков в классе MatricesTabWidget"""
    def __init__(self, yTest, yPred, deposits_kinds, option, parent=None):
        QWidget.__init__(self, parent)
        
        # Создаём фигуру и сопутствующие объекты
        self.figure = plt.figure()
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Создаём области рисования под: график и шкалу
        self.ax = self.figure.add_subplot(111)
        
        # Заполняем основную компоновку
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        
        # Отрисовка графика матрицы
        self.plot_class_matrix(yTest, yPred, deposits_kinds, option)
        
    def plot_class_matrix(self, yTest, yPred, deposits_kinds, option):
        """Функция создаёт график матрицы классификации"""
        
        option_titles = {
            0:'Objects count', 
            1:'Norm. by rows', 
            2:'Norm. by cols',
            }
        normalize = {
            0:None, 
            1:'true', 
            2:'pred'
            }
        
        # for option_title, normalize in fig_options:
        cm = confusion_matrix(yTest, yPred, labels = deposits_kinds, normalize = normalize[option])    
        df_cm = pd.DataFrame(cm.round(2), index = [i for i in deposits_kinds], 
                              columns = [i for i in deposits_kinds])
        # График матрицы
        sns.heatmap(df_cm, linewidths=0.2, linecolor='grey', 
                    annot=True, annot_kws={"size": 18}, 
                    cmap=plt.cm.Blues, fmt='g', cbar=False, ax = self.ax)
        labels = self.ax.get_xticklabels()
        self.ax.set_xticklabels(labels, ha='right', rotation=45, fontsize=18)
        self.ax.set_yticklabels(labels, fontsize=18)
        self.ax.set_xlabel('Predicted label', fontsize=18)
        self.ax.set_ylabel('True label', fontsize=18)
        self.ax.set_title(option_titles[option], fontsize=20)
        
        # Вывод объекта
        self.canvas.draw()
        
    
class MatricesTabWidget(QTabWidget):
    def __init__(self, parent=None):
        QTabWidget.__init__(self, parent)
    
    def make_matrixes(self, pred_model):
        """Функция создаёт шесть вкладок с графиками матриц результатов классификации"""
        # Вводные данные
        yPred_SVC = pred_model.yPred_SVC
        yPred_MLP = pred_model.yPred_MLP
        deposits_kinds = pred_model.Y_unique
        yTest = pred_model.yTest
        
        # Добавляем вкладки
        matrix_1 = MatrixCanvas(yTest, yPred_SVC, deposits_kinds, 0)
        self.addTab(matrix_1, 'SVC (test). Objects')
        matrix_2 = MatrixCanvas(yTest, yPred_SVC, deposits_kinds, 1)
        self.addTab(matrix_2, 'SVC (test). By rows')
        matrix_3 = MatrixCanvas(yTest, yPred_SVC, deposits_kinds, 2)
        self.addTab(matrix_3, 'SVC (test). By cols')
        matrix_4 = MatrixCanvas(yTest, yPred_MLP, deposits_kinds, 0)
        self.addTab(matrix_4, 'MLP (test). Objects')
        matrix_5 = MatrixCanvas(yTest, yPred_MLP, deposits_kinds, 1)
        self.addTab(matrix_5, 'MLP (test). By rows')
        matrix_6 = MatrixCanvas(yTest, yPred_MLP, deposits_kinds, 2)
        self.addTab(matrix_6, 'MLP (test). By cols')
    
    def remove_tabs(self):
        """Функция удаляет все вкладки в виджете"""
        lst = list(range(self.count()))[::-1]
        for i in lst:
            self.removeTab(i)