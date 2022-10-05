from PyQt5.QtWidgets import QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt

class PandasModel(QAbstractTableModel):
    """Class shows Pandas DataFrame in a new window as table"""
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class PandasView(QTableView):
    """
    Class that shows Pandas DataFrame.
    
    Input:
        title - str, window title
        data - Pandas DataFrame
    """
    def __init__(self, title, data):
        super().__init__()
        self.setWindowTitle(title)
        model = PandasModel(data)
        self.setModel(model)
        self.resize(800, 800)
        self.show()