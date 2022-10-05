from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class TextEdit(QWidget):
    """Class shows text in a new window"""
    
    def __init__(self, title, text):
        super().__init__()
        # Title
        self.setWindowTitle(title)
        self.text_edit = QTextEdit()
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        
        self.setStyleSheet("background-color: #f0f0f0; border: 0;")
        self.show()
        self.text_edit.textChanged.connect(self.autoResize)
        self.text_edit.setHtml(text)
    
    def autoResize(self):
        height = int(self.text_edit.document().size().height()) + 50
        width = int(self.text_edit.document().size().width()) + 50
        if height > 800:
            height = 800
        if width > 500:
            width = 500
        self.resize(width, height)