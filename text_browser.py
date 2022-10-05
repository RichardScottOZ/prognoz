from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

class TextBrowser(QWidget):
    """Class shows any text in a new window"""
    
    def __init__(self, title, text):
        super().__init__()
        # Заголовок окна
        self.setWindowTitle(title)
                
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(text)
        self.text_browser.setOpenExternalLinks(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.text_browser)
        self.setLayout(layout)
        
        self.resize(350, 230)
        self.setStyleSheet("background-color: #f0f0f0; border: 0;")
        self.show()