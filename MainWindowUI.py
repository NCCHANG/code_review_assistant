import sys
from PySide6 import QtCore, QtWidgets, QtGui


class MainWindowUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Analysis Assistant")
        self.setGeometry(100, 100, 900, 800)
        
        # Set stylesheet for modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                border: none;
            }
            QPushButton#chooseFileBtn {
                background-color: #0066ff;
                color: white;
            }
            QPushButton#chooseFileBtn:hover {
                background-color: #0052cc;
            }
            QPushButton#analyzeBtn {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton#analyzeBtn:hover {
                background-color: #bbbbbb;
            }
            QPushButton#clearBtn {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
            }
            QPushButton#clearBtn:hover {
                background-color: #f0f0f0;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
        """)
        
        # Create main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        # Title
        title = QtWidgets.QLabel("Code Analysis Assistant")
        title_font = QtGui.QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QtWidgets.QLabel("Upload or paste your Python code for analysis")
        subtitle_font = QtGui.QFont()
        subtitle_font.setPointSize(10)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #666666;")
        main_layout.addWidget(subtitle)
        
        # Upload section
        upload_label = QtWidgets.QLabel("Upload Python File")
        upload_label_font = QtGui.QFont()
        upload_label_font.setPointSize(10)
        upload_label_font.setBold(True)
        upload_label.setFont(upload_label_font)
        main_layout.addWidget(upload_label)
        
        choose_file_btn = QtWidgets.QPushButton("Choose File")
        choose_file_btn.setObjectName("chooseFileBtn")
        choose_file_btn.setFixedWidth(120)
        choose_file_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        choose_file_btn.clicked.connect(self._choose_file)
        main_layout.addWidget(choose_file_btn)
        
        # OR divider
        or_label = QtWidgets.QLabel("OR")
        or_label.setAlignment(QtCore.Qt.AlignCenter)
        or_label.setStyleSheet("color: #999999; margin: 10px 0px;")
        main_layout.addWidget(or_label)
        
        # Paste code section
        paste_label = QtWidgets.QLabel("Paste Python Code")
        paste_label.setFont(upload_label_font)
        main_layout.addWidget(paste_label)
        
        self.code_input = QtWidgets.QTextEdit()
        self.code_input.setPlaceholderText("# Enter your Python code here")
        self.code_input.setMinimumHeight(150)
        main_layout.addWidget(self.code_input)
        
        # Context section
        context_label = QtWidgets.QLabel("Provide Context/Description")
        context_label.setFont(upload_label_font)
        main_layout.addWidget(context_label)
        
        self.context_input = QtWidgets.QTextEdit()
        self.context_input.setPlaceholderText("Describe what this code is supposed to do")
        self.context_input.setMinimumHeight(100)
        main_layout.addWidget(self.context_input)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.analyze_btn = QtWidgets.QPushButton("Analyze Code")
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyze_btn.clicked.connect(self._analyze_code)
        button_layout.addWidget(self.analyze_btn)
        
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setObjectName("clearBtn")
        clear_btn.setMaximumWidth(100)
        clear_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        clear_btn.clicked.connect(self._clear_fields)
        button_layout.addWidget(clear_btn, alignment=QtCore.Qt.AlignRight)
        
        main_layout.addLayout(button_layout)
        
        # Add stretch at the end
        main_layout.addStretch()
    
    def _choose_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Open Python File", 
            "", 
            "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.code_input.setText(f.read())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not read file: {str(e)}")
    
    def _analyze_code(self):
        code = self.code_input.toPlainText()
        context = self.context_input.toPlainText()
        
        if not code.strip():
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter some code to analyze.")
            return
        
        # TODO: Implement code analysis logic here
        print("Code:", code)
        print("Context:", context)
        QtWidgets.QMessageBox.information(self, "Analysis", "Code analysis started!")
    
    def _clear_fields(self):
        self.code_input.clear()
        self.context_input.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindowUI()
    window.show()
    sys.exit(app.exec())
