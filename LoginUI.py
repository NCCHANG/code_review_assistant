"""LoginUI: PySide6 QDialog shown on app startup.

Two tabs — Login and Register — plus a "Skip / Continue as Guest" button.
Emits ``login_successful(username: str)`` on success; ``reject()`` for guest.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from AuthManager import AuthManager

# ── Shared stylesheet ────────────────────────────────────────────────────────

_STYLE = """
QDialog {
    background-color: #f5f5f5;
}
QLabel {
    color: #333333;
}
QLabel#errorLabel {
    color: #cc0000;
    font-size: 12px;
}
QLineEdit {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 8px;
    background-color: white;
    font-size: 13px;
    color: #333333;
}
QLineEdit:focus {
    border: 1px solid #0066ff;
}
QPushButton#primaryBtn {
    background-color: #0066ff;
    color: white;
    border-radius: 4px;
    padding: 10px 16px;
    font-weight: bold;
    font-size: 13px;
    border: none;
}
QPushButton#primaryBtn:hover {
    background-color: #0052cc;
}
QPushButton#primaryBtn:disabled {
    background-color: #cccccc;
    color: #666666;
}
QPushButton#guestBtn {
    background-color: transparent;
    color: #666666;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 12px;
}
QPushButton#guestBtn:hover {
    background-color: #f0f0f0;
}
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background: #f5f5f5;
}
QTabBar::tab {
    background: #e8e8e8;
    color: #666666;
    padding: 8px 28px;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #f5f5f5;
    color: #0066ff;
    font-weight: bold;
}
"""


class LoginUI(QtWidgets.QDialog):
    """Authentication dialog shown before the main window."""

    login_successful = QtCore.Signal(str)  # emits the username

    def __init__(self, auth_manager: AuthManager, parent=None) -> None:
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.setWindowTitle("Intelligent Code Review Assistant — Sign In")
        self.setFixedSize(440, 500)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )
        self.setStyleSheet(_STYLE)
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(14)

        # Header
        title = QtWidgets.QLabel("Intelligent Code Review Assistant")
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QtWidgets.QLabel("Sign in to save and review your analysis history")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666666; font-size: 12px;")
        root.addWidget(subtitle)

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_login_tab(), "  Login  ")
        self.tabs.addTab(self._build_register_tab(), "  Register  ")

        # Guest / skip button
        guest_btn = QtWidgets.QPushButton("Skip / Continue as Guest")
        guest_btn.setObjectName("guestBtn")
        guest_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        guest_btn.clicked.connect(self.reject)
        root.addWidget(guest_btn)

    def _field_row(self, label_text: str, placeholder: str, masked: bool = False):
        """Return (QLabel, QLineEdit) for a single form field."""
        label = QtWidgets.QLabel(label_text)
        field = QtWidgets.QLineEdit()
        field.setPlaceholderText(placeholder)
        if masked:
            field.setEchoMode(QtWidgets.QLineEdit.Password)
        return label, field

    def _error_label(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel("")
        lbl.setObjectName("errorLabel")
        lbl.setWordWrap(True)
        lbl.setMinimumHeight(18)
        return lbl

    def _build_login_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(8)

        lbl_user, self.login_username = self._field_row("Username", "Enter username")
        layout.addWidget(lbl_user)
        layout.addWidget(self.login_username)

        lbl_pw, self.login_password = self._field_row("Password", "Enter password", masked=True)
        layout.addWidget(lbl_pw)
        layout.addWidget(self.login_password)

        self.login_error = self._error_label()
        layout.addWidget(self.login_error)

        login_btn = QtWidgets.QPushButton("Login")
        login_btn.setObjectName("primaryBtn")
        login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        login_btn.clicked.connect(self._do_login)
        layout.addWidget(login_btn)

        # Allow Enter key to submit
        self.login_username.returnPressed.connect(self._do_login)
        self.login_password.returnPressed.connect(self._do_login)

        layout.addStretch()
        return tab

    def _build_register_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(8)

        lbl_user, self.reg_username = self._field_row("Username", "Min. 3 characters")
        layout.addWidget(lbl_user)
        layout.addWidget(self.reg_username)

        lbl_pw, self.reg_password = self._field_row("Password", "Min. 6 characters", masked=True)
        layout.addWidget(lbl_pw)
        layout.addWidget(self.reg_password)

        lbl_conf, self.reg_confirm = self._field_row(
            "Confirm Password", "Re-enter password", masked=True
        )
        layout.addWidget(lbl_conf)
        layout.addWidget(self.reg_confirm)

        self.reg_error = self._error_label()
        layout.addWidget(self.reg_error)

        reg_btn = QtWidgets.QPushButton("Create Account")
        reg_btn.setObjectName("primaryBtn")
        reg_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        reg_btn.clicked.connect(self._do_register)
        layout.addWidget(reg_btn)

        self.reg_confirm.returnPressed.connect(self._do_register)

        layout.addStretch()
        return tab

    # ── Event handlers ───────────────────────────────────────────────────────

    def _do_login(self) -> None:
        username = self.login_username.text().strip()
        password = self.login_password.text()
        self.login_error.setText("")

        if len(username) < 3:
            self.login_error.setText("Username must be at least 3 characters.")
            return
        if len(password) < 6:
            self.login_error.setText("Password must be at least 6 characters.")
            return

        success, message = self.auth_manager.login(username, password)
        if success:
            self.login_successful.emit(username)
            self.accept()
        else:
            self.login_error.setText(message)

    def _do_register(self) -> None:
        username = self.reg_username.text().strip()
        password = self.reg_password.text()
        confirm = self.reg_confirm.text()
        self.reg_error.setText("")

        if len(username) < 3:
            self.reg_error.setText("Username must be at least 3 characters.")
            return
        if len(password) < 6:
            self.reg_error.setText("Password must be at least 6 characters.")
            return
        if password != confirm:
            self.reg_error.setText("Passwords do not match.")
            return

        success, message = self.auth_manager.register(username, password)
        if success:
            # Auto-login immediately after successful registration
            self.auth_manager.login(username, password)
            self.login_successful.emit(username)
            self.accept()
        else:
            self.reg_error.setText(message)
