"""HistoryUI: embeddable panel that shows past analysis sessions.

HistoryUI       – QWidget, embed as a tab in the main window.
HistoryDetailDialog – QDialog, opens when a session row is double-clicked;
                      shows per-function verdict with a red/green code diff.
"""

from PySide6 import QtCore, QtGui, QtWidgets

import db
from AuthManager import AuthManager

# ── Shared stylesheet ────────────────────────────────────────────────────────

_PANEL_STYLE = """
QWidget {
    background-color: #f5f5f5;
}
QLabel {
    color: #333333;
}
QTableWidget {
    background-color: white;
    border: 1px solid #e0e0e0;
    gridline-color: #f0f0f0;
    font-size: 13px;
}
QTableWidget::item {
    padding: 6px;
}
QTableWidget::item:selected {
    background-color: #e6f0ff;
    color: #333333;
}
QTableWidget::item:alternate {
    background-color: #fafafa;
}
QHeaderView::section {
    background-color: #f0f0f0;
    color: #555555;
    padding: 8px;
    border: none;
    font-weight: bold;
}
QPushButton#refreshBtn {
    background-color: #0066ff;
    color: white;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    border: none;
}
QPushButton#refreshBtn:hover {
    background-color: #0052cc;
}
"""

_DETAIL_STYLE = """
QDialog {
    background-color: #f5f5f5;
}
QLabel {
    color: #333333;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    background: white;
}
QTabBar::tab {
    background: #e8e8e8;
    color: #555555;
    padding: 6px 14px;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: white;
    color: #0066ff;
    font-weight: bold;
}
QPushButton {
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
    border: 1px solid #cccccc;
    background-color: white;
    color: #333333;
}
QPushButton:hover {
    background-color: #f0f0f0;
}
"""


# ── Detail dialog ─────────────────────────────────────────────────────────────

class HistoryDetailDialog(QtWidgets.QDialog):
    """Full session detail: per-function verdicts and red/green code diffs."""

    def __init__(self, session_id: int, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_DETAIL_STYLE)
        self.resize(860, 640)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )

        detail = db.get_session_detail(session_id)
        if detail is None:
            self.setWindowTitle("Session not found")
            QtWidgets.QLabel("Session data could not be loaded.", self)
            return

        session = detail["session"]
        file_info = detail["file"] or {}
        issues = detail["issues"]

        date_str = session.get("DateAnalyzed", "N/A")[:19]
        self.setWindowTitle(f"Session #{session_id}  —  {date_str}")
        self._build_ui(session, file_info, issues)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self, session: dict, file_info: dict, issues: list) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ── Summary bar ──────────────────────────────────────────────────────
        score = session.get("OverallScore")
        score_str = f"{score:.1%}" if score is not None else "N/A"
        file_name = file_info.get("FileName") or "Manual Input"
        line_count = file_info.get("LineCount", 0)
        date_str = session.get("DateAnalyzed", "N/A")[:19]

        buggy_count = sum(
            1 for i in issues if (i.get("IssueType") or "Clean") != "Clean"
        )

        summary = QtWidgets.QLabel(
            f"<b>File:</b> {file_name} &nbsp;|&nbsp; "
            f"<b>Lines:</b> {line_count} &nbsp;|&nbsp; "
            f"<b>Functions:</b> {len(issues)} &nbsp;|&nbsp; "
            f"<b>Issues found:</b> {buggy_count} &nbsp;|&nbsp; "
            f"<b>Avg. score:</b> {score_str} &nbsp;|&nbsp; "
            f"<b>Analyzed:</b> {date_str}"
        )
        summary.setWordWrap(True)
        summary.setStyleSheet(
            "background: white; border: 1px solid #e0e0e0; "
            "border-radius: 4px; padding: 8px; font-size: 12px;"
        )
        root.addWidget(summary)

        # ── Issue tabs ────────────────────────────────────────────────────────
        if not issues:
            root.addWidget(QtWidgets.QLabel("No issues recorded for this session."))
        else:
            tabs = QtWidgets.QTabWidget()
            root.addWidget(tabs)
            for idx, issue in enumerate(issues, 1):
                tabs.addTab(self._build_issue_tab(issue), self._tab_label(idx, issue))

        # ── Close button ──────────────────────────────────────────────────────
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

    def _tab_label(self, idx: int, issue: dict) -> str:
        issue_type = issue.get("IssueType") or "Unknown"
        line_no = issue.get("LineNumber") or "?"
        # Truncate long type names for the tab label
        short_type = issue_type[:12] + "…" if len(issue_type) > 12 else issue_type
        return f"#{idx}  L{line_no}: {short_type}"

    def _build_issue_tab(self, issue: dict) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        issue_type = issue.get("IssueType") or "Unknown"
        confidence = issue.get("Confidence")
        conf_str = f"{confidence:.1%}" if confidence is not None else "N/A"
        line_no = issue.get("LineNumber") or "?"

        meta = QtWidgets.QLabel(
            f"<b>Type:</b> {issue_type} &nbsp;|&nbsp; "
            f"<b>Confidence:</b> {conf_str} &nbsp;|&nbsp; "
            f"<b>Line:</b> {line_no}"
        )
        meta.setStyleSheet(
            "background: #f9f9f9; border: 1px solid #ebebeb; "
            "border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(meta)

        original = issue.get("OriginalCode") or ""
        fixed = issue.get("FixedCode") or ""
        explanation = issue.get("Explanation") or ""

        # Code diff pane (only shown for buggy functions that have fixes)
        if original or fixed:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

            orig_pane = self._code_pane(
                "<b>Original (Buggy)</b>",
                original or "(no original code recorded)",
                bg="#fff0f0",
                border="#ffcccc",
            )
            splitter.addWidget(orig_pane)

            fixed_pane = self._code_pane(
                "<b>Fixed Code</b>",
                fixed or "(no fix generated)",
                bg="#f0fff0",
                border="#99dd99",
            )
            splitter.addWidget(fixed_pane)
            layout.addWidget(splitter, stretch=1)
        else:
            # Clean function — show the source
            src_pane = self._code_pane(
                "<b>Source</b>",
                "(source not recorded for clean functions)",
                bg="#f9f9ff",
                border="#e0e0e0",
            )
            layout.addWidget(src_pane, stretch=1)

        # Explanation / feedback
        if explanation:
            exp_label = QtWidgets.QLabel("<b>Explanation:</b>")
            layout.addWidget(exp_label)
            exp_box = QtWidgets.QTextEdit()
            exp_box.setReadOnly(True)
            exp_box.setPlainText(explanation)
            exp_box.setFixedHeight(90)
            exp_box.setStyleSheet(
                "background-color: white; border: 1px solid #e0e0e0; "
                "font-size: 12px; padding: 4px;"
            )
            layout.addWidget(exp_box)

        return widget

    @staticmethod
    def _code_pane(title: str, code: str, bg: str, border: str) -> QtWidgets.QWidget:
        """Return a titled, read-only code pane with the given background colour."""
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 4, 0)
        vbox.setSpacing(4)

        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-size: 12px;")
        vbox.addWidget(lbl)

        editor = QtWidgets.QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(code)
        editor.setStyleSheet(
            f"background-color: {bg}; border: 1px solid {border}; "
            "font-family: monospace; font-size: 12px; padding: 4px;"
        )
        vbox.addWidget(editor)
        return container


# ── History panel (embeds as a tab) ──────────────────────────────────────────

class HistoryUI(QtWidgets.QWidget):
    """Table of past analysis sessions; double-click opens HistoryDetailDialog."""

    def __init__(self, auth_manager: AuthManager, parent=None) -> None:
        super().__init__(parent)
        self.auth_manager = auth_manager
        self._session_ids: list[int] = []
        self.setStyleSheet(_PANEL_STYLE)
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        # Header row
        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Analysis History")
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_row.addWidget(title)
        header_row.addStretch()

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.setObjectName("refreshBtn")
        refresh_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(refresh_btn)
        root.addLayout(header_row)

        # Session table
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Date", "File Name", "Functions Analysed", "Issues Found", "Overall Score"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip("Double-click a row to view session detail")
        self.table.doubleClicked.connect(self._open_detail)
        root.addWidget(self.table)

        # Placeholder label (shown when table is empty)
        self.placeholder = QtWidgets.QLabel("")
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #999999; font-size: 13px; padding: 20px;")
        root.addWidget(self.placeholder)

        self.refresh()

    # ── Data loading ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload sessions from the database."""
        self.table.setRowCount(0)
        self._session_ids = []

        if not self.auth_manager.is_logged_in():
            self.placeholder.setText(
                "You are in guest mode. Log in to view your analysis history."
            )
            self.table.hide()
            self.placeholder.show()
            return

        self.table.show()
        sessions = db.get_user_sessions(self.auth_manager.current_user_id)

        if not sessions:
            self.placeholder.setText(
                "No history yet. Run an analysis to get started."
            )
            self.placeholder.show()
            return

        self.placeholder.hide()
        for s in sessions:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._session_ids.append(s["session_id"])

            date_str = (s.get("date") or "")[:19] or "N/A"
            file_name = s.get("file_name") or "Manual Input"
            func_count = s.get("function_count", 0)
            issue_count = s.get("issue_count", 0)
            score = s.get("overall_score")
            score_str = f"{score:.1%}" if score is not None else "N/A"

            for col, value in enumerate(
                [date_str, file_name, str(func_count), str(issue_count), score_str]
            ):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, col, item)

    # ── Detail dialog ─────────────────────────────────────────────────────────

    def _open_detail(self, index: QtCore.QModelIndex) -> None:
        row = index.row()
        if 0 <= row < len(self._session_ids):
            dialog = HistoryDetailDialog(self._session_ids[row], self)
            dialog.exec()
