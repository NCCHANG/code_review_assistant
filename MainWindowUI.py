"""MainWindowUI: the top-level application window.

Startup flow
------------
1. db.init_db() — ensure schema exists.
2. Show LoginUI dialog — user logs in, registers, or skips (guest mode).
3. Show this main window with two tabs: Analyze and History.

The Analyze tab runs the full analysis pipeline.  If a user is logged in,
results are automatically persisted to the database after each run.
"""

import os
import pathlib
import sys

from PySide6 import QtCore, QtGui, QtWidgets

import db
import CodeAssistant
from AuthManager import AuthManager
from LoginUI import LoginUI
from HistoryUI import HistoryUI
from StaticEngine import StaticEngine

# ── Analyse tab ───────────────────────────────────────────────────────────────

class AnalyzeTab(QtWidgets.QWidget):
    """The main code analysis panel."""

    # Emits (bugginess_list, fix_list, code_str, file_path) after a successful run
    analysis_done = QtCore.Signal(list, list, str, object)

    def __init__(self, code_assistant: CodeAssistant.CodeAssistant, parent=None) -> None:
        super().__init__(parent)
        self.code_assistant = code_assistant
        self._current_file_path: str | None = None
        self._static_engine = StaticEngine()
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # Title
        title = QtWidgets.QLabel("Intelligent Code Review Assistant")
        title_font = QtGui.QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        subtitle = QtWidgets.QLabel("Upload or paste Python code for AI-powered analysis")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        label_font = QtGui.QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)

        # File upload row
        upload_label = QtWidgets.QLabel("Upload Python File")
        upload_label.setFont(label_font)
        root.addWidget(upload_label)

        file_row = QtWidgets.QHBoxLayout()
        choose_btn = QtWidgets.QPushButton("Choose File")
        choose_btn.setObjectName("chooseFileBtn")
        choose_btn.setFixedWidth(120)
        choose_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        choose_btn.clicked.connect(self._choose_file)
        file_row.addWidget(choose_btn)

        self.file_label = QtWidgets.QLabel("No file selected")
        self.file_label.setStyleSheet("color: #94a3b8; font-size: 12px; margin-left: 8px;")
        file_row.addWidget(self.file_label)
        file_row.addStretch()
        root.addLayout(file_row)

        # OR divider
        or_label = QtWidgets.QLabel("OR")
        or_label.setObjectName("orLabel")
        or_label.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(or_label)

        # Code input
        paste_label = QtWidgets.QLabel("Paste Python Code")
        paste_label.setFont(label_font)
        root.addWidget(paste_label)

        self.code_input = QtWidgets.QTextEdit()
        self.code_input.setObjectName("codeInput")
        self.code_input.setPlaceholderText("# Enter your Python code here")
        self.code_input.setMinimumHeight(160)
        self.code_input.setFont(QtGui.QFont("Monospace", 11))
        self.code_input.textChanged.connect(self._update_analyze_btn)
        root.addWidget(self.code_input)

        # Action buttons
        btn_row = QtWidgets.QHBoxLayout()

        self.analyze_btn = QtWidgets.QPushButton("Analyze Code")
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyze_btn.clicked.connect(self._analyze_code)
        self.analyze_btn.setEnabled(False)
        btn_row.addWidget(self.analyze_btn)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setObjectName("clearBtn")
        clear_btn.setMaximumWidth(100)
        clear_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(clear_btn, alignment=QtCore.Qt.AlignRight)
        root.addLayout(btn_row)

        # Results scroll area
        results_label = QtWidgets.QLabel("Analysis Results")
        results_label.setFont(label_font)
        root.addWidget(results_label)

        self.results_scroll = QtWidgets.QScrollArea()
        self.results_scroll.setObjectName("resultsScroll")
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMinimumHeight(300)
        self.results_scroll.setWidget(self._make_placeholder())
        root.addWidget(self.results_scroll, 1)

    def _make_placeholder(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setStyleSheet("background: #f0f2f5;")
        lbl = QtWidgets.QLabel("Results will appear here after analysis…")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("color: #94a3b8; font-size: 13px;")
        lay = QtWidgets.QVBoxLayout(w)
        lay.addStretch()
        lay.addWidget(lbl)
        lay.addStretch()
        return w

    # ── Event handlers ───────────────────────────────────────────────────────

    def _choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Python File", "", "Python Files (*.py);;All Files (*)"
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.code_input.setText(fh.read())
                self._current_file_path = path
                self.file_label.setText(os.path.basename(path))
                self.file_label.setStyleSheet(
                    "color: #1e293b; font-size: 12px; margin-left: 8px;"
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not read file: {exc}")

    def _clear(self) -> None:
        self.code_input.clear()
        self._current_file_path = None
        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet("color: #94a3b8; font-size: 12px; margin-left: 8px;")
        old = self.results_scroll.takeWidget()
        if old:
            old.deleteLater()
        self.results_scroll.setWidget(self._make_placeholder())

    def _update_analyze_btn(self) -> None:
        self.analyze_btn.setEnabled(bool(self.code_input.toPlainText().strip()))

    def _analyze_code(self) -> None:
        code = self.code_input.toPlainText()
        if not code.strip():
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter some code to analyze.")
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing…")
        QtWidgets.QApplication.processEvents()

        try:
            self.code_assistant.process_file_or_input(code)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(exc))
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze Code")
            return

        bugginess, fix_feedback = self.code_assistant.get_analysis_results()
        self._display_results(bugginess, fix_feedback, code)

        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Code")

        self.analysis_done.emit(bugginess, fix_feedback, code, self._current_file_path)

    # ── Results rendering ─────────────────────────────────────────────────────

    def _display_results(self, bugginess: list, fix_feedback: list, code: str) -> None:
        container = QtWidgets.QWidget()
        container.setStyleSheet("background: #f0f2f5;")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Export button in top-right
        export_row = QtWidgets.QHBoxLayout()
        export_row.addStretch()
        export_btn = QtWidgets.QPushButton("  Export Report")
        export_btn.setObjectName("exportBtn")
        export_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        export_btn.clicked.connect(
            lambda: self._export_report(bugginess, fix_feedback, code)
        )
        export_row.addWidget(export_btn)
        layout.addLayout(export_row)

        # Run pylint once for the whole code block
        self._static_engine.run_on_code(code)
        violations = self._static_engine.get_violations()
        print(f"[AnalyzeTab] violations found: {len(violations)}")
        for v in violations:
            print(f"  [{v['severity']}] {v['title']} — {v['description']} (line {v['line_number']})")
        if not violations:
            print("[AnalyzeTab] Static section will NOT render (no violations)")

        # Sort: buggy first (descending confidence), then safe (ascending confidence)
        sorted_bugginess = sorted(bugginess, key=lambda x: (-x[1], -x[2]))

        # Section 1 — LightGBM Classifier
        layout.addWidget(self._build_classifier_section(sorted_bugginess))

        # Section 2 — Static Analysis (only when violations exist)
        if violations:
            layout.addWidget(self._build_static_section(violations))

        # Section 3 — Suggested Code Fixes
        if fix_feedback:
            layout.addWidget(self._build_fixes_section(fix_feedback, bugginess))

        layout.addStretch()

        old = self.results_scroll.takeWidget()
        if old:
            old.deleteLater()
        self.results_scroll.setWidget(container)

    # ── Section builders ──────────────────────────────────────────────────────

    def _section_card(self, title: str, subtitle: str) -> tuple:
        """Return (card_widget, content_layout) with the section header pre-built."""
        card = QtWidgets.QWidget()
        card.setObjectName("sectionCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        t = QtWidgets.QLabel(title)
        tf = QtGui.QFont()
        tf.setPointSize(13)
        tf.setBold(True)
        t.setFont(tf)
        layout.addWidget(t)

        s = QtWidgets.QLabel(subtitle)
        s.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(s)

        layout.addSpacing(4)
        return card, layout

    # ── Classifier section ────────────────────────────────────────────────────

    def _build_classifier_section(self, bugginess: list) -> QtWidgets.QWidget:
        card, layout = self._section_card(
            "LightGBM Classifier — Bug Confidence Scores",
            "Score shows confidence in the predicted classification (Clean or Buggy)",
        )
        for entry in bugginess:
            func_name, is_buggy, confidence, line_no, _, bug_type = entry
            layout.addWidget(
                self._build_function_card(func_name, is_buggy, confidence, line_no, bug_type)
            )
        return card

    def _build_function_card(
        self,
        func_name: str,
        is_buggy: bool,
        confidence: float,
        line_no: int,
        bug_type: str,
    ) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget()
        row_widget.setObjectName("fnCard")
        row_widget.setProperty("buggy", "true" if is_buggy else "false")

        row = QtWidgets.QHBoxLayout(row_widget)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        # Function name
        name_lbl = QtWidgets.QLabel(f"<b>{func_name}()</b>")
        name_lbl.setStyleSheet("font-family: Monospace; font-size: 13px; color: #222222;")
        row.addWidget(name_lbl)

        # Line badge
        line_badge = QtWidgets.QLabel(f"Line {line_no}")
        line_badge.setStyleSheet(
            "background: #f0f0f0; color: #666666; border-radius: 3px; "
            "padding: 2px 7px; font-size: 11px;"
        )
        row.addWidget(line_badge)

        # Bug-type badge (type already resolved by caller via pylint violations)
        if is_buggy:
            badge_style = (
                "background: #ffe0e0; color: #cc0000; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px;"
            )
        else:
            badge_style = (
                "background: #e0f5e0; color: #006600; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px;"
            )
        type_badge = QtWidgets.QLabel(bug_type)
        type_badge.setStyleSheet(badge_style)
        row.addWidget(type_badge)
        row.addStretch()

        # Confidence score chip
        conf_color = "#cc3300" if is_buggy else "#006600"
        conf_bg = "#fff5f5" if is_buggy else "#f5fff5"
        conf_lbl = QtWidgets.QLabel(f"<b>{confidence:.1%}</b>")
        conf_lbl.setStyleSheet(
            f"color: {conf_color}; background: {conf_bg}; font-size: 14px; "
            "border-radius: 4px; padding: 2px 10px; min-width: 54px;"
        )
        conf_lbl.setAlignment(QtCore.Qt.AlignCenter)
        row.addWidget(conf_lbl)

        return row_widget

    # ── Static analysis section ───────────────────────────────────────────────

    def _build_static_section(self, violations: list) -> QtWidgets.QWidget:
        card, layout = self._section_card(
            "Static Analysis — Code Violations",
            "Code smells and style violations detected",
        )
        for v in violations:
            layout.addWidget(self._build_violation_card(v))
        return card

    def _build_violation_card(self, v: dict) -> QtWidgets.QWidget:
        card = QtWidgets.QWidget()
        card.setObjectName("vCard")

        row = QtWidgets.QHBoxLayout(card)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(12)

        severity = v["severity"]
        if severity == "ERROR":
            badge_style = (
                "background: #ffe0e0; color: #cc0000; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
        elif severity == "WARNING":
            badge_style = (
                "background: #fff3e0; color: #cc7700; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
        else:
            badge_style = (
                "background: #e0f0ff; color: #0055cc; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )

        sev_badge = QtWidgets.QLabel(severity)
        sev_badge.setStyleSheet(badge_style)
        sev_badge.setFixedWidth(72)
        sev_badge.setAlignment(QtCore.Qt.AlignCenter)
        row.addWidget(sev_badge, alignment=QtCore.Qt.AlignTop)

        info = QtWidgets.QVBoxLayout()
        info.setSpacing(2)

        t_lbl = QtWidgets.QLabel(f"<b>{v['title']}</b>")
        t_lbl.setStyleSheet("font-size: 13px; color: #333333;")
        info.addWidget(t_lbl)

        d_lbl = QtWidgets.QLabel(v["description"])
        d_lbl.setStyleSheet("color: #666666; font-size: 12px;")
        d_lbl.setWordWrap(True)
        info.addWidget(d_lbl)

        l_lbl = QtWidgets.QLabel(f"Line {v['line_number']}")
        l_lbl.setStyleSheet("color: #999999; font-size: 11px;")
        info.addWidget(l_lbl)

        row.addLayout(info)
        row.addStretch()
        return card

    # ── Fixes section ─────────────────────────────────────────────────────────

    def _build_fixes_section(self, fix_feedback: list, bugginess: list) -> QtWidgets.QWidget:
        outer = QtWidgets.QWidget()
        outer.setStyleSheet("background: #f0f2f5;")
        layout = QtWidgets.QVBoxLayout(outer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hdr = QtWidgets.QWidget()
        hdr.setStyleSheet("background: transparent;")
        hdr_lay = QtWidgets.QVBoxLayout(hdr)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(2)

        t = QtWidgets.QLabel("Suggested Code Fixes (Sequence-to-Sequence)")
        tf = QtGui.QFont()
        tf.setPointSize(13)
        tf.setBold(True)
        t.setFont(tf)
        hdr_lay.addWidget(t)

        s = QtWidgets.QLabel('Fixes for functions labeled as "potentially buggy"')
        s.setStyleSheet("color: #888888; font-size: 12px;")
        hdr_lay.addWidget(s)
        layout.addWidget(hdr)

        line_map = {e[0]: e[3] for e in bugginess}
        code_map = {e[0]: e[4] for e in bugginess}

        for func_name, fixed_code, feedback in fix_feedback:
            layout.addWidget(
                self._build_fix_card(
                    func_name,
                    code_map.get(func_name, ""),
                    fixed_code,
                    feedback,
                    line_map.get(func_name, "?"),
                )
            )
        return outer

    def _build_fix_card(
        self,
        func_name: str,
        orig_code: str,
        fixed_code: str,
        feedback: str | None,
        line_no,
    ) -> QtWidgets.QWidget:
        card = QtWidgets.QWidget()
        card.setObjectName("fixCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title
        title_lbl = QtWidgets.QLabel(f"<b>Fix issues in {func_name}</b>")
        title_lbl.setStyleSheet("font-size: 14px; color: #222222;")
        layout.addWidget(title_lbl)

        line_lbl = QtWidgets.QLabel(f"Starting at line {line_no}")
        line_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(line_lbl)

        # Analysis & Reasoning box
        if feedback:
            reason_box = QtWidgets.QWidget()
            reason_box.setObjectName("reasonBox")
            rb_lay = QtWidgets.QVBoxLayout(reason_box)
            rb_lay.setContentsMargins(12, 8, 12, 8)
            rb_lay.setSpacing(4)

            rb_title = QtWidgets.QLabel("<b>Analysis &amp; Reasoning:</b>")
            rb_title.setStyleSheet("color: #003399; font-size: 12px;")
            rb_lay.addWidget(rb_title)

            rb_text = QtWidgets.QLabel(feedback)
            rb_text.setStyleSheet("color: #003399; font-size: 12px;")
            rb_text.setWordWrap(True)
            rb_lay.addWidget(rb_text)

            layout.addWidget(reason_box)

        # Side-by-side code diff
        diff_row = QtWidgets.QHBoxLayout()
        diff_row.setSpacing(8)

        orig_col = QtWidgets.QVBoxLayout()
        orig_col.setSpacing(4)
        orig_lbl = QtWidgets.QLabel("<b>Original Code</b>")
        orig_lbl.setStyleSheet("font-size: 12px; color: #333333;")
        orig_col.addWidget(orig_lbl)
        orig_edit = QtWidgets.QTextEdit()
        orig_edit.setReadOnly(True)
        orig_edit.setPlainText(orig_code or "(original code not available)")
        orig_edit.setStyleSheet(
            "background: #fff0f0; border: 1px solid #ffcccc; "
            "font-family: Monospace; font-size: 11px; padding: 4px;"
        )
        orig_edit.setFixedHeight(155)
        orig_col.addWidget(orig_edit)
        diff_row.addLayout(orig_col)

        fix_col = QtWidgets.QVBoxLayout()
        fix_col.setSpacing(4)
        fix_lbl = QtWidgets.QLabel("<b>Suggested Fix</b>")
        fix_lbl.setStyleSheet("font-size: 12px; color: #333333;")
        fix_col.addWidget(fix_lbl)
        fix_edit = QtWidgets.QTextEdit()
        fix_edit.setReadOnly(True)
        fix_edit.setPlainText(fixed_code or "(no fix generated)")
        fix_edit.setStyleSheet(
            "background: #f0fff0; border: 1px solid #99dd99; "
            "font-family: Monospace; font-size: 11px; padding: 4px;"
        )
        fix_edit.setFixedHeight(155)
        fix_col.addWidget(fix_edit)
        diff_row.addLayout(fix_col)

        layout.addLayout(diff_row)

        # Copy button
        copy_btn = QtWidgets.QPushButton("  Copy Fixed Code")
        copy_btn.setObjectName("copyBtn")
        copy_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        _fc = fixed_code
        copy_btn.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(_fc or "")
        )
        layout.addWidget(copy_btn, alignment=QtCore.Qt.AlignRight)

        return card

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_report(self, bugginess: list, fix_feedback: list, code: str) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Report", "analysis_report.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        fix_map = {n: (fc, fb) for n, fc, fb in fix_feedback}
        self._static_engine.run_on_code(code)
        violations = self._static_engine.get_violations()

        lines = ["=" * 60, "  CODE REVIEW ANALYSIS REPORT", "=" * 60, ""]
        lines += ["LightGBM Classifier — Bug Confidence Scores", "-" * 60]
        for func_name, is_buggy, confidence, line_no, _, bug_type in bugginess:
            status = f"BUGGY — {bug_type}" if is_buggy else "SAFE"
            lines.append(f"  {func_name}()  Line {line_no}  [{status}]  {confidence:.1%}")

        if violations:
            lines += ["", "Static Analysis — Code Violations", "-" * 60]
            for v in violations:
                lines.append(
                    f"  [{v['severity']}] {v['title']}  (Line {v['line_number']})"
                )
                lines.append(f"         {v['description']}")

        if fix_feedback:
            lines += ["", "Suggested Code Fixes", "-" * 60]
            for func_name, fixed_code, feedback in fix_feedback:
                line_no = next((e[3] for e in bugginess if e[0] == func_name), "?")
                lines += [
                    f"\n  {func_name}()  (Starting at line {line_no})",
                    "  Analysis & Reasoning:",
                    f"  {feedback or 'N/A'}",
                    "",
                    "  Suggested Fix:",
                    fixed_code or "(no fix generated)",
                ]

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            QtWidgets.QMessageBox.information(
                self, "Export Complete", f"Report saved to:\n{path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(exc))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindowUI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        db.init_db()

        self.auth_manager = AuthManager()
        self.code_assistant = CodeAssistant.CodeAssistant()

        self.setWindowTitle("Intelligent Code Review Assistant")
        self.setGeometry(100, 100, 960, 820)
        _qss = pathlib.Path(__file__).parent / "style.qss"
        if _qss.exists():
            self.setStyleSheet(_qss.read_text())

        self._build_ui()
        self._show_login()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar (user info + logout) ─────────────────────────────────────
        self.top_bar = QtWidgets.QWidget()
        self.top_bar.setObjectName("topBar")
        top_layout = QtWidgets.QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 6, 16, 6)

        self.user_label = QtWidgets.QLabel("")
        self.user_label.setObjectName("userLabel")
        top_layout.addWidget(self.user_label)
        top_layout.addStretch()

        self.logout_btn = QtWidgets.QPushButton("Logout")
        self.logout_btn.setObjectName("logoutBtn")
        self.logout_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.logout_btn.clicked.connect(self._logout)
        self.logout_btn.hide()
        top_layout.addWidget(self.logout_btn)

        root.addWidget(self.top_bar)

        # ── Tab widget ────────────────────────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self.analyze_tab = AnalyzeTab(self.code_assistant)
        self.analyze_tab.analysis_done.connect(self._on_analysis_done)
        self.tabs.addTab(self.analyze_tab, "  Analyze  ")

        self.history_tab = HistoryUI(self.auth_manager)
        self.tabs.addTab(self.history_tab, "  History  ")

        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.statusBar().showMessage("Ready")

    # ── Auth flow ────────────────────────────────────────────────────────────

    def _show_login(self) -> None:
        dialog = LoginUI(self.auth_manager, self)
        dialog.login_successful.connect(self._on_login_success)
        dialog.exec()
        self._update_user_ui()

    def _on_login_success(self, _username: str) -> None:
        self._update_user_ui()

    def _logout(self) -> None:
        self.auth_manager.logout()
        self._update_user_ui()
        self.history_tab.refresh()
        self._show_login()

    def _update_user_ui(self) -> None:
        if self.auth_manager.is_logged_in():
            username = self.auth_manager.current_username
            self.setWindowTitle(
                f"Intelligent Code Review Assistant — logged in as: {username}"
            )
            self.user_label.setText(f"Logged in as  {username}")
            self.logout_btn.show()
            self.statusBar().showMessage(f"Logged in as {username}")
        else:
            self.setWindowTitle("Intelligent Code Review Assistant — Guest Mode")
            self.user_label.setText("Guest mode  (history not saved)")
            self.logout_btn.hide()
            self.statusBar().showMessage("Guest mode — log in to save history")

    # ── Tab change ────────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.history_tab:
            self.history_tab.refresh()

    # ── Post-analysis: persist to DB ─────────────────────────────────────────

    def _on_analysis_done(
        self,
        bugginess: list,
        fix_feedback: list,
        code: str,
        file_path: str | None,
    ) -> None:
        if not self.auth_manager.is_logged_in():
            self.statusBar().showMessage(
                "Analysis complete (not saved — guest mode)", 5000
            )
            return

        try:
            self._save_session(bugginess, fix_feedback, code, file_path)
            self.statusBar().showMessage("Analysis complete — results saved to history", 5000)
        except Exception as exc:
            print(f"[MainWindow] Failed to save session: {exc}")
            self.statusBar().showMessage(
                "Analysis complete (warning: could not save to history)", 5000
            )

    def _save_session(
        self,
        bugginess: list,
        fix_feedback: list,
        code: str,
        file_path: str | None,
    ) -> None:
        fix_map = {name: (fixed, fb) for name, fixed, fb in fix_feedback}

        issues = []
        confidences = []

        for entry in bugginess:
            func_name, is_buggy, confidence, line_no, func_source, bug_type = entry
            confidences.append(confidence)

            issue_type = bug_type
            fixed_code = None
            explanation = None

            if is_buggy and func_name in fix_map:
                fixed_code, explanation = fix_map[func_name]

            issues.append({
                "line_number": line_no,
                "issue_type": issue_type,
                "confidence": confidence,
                "original_code": func_source,
                "fixed_code": fixed_code,
                "explanation": explanation,
            })

        overall_score = sum(confidences) / len(confidences) if confidences else 0.0
        line_count = len(code.splitlines())
        file_name = os.path.basename(file_path) if file_path else None

        db.save_analysis_session(
            user_id=self.auth_manager.current_user_id,
            overall_score=overall_score,
            file_name=file_name,
            file_path=file_path,
            line_count=line_count,
            issues=issues,
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindowUI()
    window.show()
    sys.exit(app.exec())
