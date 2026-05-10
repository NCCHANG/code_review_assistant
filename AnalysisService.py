"""Service layer that sits between the UI and the database.

Owns the transformation of raw analysis results into the structure
expected by db.save_analysis_session, keeping that logic out of the UI.
"""

import os
import db


class AnalysisService:
    def save_session(
        self,
        user_id: int,
        bugginess: list,
        fix_feedback: list,
        code: str,
        file_path: str | None,
    ) -> int:
        """Transform analysis results and persist them.

        Returns the new session_id.
        """
        fix_map = {name: (fixed, fb) for name, fixed, fb in fix_feedback}

        issues = []
        confidences = []

        for entry in bugginess:
            func_name, is_buggy, confidence, line_no, func_source, bug_type = entry
            confidences.append(confidence)

            fixed_code = None
            explanation = None
            if is_buggy and func_name in fix_map:
                fixed_code, explanation = fix_map[func_name]

            issues.append({
                "line_number": line_no,
                "issue_type": bug_type,
                "confidence": confidence,
                "original_code": func_source,
                "fixed_code": fixed_code,
                "explanation": explanation,
            })

        overall_score = sum(confidences) / len(confidences) if confidences else 0.0
        line_count = len(code.splitlines())
        file_name = os.path.basename(file_path) if file_path else None

        return db.save_analysis_session(
            user_id=user_id,
            overall_score=overall_score,
            file_name=file_name,
            file_path=file_path,
            line_count=line_count,
            issues=issues,
        )
