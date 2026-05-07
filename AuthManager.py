"""AuthManager: maintains login session state and delegates to db.py.

A single AuthManager instance is passed to every UI component that needs
to know about the current user.
"""

import db


class AuthManager:
    """Thin wrapper over db.py that holds the active login session."""

    def __init__(self) -> None:
        self.current_user_id: int | None = None
        self.current_username: str | None = None

    # ── Auth operations ──────────────────────────────────────────────────────

    def register(self, username: str, password: str) -> tuple[bool, str]:
        """Register a new user. Returns (success, message)."""
        return db.create_user(username, password)

    def login(self, username: str, password: str) -> tuple[bool, str]:
        """Authenticate and store session state. Returns (success, message)."""
        success, user_id = db.authenticate_user(username, password)
        if success:
            self.current_user_id = user_id
            self.current_username = username
            return True, f"Welcome, {username}!"
        return False, "Invalid username or password."

    def logout(self) -> None:
        """Clear the current session."""
        self.current_user_id = None
        self.current_username = None

    # ── State queries ────────────────────────────────────────────────────────

    def is_logged_in(self) -> bool:
        """Return True if a user is currently logged in."""
        return self.current_user_id is not None

    def get_current_user(self) -> dict | None:
        """Return {'user_id': int, 'username': str}, or None when in guest mode."""
        if not self.is_logged_in():
            return None
        return {"user_id": self.current_user_id, "username": self.current_username}
