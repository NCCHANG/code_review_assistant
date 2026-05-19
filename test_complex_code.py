
# -------------------------------------------------------------------------
# TEST FILE: Functions aligned to T5 + LightGBM training patterns.
#
# Classifier labels (LightGBM):
#   0 = Clean
#   1 = Wrong Binary Operator  (recall ~75%)
#   3 = Swapped Operand        (recall ~76%)
#
# T5 repair patterns (CTSSB-1M fine-tune):
#   CHANGE_BINARY_OPERATOR  — swap one operator  (~50% EM)
#   CHANGE_BOOLEAN_LITERAL  — True <-> False     (~100% EM)
#   CHANGE_UNARY_OPERATOR   — add/remove not     (~67% EM)
#
# Best pipeline coverage (both models expected correct):
#   Section 2 (Wrong Binary Operator) — strongest overlap.
# -------------------------------------------------------------------------


# =============================================================================
# SECTION 1 — Clean  (classifier label=0, T5 should produce no meaningful change)
# =============================================================================

def calculate_average(numbers):
    # [CLEAN]
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def is_palindrome(s):
    # [CLEAN]
    cleaned = s.lower().strip()
    return cleaned == cleaned[::-1]


def clamp_value(value, low, high):
    # [CLEAN]
    if value < low:
        return low
    if value > high:
        return high
    return value


def binary_search(arr, target):
    # [CLEAN]
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def count_occurrences(items, target):
    # [CLEAN]
    return sum(1 for item in items if item == target)


# =============================================================================
# SECTION 2 — Wrong Binary Operator  (classifier label=1, T5: CHANGE_BINARY_OPERATOR)
# Strongest overlap — both models trained on this pattern.
# =============================================================================

def find_minimum(numbers):
    # [BUGGY: Wrong Binary Operator]
    # Fix: 'n > min_val' -> 'n < min_val'
    if not numbers:
        return None
    min_val = numbers[0]
    for n in numbers[1:]:
        if n > min_val:
            min_val = n
    return min_val


def count_range_inclusive(start, end):
    # [BUGGY: Wrong Binary Operator]
    # Fix: 'while i < end' -> 'while i <= end'  (inclusive upper bound)
    count = 0
    i = start
    while i < end:
        count += 1
        i += 1
    return count


def has_all_permissions(is_authenticated, is_active):
    # [BUGGY: Wrong Binary Operator]
    # Fix: 'or' -> 'and'  (both conditions must hold)
    return is_authenticated or is_active


def is_within_bounds(index, size):
    # [BUGGY: Wrong Binary Operator]
    # Fix: 'index > size' -> 'index >= size'  (off-by-one on upper bound)
    if index < 0 or index > size:
        return False
    return True


def net_change(deposits, withdrawals):
    # [BUGGY: Wrong Binary Operator]
    # Fix: 'deposits + withdrawals' -> 'deposits - withdrawals'
    return deposits + withdrawals


def passes_threshold(score, threshold):
    # [BUGGY: Wrong Binary Operator]
    # Fix: 'score < threshold' -> 'score >= threshold'
    return score < threshold


# =============================================================================
# SECTION 3 — Swapped Operand  (classifier label=3)
# Operands of a binary expression are in the wrong order.
# =============================================================================

def elapsed_seconds(start_time, end_time):
    # [BUGGY: Swapped Operand]
    # Fix: 'start_time - end_time' -> 'end_time - start_time'
    return start_time - end_time


def compute_percentage(part, total):
    # [BUGGY: Swapped Operand]
    # Fix: 'total / part' -> 'part / total'
    if total == 0:
        return 0.0
    return (total / part) * 100


def relative_offset(value, base):
    # [BUGGY: Swapped Operand]
    # Fix: 'base - value' -> 'value - base'
    return base - value


# =============================================================================
# SECTION 4 — Boolean Literal  (T5: CHANGE_BOOLEAN_LITERAL, ~100% EM)
# Classifier may label these Clean; T5 handles them best.
# =============================================================================

def create_connection(host, port, use_ssl=True):
    # [BUGGY: Boolean Literal]
    # Fix: 'use_ssl=True' -> 'use_ssl=False'  (SSL off by default, opt-in)
    pass


def load_config(path, strict=False):
    # [CLEAN: correct default]
    pass


class Worker:
    def __init__(self):
        self.running = True  # [BUGGY: Boolean Literal] Fix: True -> False (not running until started)
        self.paused = False  # [CLEAN]

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


class RequestCache:
    def __init__(self):
        self.enabled = False     # [CLEAN]
        self.warm = True         # [BUGGY: Boolean Literal] Fix: True -> False (not warm until populated)


# =============================================================================
# SECTION 5 — Unary Operator  (T5: CHANGE_UNARY_OPERATOR, ~67% EM)
# Classifier may label these Clean; T5 handles them well.
# =============================================================================

def is_empty(container):
    # [BUGGY: Unary Operator]
    # Fix: 'return container' -> 'return not container'
    return container


def skip_if_disabled(self):
    # [BUGGY: Unary Operator]
    # Fix: 'if self.enabled' -> 'if not self.enabled'
    if self.enabled:
        return
    self._run()


def require_authentication(user):
    # [BUGGY: Unary Operator]
    # Fix: 'if user.logged_in' -> 'if not user.logged_in'
    if user.logged_in:
        raise PermissionError("Authentication required")
    return True
