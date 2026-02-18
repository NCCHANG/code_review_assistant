
# -------------------------------------------------------------------------
# TEST FILE: Simulated Banking System Logic
# Contains approx 200+ lines of mixed Clean and Buggy functions.
# -------------------------------------------------------------------------

class BankAccount:
    def __init__(self, account_id, balance=0):
        self.account_id = account_id
        self.balance = balance
        self.history = []

    def deposit(self, amount):
        # [CLEAN] Basic deposit logic
        if amount > 0:
            self.balance += amount
            self.history.append(f"Deposited {amount}")
            return True
        return False

    def withdraw(self, amount):
        # [BUGGY] Operator error: uses + instead of -
        # logic_operator_swap
        if self.balance >= amount:
            self.balance + amount  # BUG: Should be -=
            self.history.append(f"Withdrew {amount}")
            return True
        return False

    def get_balance(self):
        # [CLEAN] Simple getter
        return self.balance

class TransactionManager:
    def __init__(self):
        self.transactions = []

    def process_transaction(self, sender, receiver, amount):
        # [BUGGY] Logic AND/OR swap
        # bug_logic_and_or: checks if either is None, presumably wanted BOTH valid?
        # or simplified: "if not sender or not receiver" is correct, 
        # but let's simulate "if sender and receiver" being swapped to "or" in a validation check
        
        # Original intent: if either is invalid, fail.
        # Buggy code: if sender is None OR receiver is None (Wait, this is correct logic for failure)
        # Let's try: if validation checks pass.
        
        # Bug: Logic And/Or 
        # Intent: if sender.balance >= amount AND amount > 0
        if sender.balance >= amount or amount > 0: # BUG: 'or' allows negative amount if balance is high
            sender.withdraw(amount)
            receiver.deposit(amount) 
            self.transactions.append((sender, receiver, amount))
            return True
        return False

    def audit_trail(self):
        # [BUGGY] Missing return
        # bug_missing_return
        summary = "Audit Log:\n"
        for t in self.transactions:
            summary += f"{t}\n"
        # Missing return statement here

    def clear_history(self):
        # [BUGGY] Wrong Method (append vs clear/extend)
        # Intent: self.transactions.clear()
        # Bug: self.transactions.append() ?? No, let's look at the dataset bugs.
        # bug_wrong_method might swap commonly used methods.
        self.transactions.extend([]) # harmless but weird?
        # Let's try an Off-By-One loop
        pass

def calculate_interest(principal, rate, time):
    # [CLEAN] Standard Formula
    return principal * rate * time

def validate_pin(input_pin, user_pin):
    # [BUGGY] Comparison operator
    # Intent: if input_pin == user_pin
    if input_pin != user_pin: # BUG: Swapped == to !=
        return True
    return False

def array_processor(data):
    # [CLEAN] List comprehension
    return [x * 2 for x in data if x > 0]

def find_max(numbers):
    # [BUGGY] Off by one in range? 
    # Or variable swap.
    if not numbers:
        return 0
    max_val = numbers[0]
    # bug_logic_off_by_one might change range(1, len) to range(0, len)
    for i in range(0, len(numbers)): # This is actually safe, just redundant check of 0
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val

def check_access(role):
    # [BUGGY] String comparison
    # Intent: if role == "admin"
    # if role = "admin": # Syntax error? No, python syntax error.
    # The dataset injectors produce syntactically valid python usually (logic swaps).
    # Let's do a logic swap.
    if role != "admin": # Logic error: returns True for guest instead of admin?
        return True
    return False

def compute_stats(values):
    # [BUGGY] Variable Swap
    # bug_logic_variable_swap
    total = sum(values)
    count = len(values)
    # Intent: return total / count
    if count == 0:
        return 0
    return count / total # BUG: Swapped numerator/denominator

# ---------------------------------------------------------
# Filler functions to reach ~200 lines simulation
# ---------------------------------------------------------

def helper_one():
    return 1

def helper_two():
    return 2

def helper_three(x):
    # [CLEAN]
    if x > 10:
        return x * 2
    return x

def helper_four(lst):
    # [BUGGY] Missing Return
    lst.sort()
    # Missing return lst

def helper_five(a, b):
    # [BUGGY] Operator
    return a * b # Intent was +, let's say. Context makes it hard to know, but model might guess.

class Logger:
    def log(self, msg):
        print(f"[LOG]: {msg}")

    def error(self, msg):
        # [CLEAN]
        print(f"[ERROR]: {msg}")

    def warning(self, msg):
        # [BUGGY] Off by one logic?
        # logic_network_timeout? No.
        pass

def binary_search(arr, target):
    # [CLEAN] 
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def merge_sort(arr):
    # [CLEAN] implementation
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    # [CLEAN]
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def factorial(n):
    # [BUGGY] Off by one
    if n == 0:
        return 1
    # bug_logic_operator: * becomes /
    return n / factorial(n - 1) # BUG

def is_prime(n):
    # [CLEAN]
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def fibonacci(n):
    # [BUGGY] Logic
    if n <= 0:
        return 0 # Should return [] or similar?
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) - fibonacci(n - 2) # BUG: + became -

def matrix_multiply(A, B):
    # [CLEAN]
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    # Simple N^3
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def check_permissions(user):
    # [BUGGY] Logic And/Or
    # Intent: is_active and is_admin
    if user.is_active or user.is_admin: # BUG: 'or' gives admin access to anyone active
        return True
    return False

def safe_divide(a, b):
    # [BUGGY] Logic variable swap
    if a == 0: # Check wrong variable (should be b)
        return None
    return a / b

# ... (End of file)
