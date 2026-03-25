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