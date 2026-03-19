import subprocess
import json

class StaticEngine:
    def run(self, path:str):
        jsonStaticResult = subprocess.run(['pylint',]) 