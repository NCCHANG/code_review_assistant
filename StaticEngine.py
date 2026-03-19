import subprocess
import json

class StaticEngine:
    staticResults = None
    path = None
    def __init__(self, path: str):
        self.path = path
    def run(self):
        """Runs pylint on the given path and stores the results in self.staticResults."""
        jsonStaticResult = subprocess.run(['pylint', self.path, '-f', 'json'], capture_output=True, text=True)
        self.staticResults = jsonStaticResult.stdout
    def get_results(self):
        """Returns the parsed JSON results from pylint."""
        if self.staticResults is None:
            print("No results available. Please run the engine first using run().")
            return None
        
        try:
            return json.loads(self.staticResults)
        except json.JSONDecodeError:
            print("Error parsing JSON results.")
            return None
    

if __name__ == "__main__":
    engine = StaticEngine("test_complex_code.py")
    engine.run(engine.path) 