import random
import pandas as pd
import re
from datasets import load_dataset

# --- LOGIC INJECTORS (The Smart Saboteurs) ---

def bug_logic_operator(code):
    """Swaps math operators to break logic (e.g., + becomes -)"""
    # Expanded to handle flexible spacing (x+1 vs x + 1)
    swaps = {
        r'\+': '-', 
        r'-': '+', 
        r'\*': '/', 
        r'/': '*',
        r'<': '>',
        r'>': '<',
        r'==': '!=',
    }
    
    # Check which operators are present
    # We look for the operator symbol, preserving surrounding text implies we just find it
    found_ops = []
    for op_pat in swaps.keys():
        # Match operator allowing for surrounding spaces, ensuring we don't match inside words if possible
        # For symbols, usually safe. \s* handles 0 or more spaces.
        pat = r'\s*' + op_pat + r'\s*'
        if re.search(pat, code):
            found_ops.append((pat, swaps[op_pat]))
            
    if not found_ops:
        return code
        
    target_pattern, replacement = random.choice(found_ops)
    
    # Replace with padded operator to ensure safety (x+1 -> x - 1)
    return re.sub(target_pattern, f" {replacement} ", code, count=1)

def bug_logic_boolean(code):
    """Flips booleans (True -> False)"""
    if "True" in code:
        return code.replace("True", "False", 1)
    if "False" in code:
        return code.replace("False", "True", 1)
    return code

def bug_logic_off_by_one(code):
    """Changes loop ranges or comparisons (i < 10 -> i <= 10)"""
    # Change < to <=
    if " < " in code:
        return code.replace(" < ", " <= ", 1)
    # Change > to >=
    if " > " in code:
        return code.replace(" > ", " >= ", 1)
    # Change 0 to 1 (classic array indexing error)
    if "range(0" in code:
        return code.replace("range(0", "range(1", 1)
    return code

def bug_logic_variable_swap(code):
    """Swaps variable usage (return x -> return y)"""
    # This is hard to do safely with regex, but we can try simple argument swaps
    # Find patterns like "f(a, b)" and swap to "f(b, a)"
    match = re.search(r'\((\w+), (\w+)\)', code)
    if match:
        original = match.group(0) # (a, b)
        v1, v2 = match.group(1), match.group(2)
        swapped = f"({v2}, {v1})"
        return code.replace(original, swapped, 1)
    return code

def bug_wrong_method(code):
    """Swaps common list/set/string methods"""
    method_swaps = {
        'append': 'extend',
        'extend': 'append',
        'add': 'update',   
        'update': 'add',
        'strip': 'split',
        'split': 'strip',
    }
    
    candidates = [m for m in method_swaps.keys() if f".{m}(" in code]
    if not candidates:
        return code
        
    target = random.choice(candidates)
    replacement = method_swaps[target]
    return code.replace(f".{target}(", f".{replacement}(", 1)

def bug_missing_return(code):
    """Removes the return keyword (logic error: returns None instead of value)"""
    # Matches 'return' followed by whitespace
    if re.search(r'\breturn\s+', code):
        return re.sub(r'\breturn\s+', '', code, count=1)
    return code

def bug_logic_and_or(code):
    """Swaps 'and' with 'or' and vice versa"""
    # Use word boundaries \b to avoid matching 'hand' or 'door'
    if re.search(r'\band\b', code):
        return re.sub(r'\band\b', 'or', code, count=1)
    if re.search(r'\bor\b', code):
        return re.sub(r'\bor\b', 'and', code, count=1)
    return code

if __name__ == "__main__":
    print("--- 1. LOADING FULL DATASET ---")
    print("This might take a few minutes because we are downloading ~1GB of data.")

    # CHANGED: We removed "[:1%]" so it loads EVERYTHING.
    # 'split="train"' means "give me all training data".
    try:
        # Switched to claudios/code_search_net to avoid "dataset scripts not supported" error
        dataset = load_dataset("claudios/code_search_net", split="train")
    except Exception as e:
        print("Standard load failed, trying specific Python config...")
        dataset = load_dataset("claudios/code_search_net", "python", split="train")

    # Filter for Python just to be safe (though the config usually handles it)
    # We use a simple filter to ensure we only get Python code
    python_data = dataset.filter(lambda x: x['language'] == 'python')

    print(f"SUCCESS: Loaded {len(python_data)} real Python functions.")
    print("Now generating synthetic bugs for ALL of them. This will take time...")

    # --- 3. GENERATE MASSIVE DATASET ---
    synthetic_pairs = []
    bug_generators = [bug_logic_operator, bug_logic_boolean, bug_logic_off_by_one, bug_logic_variable_swap, bug_wrong_method, bug_missing_return, bug_logic_and_or]

    # We will process in batches so your computer doesn't freeze
    # Let's target 100,000 samples for a solid FYP (400k might be too slow to train on a laptop)
    TARGET_SIZE = 100000 
    print(f"Targeting {TARGET_SIZE} samples for your FYP...")

    count = 0
    for example in python_data:
        original_code = example['func_code_string']
        
        # Skip very long or very short code (hard to learn)
        if len(original_code) > 512 or len(original_code) < 20:
            continue
            
        injector = random.choice(bug_generators)
        buggy_code = injector(original_code)
        
        if buggy_code != original_code:
            synthetic_pairs.append({
                'buggy_code': buggy_code, 
                'fixed_code': original_code
            })
            count += 1
            
        if count >= TARGET_SIZE:
            break

    # --- 4. SAVE TO CSV ---
    df = pd.DataFrame(synthetic_pairs)
    output_file = "synthetic_python_bugs.csv"
    df.to_csv(output_file, index=False)

    print(f"\n--- MISSION COMPLETE ---")
    print(f"Generated {len(df)} real pairs.")
    print(f"Saved to: {output_file}")