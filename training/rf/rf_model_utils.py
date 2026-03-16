import re

def tokenizer(code_text):
    code = re.sub(r'(".*?"|\'.*?\')', ' STR ', code_text)
    # Replace numbers with ' NUM ' but keep 0 and 1
    def replace_num(match):
        val = match.group(0)
        return val if val in ('0', '1') else ' NUM '
    code = re.sub(r'\b\d+\b', replace_num, code)
    # Add space around operators
    code = re.sub(r'([=\+\-\*/%<>&!\|:;\{\}\(\)\[\]])', r' \1 ', code)
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)

    # Collapse multiple spaces into a single space
    return code.split()
