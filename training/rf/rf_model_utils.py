import ast
import re
import warnings

import numpy as np
from scipy.sparse import csr_matrix

# ── AST feature names (fixed order guarantees train/inference parity) ────────

AST_FEATURE_NAMES = [
    # Arithmetic binary operators
    "op_Add", "op_Sub", "op_Mult", "op_Div", "op_Mod", "op_Pow", "op_FloorDiv",
    # Bitwise binary operators
    "op_BitAnd", "op_BitOr", "op_BitXor", "op_LShift", "op_RShift",
    # Comparison operators
    "cmp_Eq", "cmp_NotEq", "cmp_Lt", "cmp_LtE", "cmp_Gt", "cmp_GtE",
    "cmp_Is", "cmp_IsNot", "cmp_In", "cmp_NotIn",
    # Boolean / unary operators
    "bool_And", "bool_Or",
    "unary_Not", "unary_USub",
    # Augmented assignment (+=, -=, …)
    "aug_assign_count",
    # Variable usage statistics
    "total_vars", "unique_vars", "var_repeat_ratio",
    # Code-structure counts
    "call_count", "return_count", "if_count", "loop_count",
]


def extract_ast_features(code_text):
    """Extract structural features from a Python function via AST parsing.

    Counts every binary, comparison, boolean, and unary operator type plus
    variable-usage statistics and basic code-structure metrics.  Falls back
    to all-zeros on ``SyntaxError`` so bad snippets never crash the pipeline.
    """
    features = {name: 0.0 for name in AST_FEATURE_NAMES}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code_text)
    except SyntaxError:
        return features

    variables = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            key = f"op_{type(node.op).__name__}"
            if key in features:
                features[key] += 1
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                key = f"cmp_{type(op).__name__}"
                if key in features:
                    features[key] += 1
        elif isinstance(node, ast.BoolOp):
            key = f"bool_{type(node.op).__name__}"
            if key in features:
                features[key] += 1
        elif isinstance(node, ast.UnaryOp):
            key = f"unary_{type(node.op).__name__}"
            if key in features:
                features[key] += 1
        elif isinstance(node, ast.AugAssign):
            features["aug_assign_count"] += 1
        elif isinstance(node, ast.Name):
            variables.append(node.id)
        elif isinstance(node, ast.Call):
            features["call_count"] += 1
        elif isinstance(node, ast.Return):
            features["return_count"] += 1
        elif isinstance(node, ast.If):
            features["if_count"] += 1
        elif isinstance(node, (ast.For, ast.While)):
            features["loop_count"] += 1

    n_vars = len(variables)
    n_unique = len(set(variables))
    features["total_vars"] = n_vars
    features["unique_vars"] = n_unique
    features["var_repeat_ratio"] = n_vars / max(n_unique, 1)
    return features


def build_ast_feature_matrix(code_series):
    """Extract AST features for every code string in a pandas Series.

    Returns a CSR sparse matrix of shape ``(n_samples, len(AST_FEATURE_NAMES))``
    ready to be ``hstack``-ed with TF-IDF output.
    """
    rows = [extract_ast_features(code) for code in code_series]
    matrix = np.array(
        [[row[name] for name in AST_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    return csr_matrix(matrix)


# ── Legacy tokenizer (used by rf_Classifier_trainer.py) ─────────────────────

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


# Pre-compiled pattern: multi-char operators must appear before single-char
# alternatives so the regex engine matches the longer form first.
_BUG_TOKEN_RE = re.compile(
    r'[A-Za-z_]\w*'                                      # identifiers / keywords
    r'|\d+(?:\.\d+)?'                                    # ints and floats (kept as-is)
    r'|\*\*|//|==|!=|<=|>=|\+=|-=|\*=|/=|%=|->|<<|>>'  # multi-char operators
    r'|[+\-*/%<>&|!^~=:;,.\(\)\[\]{}@]'                 # single-char operators
)

_STRING_RE = re.compile(
    r'""".*?"""|\'\'\'.*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\'',
    re.DOTALL,
)


def bug_aware_tokenizer(code_text):
    """Tokenizer for bug detection that preserves operator and numeric context.

    Differences from the generic ``tokenizer``:
    - Actual number values are kept (not replaced with NUM), so the model can
      distinguish ``a + 1`` from ``a + 2`` and learn numeric operator patterns.
    - Multi-character operators (``**``, ``==``, ``!=``, ``<=``, ``>=``, etc.)
      are emitted as single tokens rather than being split into their constituent
      characters — critical for wrong-binary-operator detection.
    - String literals are masked to ``STR`` to reduce noise without hiding the
      surrounding operator structure.
    """
    code = _STRING_RE.sub('STR', code_text)
    return _BUG_TOKEN_RE.findall(code)
