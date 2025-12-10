# import javalang
# from javalang import tree

# def calculate_cyclomatic_complexity(node):
#     """
#     Recursively calculates the Cyclomatic Complexity of a given AST node.
#     Complexity = Number of decision points + 1.
#     """
#     complexity = 0
#     # Base case: if node is None or not a Node, return 0
#     if node is None or not isinstance(node, tree.Node):
#         return 0

#     # Check for decision points in the current node
#     if isinstance(node, (
#         tree.IfStatement,
#         tree.ForStatement,
#         tree.WhileStatement,
#         tree.SwitchStatementCase, # <-- CORRECTED: Was tree.Case
#         tree.CatchClause,
#         tree.TernaryExpression
#     )):
#         complexity += 1
    
#     # Check for binary logical operators (&& and ||) which are also decision points
#     if isinstance(node, tree.BinaryOperation) and node.operator in ['&&', '||']:
#         complexity += 1

#     # Recursively sum the complexity of all children
#     for child in node.children:
#         if isinstance(child, tree.Node):
#             complexity += calculate_cyclomatic_complexity(child)
#         elif isinstance(child, list):
#             for item in child:
#                 complexity += calculate_cyclomatic_complexity(item)

#     return complexity


# def extract_features(code_string):
#     """
#     Parses a Java code string into an AST and extracts a rich set of numerical features,
#     including Cyclomatic Complexity.
#     """
#     if not code_string:
#         return [0] * 5  # Update for the new number of features

#     try:
#         tree_root = javalang.parse.parse(code_string)
        
#         # --- NEW, MORE POWERFUL FEATURES ---
        
#         # 1. Lines of Code
#         lines_of_code = len(code_string.splitlines())
        
#         # 2. Number of Methods
#         method_count = len(list(tree_root.filter(tree.MethodDeclaration)))
        
#         # 3. Cyclomatic Complexity (a very strong indicator of defect-proneness)
#         # We add 1 because the base complexity of any program is 1
#         cyclomatic_complexity = calculate_cyclomatic_complexity(tree_root) + 1
        
#         # 4. Number of try-catch blocks
#         try_statements = len(list(tree_root.filter(tree.TryStatement)))
        
#         # 5. Number of variables declared
#         variable_declarators = len(list(tree_root.filter(tree.VariableDeclarator)))

#         return [
#             lines_of_code,
#             method_count,
#             cyclomatic_complexity,
#             try_statements,
#             variable_declarators
#         ]
        
#     except (javalang.tokenizer.LexerError, javalang.parser.JavaSyntaxError, RecursionError):
#         # If code can't be parsed, return a default vector
#         return [0] * 5






# import javalang
# from javalang import tree
# import math

# def get_node_counts(tree_root):
#     """Gets comprehensive counts of various node types."""
#     counts = {
#         'methods': 0,
#         'ifs': 0,
#         'loops': 0,
#         'tries': 0,
#         'catches': 0,
#         'variables': 0,
#         'return_statements': 0,
#         'method_invocations': 0,
#         'binary_operations': 0,
#         'assignments': 0,
#         'classes': 0,
#         'switch_statements': 0,
#         'ternary_expressions': 0,
#         'null_checks': 0,
#         'array_accesses': 0,
#         'cast_expressions': 0
#     }
    
#     for _, node in tree_root:
#         if isinstance(node, tree.MethodDeclaration):
#             counts['methods'] += 1
#         elif isinstance(node, tree.IfStatement):
#             counts['ifs'] += 1
#         elif isinstance(node, (tree.ForStatement, tree.WhileStatement, tree.DoStatement)):
#             counts['loops'] += 1
#         elif isinstance(node, tree.TryStatement):
#             counts['tries'] += 1
#         elif isinstance(node, tree.CatchClause):
#             counts['catches'] += 1
#         elif isinstance(node, tree.VariableDeclarator):
#             counts['variables'] += 1
#         elif isinstance(node, tree.ReturnStatement):
#             counts['return_statements'] += 1
#         elif isinstance(node, tree.MethodInvocation):
#             counts['method_invocations'] += 1
#         elif isinstance(node, tree.BinaryOperation):
#             counts['binary_operations'] += 1
#             # Check for null comparisons
#             if hasattr(node, 'operandl') and hasattr(node, 'operandr'):
#                 if (isinstance(node.operandr, tree.Literal) and str(node.operandr.value) == 'null') or \
#                    (isinstance(node.operandl, tree.Literal) and str(node.operandl.value) == 'null'):
#                     counts['null_checks'] += 1
#         elif isinstance(node, tree.Assignment):
#             counts['assignments'] += 1
#         elif isinstance(node, tree.ClassDeclaration):
#             counts['classes'] += 1
#         elif isinstance(node, tree.SwitchStatement):
#             counts['switch_statements'] += 1
#         elif isinstance(node, tree.TernaryExpression):
#             counts['ternary_expressions'] += 1
#         elif isinstance(node, tree.ArraySelector):
#             counts['array_accesses'] += 1
#         elif isinstance(node, tree.Cast):
#             counts['cast_expressions'] += 1
    
#     return counts

# def calculate_cyclomatic_complexity(node):
#     """Recursively calculates the Cyclomatic Complexity."""
#     complexity = 0
#     if node is None or not isinstance(node, tree.Node):
#         return 0

#     # Decision points
#     if isinstance(node, (
#         tree.IfStatement, tree.ForStatement, tree.WhileStatement, tree.DoStatement,
#         tree.SwitchStatementCase, tree.CatchClause, tree.TernaryExpression
#     )) or (isinstance(node, tree.BinaryOperation) and node.operator in ['&&', '||']):
#         complexity += 1

#     # Recursion
#     for child in node.children:
#         if isinstance(child, tree.Node):
#             complexity += calculate_cyclomatic_complexity(child)
#         elif isinstance(child, list):
#             for item in child:
#                 complexity += calculate_cyclomatic_complexity(item)
#     return complexity

# def calculate_nesting_depth(node, current_depth=0):
#     """Calculate maximum nesting depth of control structures."""
#     if node is None or not isinstance(node, tree.Node):
#         return current_depth
    
#     max_depth = current_depth
    
#     # Increase depth for control structures
#     if isinstance(node, (tree.IfStatement, tree.ForStatement, tree.WhileStatement, 
#                          tree.DoStatement, tree.TryStatement, tree.SwitchStatement)):
#         current_depth += 1
#         max_depth = current_depth
    
#     # Recursively check children
#     for child in node.children:
#         if isinstance(child, tree.Node):
#             child_depth = calculate_nesting_depth(child, current_depth)
#             max_depth = max(max_depth, child_depth)
#         elif isinstance(child, list):
#             for item in child:
#                 child_depth = calculate_nesting_depth(item, current_depth)
#                 max_depth = max(max_depth, child_depth)
    
#     return max_depth

# def calculate_halstead_metrics(tree_root):
#     """Calculate simplified Halstead complexity metrics."""
#     operators = set()
#     operands = set()
    
#     for _, node in tree_root:
#         if isinstance(node, tree.BinaryOperation):
#             operators.add(node.operator)
#         elif isinstance(node, tree.MethodInvocation):
#             if node.member:
#                 operands.add(node.member)
#         elif isinstance(node, tree.MemberReference):
#             if node.member:
#                 operands.add(node.member)
    
#     n1 = len(operators)  # unique operators
#     n2 = len(operands)   # unique operands
    
#     # Halstead volume (simplified)
#     vocabulary = n1 + n2
#     if vocabulary > 0:
#         volume = vocabulary * math.log2(vocabulary) if vocabulary > 1 else 0
#     else:
#         volume = 0
    
#     return volume, n1, n2

# def extract_features(code_string):
#     """
#     Extracts comprehensive features from Java code for bug prediction.
#     Returns 20 features designed to capture code quality and complexity.
#     """
#     if not code_string or len(code_string.strip()) == 0:
#         return [0] * 20

#     try:
#         tree_root = javalang.parse.parse(code_string)
        
#         # Basic metrics
#         lines_of_code = len(code_string.splitlines())
#         non_empty_lines = len([line for line in code_string.splitlines() if line.strip()])
        
#         # Get comprehensive node counts
#         counts = get_node_counts(tree_root)
#         method_count = max(counts['methods'], 1)  # Avoid division by zero
        
#         # Complexity metrics
#         complexity = calculate_cyclomatic_complexity(tree_root) + 1
#         nesting_depth = calculate_nesting_depth(tree_root)
        
#         # Halstead metrics
#         halstead_volume, unique_operators, unique_operands = calculate_halstead_metrics(tree_root)
        
#         # Derived ratio features
#         complexity_per_method = complexity / method_count
#         loc_per_method = lines_of_code / method_count
        
#         # Decision density
#         decision_points = counts['ifs'] + counts['loops'] + counts['switch_statements']
#         decision_density = decision_points / non_empty_lines if non_empty_lines > 0 else 0
        
#         # Exception handling ratio
#         exception_handling_ratio = (counts['tries'] + counts['catches']) / method_count
        
#         # Invocation density (method calls per line)
#         invocation_density = counts['method_invocations'] / non_empty_lines if non_empty_lines > 0 else 0
        
#         # Variable usage ratio
#         variable_ratio = counts['variables'] / non_empty_lines if non_empty_lines > 0 else 0
        
#         # Null check ratio (defensive programming indicator)
#         null_check_ratio = counts['null_checks'] / counts['binary_operations'] if counts['binary_operations'] > 0 else 0
        
#         # Array access ratio (potential index out of bounds)
#         array_ratio = counts['array_accesses'] / non_empty_lines if non_empty_lines > 0 else 0
        
#         # Cast ratio (potential class cast exceptions)
#         cast_ratio = counts['cast_expressions'] / non_empty_lines if non_empty_lines > 0 else 0
        
#         # Assignment density
#         assignment_density = counts['assignments'] / non_empty_lines if non_empty_lines > 0 else 0
        
#         # Return statement ratio
#         return_ratio = counts['return_statements'] / method_count
        
#         return [
#             lines_of_code,                  # 0: Basic size metric
#             non_empty_lines,                # 1: More accurate size
#             method_count,                   # 2: Number of methods
#             complexity,                     # 3: Cyclomatic complexity (STRONG)
#             complexity_per_method,          # 4: Average method complexity
#             loc_per_method,                 # 5: Average method size
#             nesting_depth,                  # 6: Maximum nesting depth (STRONG)
#             decision_density,               # 7: Decisions per line
#             counts['ifs'],                  # 8: Conditional complexity
#             counts['loops'],                # 9: Loop complexity
#             exception_handling_ratio,       # 10: Exception handling
#             invocation_density,             # 11: Method call frequency
#             variable_ratio,                 # 12: Variable usage
#             null_check_ratio,               # 13: Defensive programming
#             array_ratio,                    # 14: Array usage risk
#             cast_ratio,                     # 15: Type casting risk
#             assignment_density,             # 16: Assignment frequency
#             return_ratio,                   # 17: Return complexity
#             halstead_volume,                # 18: Code volume
#             unique_operators + unique_operands  # 19: Vocabulary size
#         ]
        
#     except (javalang.tokenizer.LexerError, javalang.parser.JavaSyntaxError, RecursionError, Exception):
#         return [0] * 20


# This feature extractor is designed for small code snippets.
# It uses robust, text-based features that do not rely on the snippet
# being a complete, parsable Java file.

# def extract_features_from_snippet(code_snippet):
#     """
#     Extracts a set of simple, text-based features from a code snippet.
#     """
#     if not isinstance(code_snippet, str):
#         code_snippet = ""

#     features = {}
    
#     # 1. Size-based features
#     features['num_lines'] = len(code_snippet.splitlines())
#     features['num_chars'] = len(code_snippet)
    
#     # 2. Keyword-based features (common Java keywords)
#     keywords = [
#         'if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch', 'finally',
#         'throw', 'throws', 'public', 'private', 'protected', 'static', 'final',
#         'return', 'new', 'import', 'class', 'interface', 'enum', 'extends', 'implements'
#     ]
    
#     lower_code = code_snippet.lower()
#     keyword_count = sum(lower_code.count(key) for key in keywords)
#     features['keyword_count'] = keyword_count

#     # Return features in a specific, consistent order
#     return [
#         features['num_lines'],
#         features['num_chars'],
#         features['keyword_count']
#     ]


#0.63
# import pandas as pd
# import re

# def extract_features_from_snippet(code_snippet):
#     """
#     Extracts enhanced features from a Java code snippet for bug prediction.
#     """
#     if not isinstance(code_snippet, str):
#         code_snippet = ""

#     features = {}

#     # 1. Size-based features
#     features['num_lines'] = len(code_snippet.splitlines())
#     features['num_chars'] = len(code_snippet)

#     # 2. Keyword-based features
#     java_keywords = [
#         'if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch', 'finally',
#         'throw', 'throws', 'public', 'private', 'protected', 'static', 'final',
#         'return', 'new', 'import', 'class', 'interface', 'enum', 'extends', 'implements'
#     ]
#     lower_code = code_snippet.lower()
#     features['keyword_count'] = sum(lower_code.count(k) for k in java_keywords)

#     # 3. Control flow complexity
#     control_keywords = ['if', 'for', 'while', 'switch', 'case']
#     features['control_count'] = sum(lower_code.count(k) for k in control_keywords)

#     # 4. Exception handling
#     exception_keywords = ['try', 'catch', 'throw', 'throws', 'finally']
#     features['exception_count'] = sum(lower_code.count(k) for k in exception_keywords)

#     # 5. Method calls (rough estimate using parentheses)
#     features['method_calls'] = len(re.findall(r'\w+\s*\(', code_snippet))

#     # 6. Comment density
#     comment_lines = len([line for line in code_snippet.splitlines() if '//' in line or '/*' in line])
#     features['comment_ratio'] = comment_lines / (features['num_lines'] + 1e-5)

#     # 7. Suspicious terms
#     suspicious_terms = ['null', 'fail', 'error', 'exception', 'retry']
#     features['suspicious_count'] = sum(lower_code.count(term) for term in suspicious_terms)

#     # 8. Line length variability
#     line_lengths = [len(line) for line in code_snippet.splitlines()]
#     features['line_stddev'] = pd.Series(line_lengths).std() if line_lengths else 0

#     # Return features in consistent order
#     return [
#         features['num_lines'],
#         features['num_chars'],
#         features['keyword_count'],
#         features['control_count'],
#         features['exception_count'],
#         features['method_calls'],
#         features['comment_ratio'],
#         features['suspicious_count'],
#         features['line_stddev']
#     ]


# import pandas as pd
# import re
# import math

# def extract_features_from_snippet(code_snippet):
#     if not isinstance(code_snippet, str):
#         code_snippet = ""

#     features = {}

#     # Basic size features
#     features['num_lines'] = len(code_snippet.splitlines())
#     features['num_chars'] = len(code_snippet)

#     # Java keywords
#     java_keywords = [
#         'if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch', 'finally',
#         'throw', 'throws', 'public', 'private', 'protected', 'static', 'final',
#         'return', 'new', 'import', 'class', 'interface', 'enum', 'extends', 'implements'
#     ]
#     lower_code = code_snippet.lower()
#     features['keyword_count'] = sum(lower_code.count(k) for k in java_keywords)

#     # Control flow
#     control_keywords = ['if', 'for', 'while', 'switch', 'case']
#     features['control_count'] = sum(lower_code.count(k) for k in control_keywords)

#     # Exception handling
#     exception_keywords = ['try', 'catch', 'throw', 'throws', 'finally']
#     features['exception_count'] = sum(lower_code.count(k) for k in exception_keywords)

#     # Method calls
#     features['method_calls'] = len(re.findall(r'\w+\s*\(', code_snippet))

#     # Comment ratio
#     comment_lines = len([line for line in code_snippet.splitlines() if '//' in line or '/*' in line])
#     features['comment_ratio'] = comment_lines / (features['num_lines'] + 1e-5)

#     # Suspicious terms
#     suspicious_terms = ['null', 'fail', 'error', 'exception', 'retry']
#     features['suspicious_count'] = sum(lower_code.count(term) for term in suspicious_terms)

#     # Line length variability
#     line_lengths = [len(line) for line in code_snippet.splitlines()]
#     features['line_stddev'] = pd.Series(line_lengths).std() if line_lengths else 0

#     # Bug-prone APIs
#     buggy_apis = ['thread', 'file', 'socket', 'inputstream', 'system.exit']
#     features['buggy_api_count'] = sum(lower_code.count(api) for api in buggy_apis)

#     # Entropy
#     tokens = re.findall(r'\w+', code_snippet)
#     token_freq = pd.Series(tokens).value_counts(normalize=True)
#     entropy = -sum(p * math.log2(p) for p in token_freq)
#     features['entropy'] = entropy

#     return [
#         features['num_lines'], features['num_chars'], features['keyword_count'],
#         features['control_count'], features['exception_count'], features['method_calls'],
#         features['comment_ratio'], features['suspicious_count'], features['line_stddev'],
#         features['buggy_api_count'], features['entropy']
#     ]


def extract_features_from_snippet(code_snippet):
    """
    Extracts simple, robust text-based features from a code snippet.
    Returns exactly 3 features.
    """
    if not isinstance(code_snippet, str):
        return [0, 0, 0]

    # Feature 1: Number of lines
    num_lines = len(code_snippet.splitlines())
    
    # Feature 2: Number of characters
    num_chars = len(code_snippet)
    
    # Feature 3: Keyword density (Java keywords)
    keywords = [
        'if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch', 'finally',
        'throw', 'throws', 'public', 'private', 'protected', 'static', 'final',
        'return', 'new', 'import', 'class', 'interface', 'enum'
    ]
    
    lower_code = code_snippet.lower()
    keyword_count = sum(lower_code.count(key) for key in keywords)

    return [num_lines, num_chars, keyword_count]