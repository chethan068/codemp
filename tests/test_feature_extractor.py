import sys
import os
import pytest

# Add the parent directory to the path so we can import the feature_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code_reviewer.feature_extractor import extract_features

def test_extract_features_valid_code():
    """
    Tests that the feature extractor works correctly on a simple, valid Java snippet.
    """
    java_code = """
    class HelloWorld {
        public static void main(String[] args) {
            if (args.length > 0) {
                System.out.println("Hello, " + args[0]);
            }
        }
    }
    """
    # Expected features: lines_of_code, node_count, if_statements
    # Note: The exact node_count can vary slightly with parser versions, but should be non-zero.
    features = extract_features(java_code)
    assert features[0] == 7  # 7 lines of code
    assert features[1] > 10 # Should have a reasonable number of AST nodes
    assert features[2] == 1  # 1 if statement

def test_extract_features_empty_code():
    """
    Tests that the feature extractor returns a default vector for an empty string.
    """
    java_code = ""
    features = extract_features(java_code)
    assert features == [0, 0, 0]

def test_extract_features_invalid_syntax():
    """
    Tests that the feature extractor handles syntax errors gracefully and returns a default vector.
    """
    java_code = "class HelloWorld { public static void main(String[] args) { if (args.length > 0) "
    features = extract_features(java_code)
    assert features == [0, 0, 0]

def test_extract_features_no_conditionals():
    """
    Tests code with no 'if' statements to ensure that feature is counted correctly.
    """
    java_code = """
    class Simple {
        void doSomething() {
            int x = 1;
            int y = 2;
            int z = x + y;
        }
    }
    """
    features = extract_features(java_code)
    assert features[0] == 7
    assert features[1] > 5
    assert features[2] == 0 # 0 if statements
