# import subprocess
# import os
# import xml.etree.ElementTree as ET

# class StaticAnalyzer:
#     """
#     Handles running Checkstyle on a Java file and parsing the results.
#     """
#     def __init__(self, checkstyle_jar_path='tools/checkstyle-10.12.5-all.jar'):
#         # Note: You might need to update the version number in the JAR file name
#         # to match the one you downloaded.
        
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#         self.checkstyle_jar = os.path.join(project_root, checkstyle_jar_path)
        
#         if not os.path.exists(self.checkstyle_jar):
#             raise FileNotFoundError(f"Checkstyle JAR not found at '{self.checkstyle_jar}'. Please download it and place it in the 'tools' directory.")
            
#         # Using Google's standard Java style guide, which is bundled in the JAR
#         self.checkstyle_rules = '/google_checks.xml'

#     def analyze(self, file_path):
#         """
#         Runs Checkstyle on the given Java file and returns a list of issues.
        
#         Args:
#             file_path (str): The absolute path to the Java file to analyze.
            
#         Returns:
#             list: A list of dictionaries, where each dictionary represents an issue.
#                   Returns an empty list if there are no issues or an error occurs.
#         """
#         if not file_path.endswith('.java'):
#             return []

#         # Command to execute Checkstyle
#         command = [
#             'java',
#             '-jar',
#             self.checkstyle_jar,
#             '-c',
#             self.checkstyle_rules,
#             '-f', 'xml', # Output format
#             file_path
#         ]
        
#         try:
#             # Run the command and capture the output
#             result = subprocess.run(command, capture_output=True, text=True, check=False)
            
#             # Checkstyle exits with a non-zero status code if it finds issues,
#             # so we check stderr for actual execution errors.
#             if result.stderr and "Checkstyle ends with" not in result.stderr:
#                 print(f"Error running Checkstyle: {result.stderr}")
#                 return []
            
#             # The XML report is in stdout
#             return self._parse_xml_output(result.stdout)

#         except FileNotFoundError:
#             print("Error: 'java' command not found. Please ensure Java is installed and in your PATH.")
#             return []
#         except Exception as e:
#             print(f"An unexpected error occurred during static analysis: {e}")
#             return []

#     def _parse_xml_output(self, xml_string):
#         """

#         Parses the XML output from Checkstyle into a structured list of issues.
#         """
#         issues = []
#         if not xml_string:
#             return issues
            
#         try:
#             root = ET.fromstring(xml_string)
#             # Find all <error> elements within any <file> element
#             for error in root.findall('.//error'):
#                 severity = error.get('severity', 'minor')
                
#                 # Map Checkstyle severities to our project's severities
#                 if severity in ['error', 'fatal']:
#                     mapped_severity = 'major'
#                 else: # 'warning', 'info', 'ignore'
#                     mapped_severity = 'minor'

#                 issues.append({
#                     'type': 'static',
#                     'line': int(error.get('line', 0)),
#                     'severity': mapped_severity,
#                     'message': error.get('message', 'No message provided.'),
#                     'source': error.get('source', 'Checkstyle')
#                 })
#             return issues
#         except ET.ParseError as e:
#             print(f"Error parsing Checkstyle XML output: {e}")
#             return []


# # Create a single, reusable instance for the app to import
# static_analyzer = StaticAnalyzer()



import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET

# --- Configuration ---
# IMPORTANT: Update this path to the exact name of your Checkstyle JAR file if it's different.
# This assumes you have a 'tools' folder in your main project directory.
CHECKSTYLE_JAR_PATH = os.path.join('tools', 'checkstyle-11.0.1-all.jar') 

def analyze_with_checkstyle(file_content):
    """
    Analyzes a string of Java code using Checkstyle.

    Args:
        file_content (str): The Java code to analyze.

    Returns:
        list: A list of dictionaries, where each dictionary represents an issue found.
              Returns an empty list if no issues are found or if Checkstyle is not configured.
    """
    if not os.path.exists(CHECKSTYLE_JAR_PATH):
        print(f"⚠️ Warning: Checkstyle JAR not found at '{CHECKSTYLE_JAR_PATH}'. Skipping static analysis.")
        return []

    issues = []
    # Use a temporary file to pass the content to Checkstyle, as it operates on files.
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.java', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # Run Checkstyle as a separate process. We use the built-in sun_checks.xml configuration.
        # The '-f xml' flag tells it to output the results in a machine-readable XML format.
        command = [
            'java', '-jar', CHECKSTYLE_JAR_PATH,
            '-c', '/sun_checks.xml',
            '-f', 'xml',
            temp_file_path
        ]
        
        # We capture stdout (for results) and stderr (for errors)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        
        # A return code of 0 usually means no issues were found. We process the XML output.
        if result.stdout:
            root = ET.fromstring(result.stdout)
            # Checkstyle XML format is <checkstyle><file><error ... /></file></checkstyle>
            for file_element in root.findall('file'):
                for error_element in file_element.findall('error'):
                    issues.append({
                        'line': error_element.get('line'),
                        'severity': error_element.get('severity', 'info').lower(), # e.g., 'error', 'warning'
                        'message': error_element.get('message'),
                        'source': 'Checkstyle' # Tag the source of the issue
                    })
    except Exception as e:
        print(f"❌ An error occurred while running Checkstyle: {e}")
    finally:
        # Always clean up the temporary file
        os.remove(temp_file_path)

    return issues
