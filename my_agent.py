from __future__ import annotations
import ast
import asyncio
import glob
import io
import json
import math
import os
import tokenize
import aiohttp
import requests
import subprocess
import ast, sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
from uuid import uuid4

from tqdm import tqdm

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1800"))

GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS=[GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
MAX_FIX_TASK_STEPS = 100

PYTEST_COMMAND_TEMPLATE = textwrap.dedent("""\
python -c "import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
sys.exit(pytest.main([{file_paths}, '-v']))"\
""")

AVAILABLE_INSTANCE_TYPES = {
    "list": list,
    "dict": dict,
    "str": str,
    "int": int,
    "bool": bool,
    "tuple": tuple,
    "any": type,
    "function": type(lambda: None),
    "NoneType": type(None),
    "varargs": "varargs",  # Special string identifier for *args
    "kwargs": "kwargs",    # Special string identifier for **kwargs
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent("""
<role>
You are a problem type classifier. Analyze the problem statement to understand its fundamental nature and intent.
</role>

<core_principle>
- FIX: The problem addresses existing functionality that is not working correctly or as intended
- CREATE: The problem requests creating new functionality that doesn't currently exist
</core_principle>
</output>
<analysis_framework>
1. **Context Analysis**: What is the current state of the system?
   - Is there existing functionality being discussed?
   - Is the problem about something that already exists but has issues?

2. **Intent Analysis**: What is the primary goal?
   - Is the goal to make existing functionality work correctly?
   - Is the goal to add new capabilities to the system?

3. **Scope Analysis**: What is the scope of the solution?
   - Does it involve modifying/improving existing code?
   - Does it involve creating entirely new code from scratch?
</analysis_framework>

<classification_logic>
**FIX applies when:**
- The problem describes existing functionality that is not working as expected
- The solution involves correcting, improving, or fixing existing behavior
- The context indicates there's a gap between current behavior and desired behavior
- The problem arises from limitations or issues in current implementation

**CREATE applies when:**
- The problem requests entirely new functionality not currently present
- The solution involves creating new features, modules, or capabilities
- There's no existing functionality to fix or improve
- The request is for adding new capabilities to the system
</classification_logic>

<important_considerations>
- A "new approach" to fix existing problems is still FIX
- Proposing alternative solutions to existing issues is still FIX
- The method of solution doesn't change the fundamental nature of the problem

Analyze the problem statement holistically, considering context, intent, and scope to make your classification.
</important_considerations>

<output>
Only respond with "FIX" or "CREATE".
</output>
""")

DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}

Try to use something different!
""")

INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
"""
You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.

CRITICAL INFINITE LOOP DETECTION:
1. Check for while True: loops without guaranteed exit conditions
2. Verify all while loops have clear termination conditions
3. Ensure recursive functions have proper base cases
4. Look for loops that depend on external state that might never change
5. Check for patterns that could lead to infinite iteration

If you find potential infinite loops:
- Provide a corrected version of the code
- Ensure all loops have finite termination conditions
- Add reasonable iteration limits or timeout mechanisms where appropriate

If no infinite loops are detected:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

example:
```python
a.py
contents of a.py

b.py
contents of b.py
```
"""
)

GENERATE_INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit
5. Use unit rescaling on float values into integers as return values instead of floats to prevent floating-point precision errors.
6. If there is no provided value for invalid solutions as return, assume None should be returned.

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)

GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
9. Use unit rescaling on float values into integers as return values instead of floats to prevent floating-point precision errors.
10. If there is no provided value for invalid solutions as return, assume None should be returned.
11. When generating sequences of statements or outputs derived from input lists, distinguish between intermediate and terminal elements. Apply special formatting or modifiers only to the terminal element unless explicitly instructed otherwise.
12. When regenerating or resetting values, always validate that the new result differs from the previous one, and retry generation until a unique value is obtained.

Return only the final Python code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)
GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit
5. Use unit rescaling on float values into integers as return values instead of floats to prevent floating-point precision errors.
6. If there is no provided value for invalid solutions as return, assume None should be returned.

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)


FUNCTION_METADATA_GENERATION_SYSTEM_PROMPT = textwrap.dedent(
'''
You are the function description generator from the given instruction and code skeleton.
Your role is to generate detailed descriptions for all functions or methods by following these steps:
- Understand the instruction completely
- Extract functions, or methods from the given code skeleton.
- Based on instructions, for each functions, or methods, you need to generate:
    - "name": function name or class method name:
        - for functions, simply function name(e.g. "add")
        - for class methods, class_name.method_name(e.g. "classA.func_a")
    - "usage": general usages of function or method
    - "parameters": List of parameters, for each parameter:
        - "name": name of the parameter (include * for varargs, ** for kwargs)
        - "type": detect parameter instance types: "list", "dict", "int", "str", "bool", "tuple", "function", "NoneType", "varargs", "kwargs", "any", or specific types
        - "parameter_type": one of: "positional", "keyword", "varargs", "kwargs"
        - "required": whether this parameter is required or not: true or false
        - "description": detailed description of the parameter
        - "example": example of the parameter
        - "default_value": default value of the parameter
    - "return": detailed description of the return value with:
        - "description": detailed description of the return value
        - "example": example of the return value
        - "type": detect return value instance types: "list", "dict", "int", "str", "bool", "tuple", "function", "NoneType", "any"
    - "exceptions": List of exceptions, for each exception:
        - "type": type of Exception
        - "error_message": detailed description of the exception
        - "explictely_defined_from_instruction": whether exception error message is explicitely defined in instruction or you created it yourself
        - "description": detailed description of the exception

# Special Parameter Handling:
- For *args parameters:
    - "name": "*args"
    - "type": "varargs" or specific type if known (e.g., "list")
    - "parameter_type": "varargs"
    - "description": should explain it accepts variable positional arguments

- For **kwargs parameters:
    - "name": "**kwargs"
    - "type": "kwargs" or specific type if known (e.g., "dict")
    - "parameter_type": "kwargs"
    - "description": should explain it accepts variable keyword arguments

- For parameters with default values:
    - "required": false
    - "default_value": the default value

- For parameters without default values:
    - "required": true
    - "default_value": null

# Special Type Handling:
- For type "tuple" or "dict":
    - Wrap example with "" because tuple or dict are not valid json(**This rule applies only to "tuple" or "dict" type, for other types, do not wrap example with ""**)
        example:
        ```
        {
            "type": "tuple",
            "example": ("A", "B"),
            "description": "A tuple of two strings"
        }
        ```
        should be:
        ```
        {
            "type": "tuple",
            "example": "("A", "B")",
            "description": "A tuple of two strings"
        }
        ```

- Do not generate exception case **when exception error message is not explicitely defined in instructions**.

- Do not add markups in the code like `[EMPTY_LINE]` or [2_TRAILING_SPACES]`, these are for internal use only.

- **Do not use list for multi-line string** if they are not explicitely formatted or defined as list in the instruction or skeleton code, use just string type.
    example:
    ```
    "line 1
    line 2
    "
    ```
    should be:
    ```
    "line 1\nline 2"
    ```

# Response Format:
You should respond in json format

## Response Example
[
    {
        "name": "func_with_variadic",
        "usage": "Function that accepts various types of arguments",
        "parameters": [
            {
                "name": "required_arg",
                "type": "int",
                "parameter_type": "positional",
                "required": true,
                "example": 42,
                "default_value": null,
                "description": "A required positional argument"
            },
            {
                "name": "optional_arg",
                "type": "str",
                "parameter_type": "keyword",
                "required": false,
                "example": "default",
                "default_value": "default",
                "description": "An optional keyword argument with default value"
            },
            {
                "name": "*args",
                "type": "varargs",
                "parameter_type": "varargs",
                "required": false,
                "example": [1, 2, 3],
                "default_value": null,
                "description": "Variable positional arguments"
            },
            {
                "name": "**kwargs",
                "type": "kwargs",
                "parameter_type": "kwargs",
                "required": false,
                "example": {"key": "value"},
                "default_value": null,
                "description": "Variable keyword arguments"
            }
        ],
        "return": {
            "type": "list",
            "example": ["result1", "result2"],
            "description": "Returns a list with processing results"
        },
        "exceptions": [
            {
                "type": "TypeError",
                "error_message": "Invalid argument type provided",
                "explictely_defined_from_instruction": true,
                "description": "Raised when incorrect argument types are passed"
            }
        ]
    }
]
'''
)

GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent("""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
9. Use unit rescaling on float values into integers as return values instead of floats to prevent floating-point precision errors.
10. If there is no provided value for invalid solutions as return, assume None should be returned.
11. When generating sequences of statements or outputs derived from input lists, distinguish between intermediate and terminal elements. Apply special formatting or modifiers only to the terminal element unless explicitly instructed otherwise.
12. When regenerating or resetting values, always validate that the new result differs from the previous one, and retry generation until a unique value is obtained.

Return only the final python files code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)

TESTCASES_CHECK_PROMPT = textwrap.dedent(
"""
You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

Important:
1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
3. Minimize all testcases as you have context and generation limit
4. If there is no provided value for invalid solutions as return, assume None should be returned.


If no invalid testcases are detected and covered all critical edge cases:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.

example:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)

CREATE_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.

## Follow these steps to fix the issue:
1. As a first step, find the relevant files in the repo to work on.
2. Localise the code causing the issue.
3. Edit the sourcecode of the repo to resolve the issue.
4. Think about edgecases and make sure the fix handles them as well.
5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
13. If you find that the error while running the run_code or run_repo_tests_create tool due to missing dependencies, do not try to solve it as you don't have any internet access.

## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

## Test generation guidance:
- Use `generate_test_function(file_path, test_function_code, position)` after discovering the most relevant existing test file.
- Prefer `position="auto"` which inserts after imports or before the `if __name__ == "__main__":` block when present, falling back to append.
- Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.
- Keep generated tests minimal and focused on the bug and its edge cases.
- Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.
Always think what step you are in and what you need to do next.

<workflow_steps>
  <step number="1">
    <description>Find the relevant files in the repo to work on</description>
    <rules>
      - Use `search_in_all_files_content` to enumerate matches across the codebase
      - Tests and patch contexts may span multiple files. Do not stop after the first similar match
      - Keep searching the repository after each match to find all relevant files
    </rules>
  </step>
  
  <step number="2">
    <description>Localise the code causing the issue</description>
    <rules>
      - Use `search_in_specified_file` to drill into each file
      - Iterate until no applicable occurrences remain
    </rules>
  </step>
  
  <step number="3">
    <description>Propose at least 2 meaningfully different and accurate solutions</description>
    <rules>
      - You need to propose at least 2 meaningfully different and elegant solutions to the problem to the user for approval
      - All the solutions proposed need to be accurate
      - You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important
      - Expected output should be closest to the most relevant test case
      - Each solution must be very detailed and explain why they are better than the other solutions
    </rules>
  </step>
  
  <step number="4">
    <description>Get approval for your proposed solution</description>
    <rules>
      - Use `get_approval_for_solution` tool with your list of proposed solutions
      - Provide the index of the solution you think is the best
      - Explain the reason for selecting that solution over other solutions
    </rules>
  </step>
  
  <step number="5">
    <description>Edit the sourcecode of the repo to resolve the issue</description>
    <rules>
      - Use `apply_code_edit` tool after getting approval
      - Apply consistent changes to every relevant file before finishing
      - Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement
      - Do not create any new files or directories unless absolutely necessary for the fix
    </rules>
  </step>
  
  <step number="6">
    <description>Think about edge cases and ensure the fix handles them</description>
    <rules>
      - Consider boundary conditions, null/empty inputs, and error scenarios
      - Test your fix mentally against various edge cases before implementing
    </rules>
  </step>
  
  <step number="7">
    <description>Validate your changes don't break existing functionality</description>
    <rules>
      - Thoroughly check the entire code base to ensure the changes made are exhaustive
      - Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified
      - Always check **only the test cases provided or mentioned by the user**
      - Re-run tests only for the specific test cases or files identified by the user
      - Do not attempt to fix or modify other unrelated test cases
    </rules>
  </step>
</workflow_steps>

<testing_rules>
  <rule>Never edit/update the existing test files directly when validating a hypothesis</rule>
  <rule>Only consider and validate against the tests explicitly identified or provided by the user.</rule>
  <rule>Do not attempt to auto-generate new test functions or files.</rule>
  <rule>Run only the user-specified failing or relevant test cases; ignore unrelated ones.</rule>
  <rule>Keep the fix minimal and scoped exactly to the user-specified issue.</rule>
</testing_rules>

<important_constraints>
  <constraint>If you find that the error while running the run_code or run_repo_tests tool is due to missing dependencies, do not try to solve it as you don't have any internet access.</constraint>
  <constraint>If you fix all tests provided by the user call finish tool method </constraint>
</important_constraints>


You have access to the following tools:-
{tools_docs}

{format_prompt}
""")


FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
# Here are the tests that are most likely to be relevant to the problem:
{test_func_names}                                   
""")

CREATE_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
""")
FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
You are a helpful assistant that can find the test runner for a given repository.
- The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
- Do not use the test runner to run test for whole repository or test setup.
- Read the README file and find the test runner. If there is no test runner, return pytest.
- Output format should be as the following. No other texts are allowed.
abc/test.py
""")

TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
You are a helpful assistant that determines the mode of the test runner.
Read the test runner file and determine if it requires a module or a file path to run the test.
Output should be one of MODULE or FILE, No other texts are allowed.
- MODULE: When the test runner requires a module path to run the test.
- FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
""")


STOP_INSTRUCTION=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")

FORMAT_PROMPT_V0=textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `next_thought`: Detailed reasoning (include:
     - Which step are you in and what is success criteria for that step.
     - Problem understanding
     - Code analysis
     - Solution justification
     - Validation plan)
   - `next_tool_name`: Must be an exact tool name from the tool list
   - `next_tool_args`: Valid JSON with:
     - Proper escaping
     - No trailing commas
     - Tool-specific parameters

2. **Error Handling Format**:
   - For errors: 
     next_thought: "Error: [detailed explanation]"
     next_tool_name: ""
     next_tool_args: {}

3. **Example Valid Format**:
   next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   next_tool_name: "apply_code_edit"
   next_tool_args: {
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\n    return json.loads(response)\nexcept JSONDecodeError:\n    logger.error(f'Invalid JSON: {{response}}')\n    raise"
   }

4. **Invalid Format Examples** (Avoid These):
   - Missing any of the three required fields
   - JSON syntax errors in next_tool_args
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
""")

run_id=None
  
class EnhancedCOT:
    class Action:
            
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
    def __init__(self,latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep

    def add_action(self, action: EnhancedCOT.Action) -> bool: # don't add if thought is repeated
        logger.info(f"Adding action: tool={action.next_tool_name}, thought_length={len(action.next_thought) if action.next_thought else 0}, observation_length={len(str(action.observation)) if action.observation else 0}")
        if action.observation:
            logger.info(f"Observation: {str(action.observation)[:200]}{'...' if len(str(action.observation)) > 200 else ''}")
        self.thoughts.append(action)
        return True
        
    def is_thought_repeated(self)->bool:
        # Check if the last thought is the same as the previous thought.
        # If there are less than 2 thoughts, skip (return False).
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False
    def to_str(self):
        messages=[]
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i<len(self.thoughts)-self.latest_observations_to_keep:
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n"
                    f"next_tool_name:{thought.next_tool_name}\n"
                    f"next_tool_args:{thought.next_tool_args}\n"
                )
                # Compute observation summary length safely for str/list/None
                if thought.observation is None:
                    _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)):
                    _obs_len = len(thought.observation)
                else:
                    _obs_len = len(str(thought.observation).splitlines())
                user_str=( f"observation: {'error ocurred.' if thought.is_error else ''} "
                    f"output omitted ({_obs_len}) lines\n")
                
            else:
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
            messages.append({"role":"assistant","content":assistant_str})
            messages.append({"role":"user","content":user_str})
        return messages

class Utils:
    
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                # if fixed_json == ""
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError(f"Invalid JSON: {json_string}")
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        with open("../failed_messages.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None

class ClassVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content

    def visit_ClassDef(self, node):
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.classes[node.name] = {
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response=cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0)->str:
        logger.info(f"Making LLM request to {model} (attempt {attempt+1}, temp: {temperature})")
        global run_id
        logger.info(f"Starting inference: model={model}, num_messages={len(messages)}, temperature={temperature}, run_id={run_id}")
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"

        # Cache miss - make the actual request
        request_data = {
                "run_id": run_id if run_id else str(uuid4()),
                "messages": messages,
                "temperature": temperature,
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model
        
        try:
            response = requests.post(url, data=json.dumps(request_data), timeout=120, headers=headers)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.ConnectionError as e:
            return f"ERROR: Connection failed for model {model}"
        except requests.exceptions.HTTPError as e:
            return f"ERROR: HTTP error {e.response.status_code} for model {model}"
        except requests.exceptions.RequestException as e:
            return f"ERROR: Request failed for model {model}"
        
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            return f"ERROR: Invalid JSON response for model {model}"
        
        try:
            is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
            if is_oai_interface:
                raw_text=response_json['choices'][0]['message']['content']
            else:
                if type(response_json) is str:
                    raw_text=response_json.strip("\n").strip()
                else:
                    raw_text=response_json
            if type(raw_text) is not dict:
                raw_text=raw_text.lstrip()
            logger.info(f"LLM response received from {model} (length: {len(str(raw_text)) if raw_text else 0} chars)")
            return raw_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Invalid response structure for model {model}: {e}")
            return f"ERROR: Invalid response structure for model {model}"
        except Exception as e:
            logger.error(f"Unexpected error for model {model}: {e}")
            return f"ERROR: Unexpected error for model {model}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, 
                            model: str,
                            max_retries: int = 5, 
                            base_delay: float = 1.0,
                            temperature: float = 0.0) -> str:
        
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text=cls.make_request(messages,model=AGENT_MODELS[(index + attempt)%len(AGENT_MODELS)], temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                if attempt < max_retries:
                    delay = base_delay 
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name]+=1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    time.sleep(random.uniform(1.2*delay, 1.5*delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages
    
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, next_tool_args: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''

        next_tool_args=next_tool_args.replace('```json','').strip('```')
        error_msg=''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg=f"Invalid JSON: {next_tool_args}"    
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name,required=True), next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()),return_json:bool=False, temperature:float=0.0) -> dict:
        """Prod inference with caching"""
    
        logger.info(f"Starting inference: model={model}, num_messages={len(messages)}, temperature={temperature}, run_id={run_id}")
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")

            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
        
        logger.info(f"Inference completed: tool={next_tool_name}, attempts={total_attempts}, errors={sum(error_counter.values()) if error_counter else 0}")
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        # remove all leading and trailing quotes
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:
            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            # remove all leading and trailing quotes in tool_name
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)
        
        return text_resp

    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, Any, Any]:
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:")<text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_thought=text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw=text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                # Enforce arrays per new contract: if single string/object, wrap as arrays
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
            except JSONDecodeError as e:
                error_msg=f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
                
        else:
            if "next_thought:" not in text_resp:
                error_msg="Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg="Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg="Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:")>text_resp.find("next_tool_name:"):
                error_msg="Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:")>text_resp.find("next_tool_args:"):
                error_msg="Invalid response. next_tool_name is after next_tool_args"
            Utils.log_to_failed_messages(text_resp)
            return None,None,None,error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
            
        return next_thought, next_tool_name, next_tool_args,error_msg

class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_FILE_PATH=11
            INVALID_TOOL_CALL=12
            IMPORT_ERROR=13
            
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            logger.info(f"Using tool: {fn.__name__} with args: {args[:2] if len(args) > 2 else args}, kwargs: {list(kwargs.keys())}")
            self.tool_invocations[fn.__name__]+=1
            try:
                result = fn(self, *args, **kwargs)
                logger.info(f"Tool {fn.__name__} completed successfully (result length: {len(str(result)) if result else 0} chars)")
                return result
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                logger.error(f"Tool {fn.__name__} failed with error: {e.error_type} - {e.message}")
                return e.message

        # Preserve original function metadata
       
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.is_tool=True

        return wrapper

    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self,tool_name:str,required_only:bool=False)->list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']

    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        
        return tool_method
    
    
    def _check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            # self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)
    
    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
        try:
            # Update to include cfg, txt, and toml files along with py files
            # Check whether ignore_files is a property of this clas
            command = f"""
            shopt -s globstar

            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            
            return output.stdout
        except Exception as e:
            return f"Error generating git patch: {e}"

class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]

        # Check all classes in the method resolution order (MRO) to include inherited tools
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={
            k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations={
          k:0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term is not None and search_term!="":
            return self.search_in_specified_file(file_path, search_term)
            
        # check if start and end line are not between a function..
        func_ranges=self.get_function_ranges(file_path)
        if search_start_line!=None:
            for start, end, name in func_ranges:
                if start<=search_start_line<=end:
                    if start<search_start_line:
                        search_start_line=start
        if search_end_line!=None:
            for start, end, name in func_ranges:
                if start<=search_end_line<=end:
                    if end>search_end_line:
                        search_end_line=end
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit!=-1  else content
    
    @EnhancedToolManager.tool
    def test_patch_find_finish(self, test_func_names: List[str]):
        '''
        Signals completion of the test patch find workflow execution
        Arguments:
            test_func_names: The list of test function names with file path (e.g. ["path.class.function", "another_path.class.function"])
            **REMEMBER:** each name format should be "path.class.function". DON'T add any other texts like comments and line numbers.
        '''
        return "finish"

    @EnhancedToolManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
       
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
        
    @EnhancedToolManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self,solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
            
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)
        
        solutions = parsed_solutions

        if type(solutions) is not list or len(solutions)<2:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: solutions must be a list with length at least 2.")

        self.is_solution_approved = True
        return "Approved"
          
    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)
 

    @EnhancedToolManager.tool
    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        '''
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")

        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Normalize newline handling
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,"Error: test_function_code cannot be empty.")

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            # Validate standalone content before writing
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: generated test function has syntax error: {err}")
        else:
            original = self._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")

            # Try each candidate until one passes syntax check
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: inserting test caused syntax error. First error: {first_error}")

        self._save(file_path, new_content)

        # Track for exclusion from final patch
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})." 

    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            parts = function_path.split("::")
            file_path = parts[0]
            function_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = FunctionVisitor(content)
                visitor.visit(tree)
                
                if function_name in visitor.functions:
                    functions[function_path] = visitor.functions[function_name].get("body", "")
                else:
                    functions[function_path] = f"Function {function_name} not found in {file_path}"
            except FileNotFoundError:
                functions[function_path] = f"File {file_path} not found"
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {str(e)}"

        return functions

    @EnhancedToolManager.tool
    def get_classes(self, class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            parts = class_path.split("::")
            file_path = parts[0]
            class_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = ClassVisitor(content)
                visitor.visit(tree)
                if class_name in visitor.classes:
                    classes[class_path] = visitor.classes[class_name].get("body", "")
                else:
                    classes[class_path] = f"Class {class_name} not found in {file_path}"
            except FileNotFoundError:
                classes[class_path] = f"File {file_path} not found"
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {str(e)}"

        return classes

    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        Do not try to search using module names, because it wont work.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        # Walk through all directories and find Python files
        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue

                        # Parse the file content using AST
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        pass

        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in the codebase.")
        return output

    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            tree = None 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error parsing '{file_path}': {e}, {traceback.format_exc()}")
             # Fallback if file cannot be parsed.

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{file_path}'")

        func_ranges=self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    @EnhancedToolManager.tool
    def search_in_specified_file(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)

    # @tool
    def search_recurive_in_all_files_in_directory(self, directory_path: str, search_term: str)->str:
        '''
        Locates text patterns recursively within all files in a specific directory
        Arguments:
            directory_path: target directory for pattern matching
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: directory '{directory_path}' does not exist.")
        output=subprocess.run(["bash", "-c", f"grep -rn --include='*.py' {directory_path} -e '{search_term}'"], capture_output=True)
        output=output.stdout.decode("utf-8")
        output=Utils.limit_strings(output, n=100)
        if not output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{directory_path}'")
        return output
    
        
    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(self, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )

            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                pass

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"
    

    def create_new_file(self,file_path:str, content:str)->str:
        '''
        Generates new file with specified content at target location. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            file_path: destination path for new file
            content: text content for file creation
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)

    @EnhancedToolManager.tool
    def run_repo_tests_create(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        if self.test_runner == "pytest":
            print("CMD: pytest ", file_paths)
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            print("CMD: python ", file_paths)
            output = ""
            for file_path in file_paths:
                result = subprocess.run(
                    ["python", file_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                current_output = (result.stdout or "") + (result.stderr or "")
                output += current_output
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output
    
    @EnhancedToolManager.tool
    def run_repo_tests(self, files_to_test: List[str], timeout_secs: int = 60) -> tuple[str, bool]:
        '''
        Run tests for the specified test files.
        Arguments:
            files_to_test: paths to the test to run (e.g., ["path.test_file.class.test_method"])
            timeout_secs: maximum time to allow for test execution default 60 seconds
        Output:
            Success message or error message
        '''
        # Check if this is a Django project and try Django's runtests.py first
        
        print(f"current cwd: {os.getcwd()} and runtests.py exists: {os.path.exists('runtests.py')} and tests/runtests.py exists: {os.path.exists('tests/runtests.py')}")
        print(f"Current cwd contents: {os.listdir('.')}")
        
        runtests_path = None
        if os.path.exists("tests/runtests.py"):
            runtests_path = "tests/runtests.py"
        elif os.path.exists("runtests.py"):
            runtests_path = "runtests.py"
        else:
            return "ERROR: No runtests.py found.", False

        try:
            logger.info("Using Django's tests/runtests.py")
            # Convert file paths to test names for Django test runner
            django_test_names = []
            for file_path in files_to_test:
                test_name = file_path.strip()

                if test_name.startswith("tests."):
                    test_name = test_name[len("tests.") :]

                if file_path.endswith(".py"):
                    test_name = (
                        file_path.replace("./", "")
                        .replace(".py", "")
                        .replace("/", ".")
                    )

                if test_name:
                    django_test_names.append(test_name)
            
            if django_test_names:
                tests_str = " ".join(django_test_names)
                cmd = f"{sys.executable} {runtests_path} {tests_str} --parallel=1 -v 2"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_secs)
                
                logger.info(f"Run tests command result: {result}")
                
                out = (result.stdout or "") + (result.stderr or "")
                
                # Check for import errors specifically
                if "ModuleNotFoundError: No module named 'django'" in out or "ImportError" in out:
                    logger.warning("Django import error detected, virtual environment may not be properly activated")
                    return f"ERROR: Django import failed. Virtual environment may not be activated properly.\n{out}", False

                try:
                    output = out.split('============================\n', 1)[1]
                except Exception as e:
                    try:
                        output = out.split('-------------------\n', 1)[1]
                    except Exception as e:
                        output = out
                        logger.warning(f"Could not parse Django test output: {e}")
                logger.info(f"RAW OUTPUT:\n\n {output}\n\n==========================")
                return output, True
                
        except subprocess.TimeoutExpired:
            logger.warning("Django tests/runtests.py timed out")
            return "ERROR: Django tests timed out.", False
        except Exception as e:
            logger.warning(f"Django tests/runtests.py failed: {e}")
        
        # Fall back to pytest
        try:
            last_test_runner = 'pytest'
            # Build proper arguments list for pytest.main()
            if files_to_test:
                file_paths_str = ", ".join([f"'{f}'" for f in files_to_test]) + ", "
            else:
                file_paths_str = ""
            command = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            out = (result.stdout or "") + (result.stderr or "")
            output, success, failed_count = self.analyze_pytest_output(out)
            if self.test_runner != 'pytest' and self._check_dependency_errors(output):
                last_test_runner = self.test_runner
                if self.test_runner_mode == "MODULE":
                    # Use Django-specific module conversion for Django projects
                    if os.path.exists('tests/runtests.py') or os.path.exists('runtests.py'):
                        modules = []
                        for file_path in files_to_test:
                            # Handle different input formats for Django tests
                            if "::" in file_path:
                                test_name = file_path.split("::")[0]  # Take just the module part
                            elif file_path.startswith('tests/'):
                                # Convert file path to Django test module name
                                test_name = file_path.replace('tests/', '', 1).replace('.py', '').replace('/', '.')
                            elif file_path.endswith('.py'):
                                # Remove .py extension and convert slashes to dots
                                test_name = file_path.replace('.py', '').replace('/', '.')
                            else:
                                # Assume it's already a module name
                                test_name = file_path
                            
                            if test_name:
                                modules.append(test_name)
                    else:
                        modules = [filepath_to_module(f, "repo", self.test_runner) for f in files_to_test]
                    
                    
                    if "manage.py" in self.test_runner or "runtests.py" in self.test_runner:
                        cmd = f"{self.test_runner} --parallel=1 {' '.join(modules)}"
                    else:
                        cmd = f"{self.test_runner} {' '.join(modules)}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                    out = (result.stdout or "") + (result.stderr or "")
                    success = False if "error" in out.lower() else True
                    if len(out) > 20000:
                        lines = out.splitlines()
                        if len(lines) > 500:
                            output = "\n".join(lines[:400] + ["... ({} lines omitted) ...".format(len(lines)-500)] + lines[-100:])
                        else:
                            output = out
                    else:
                        output = out
                    return output, success
                else:
                    # Default file-based runner
                    files_to_test = [clean_filepath(f, "repo", self.test_runner) for f in files_to_test]
                    
                    if "manage.py" in self.test_runner or "runtests.py" in self.test_runner:
                        cmd = f"{self.test_runner} --parallel=1 {' '.join(files_to_test)}"
                    else:
                        cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                    out = (result.stdout or "") + (result.stderr or "")
                    success = False if "error" in out.lower() else True
                    if len(out) > 20000:
                        lines = out.splitlines()
                        if len(lines) > 500:
                            output = "\n".join(lines[:400] + ["... ({} lines omitted) ...".format(len(lines)-500)] + lines[-100:])
                        else:
                            output = out
                    else:
                        output = out
                    return out, success

            if not success:
                if len(out) > 20000:
                    lines = out.splitlines()
                    if len(lines) > 500:
                        output = "\n".join(lines[:400] + ["... ({} lines omitted) ...".format(len(lines)-500)] + lines[-100:])
                    else:
                        output = out
                else:
                    output = out
            else:
                if self.failed_count == -1:
                    self.failed_count = failed_count

                    debug_prints = self._extract_debug_prints_from_pytest(out)
                    failed_test_names = self._extract_failed_test_names(output)
                    if debug_prints and failed_test_names:
                        output += "\n\n=================================== Debug Prints ===================================\n\n"
                        for test_name, prints in debug_prints.items():
                            if test_name in failed_test_names:
                                if len(prints) > 0:
                                    output += f"\n---------------------------------- Debug prints for {test_name} ----------------------------------\n"
                                    for printer in prints:
                                        output += f"\n{printer}"
                        output += "\n\n=================================== End of Debug Prints ===================================\n\n"

                if self.failed_count > failed_count: # if you've made progress, checkpoint your progress
                    if failed_count > 0:
                        output += f"\n\nYou resolved {self.failed_count - failed_count} failures."
                    else:
                        output += f"\n\nCongratulations! You fixed all failures. Finish the task with `finish` tool."
                    self.failed_count = failed_count
                    self.checkpoint = self.get_final_git_patch() # manual checkpoint
            return output, True if "Successfully ran all tests." in output else False
        except subprocess.TimeoutExpired:
            return "ERROR: tests timed out.", False



    def _check_dependency_errors(self, output: str) -> bool:
        """
        Check if the output contains dependency errors.
        """
        # Check for all possible dependency error messages in the output
        dependency_error_signatures = [
            "ModuleNotFoundError",
            "No module named",
            "ImportError: cannot import name",
            "ImportError: No module named",
            "ImportError: attempted relative import",
            "ImportError: cannot import module",
            "ImportError: attempted import error",
            "ImportError: DLL load failed",
            "ImportError: dynamic module does not define",
            "ImportError: cannot import",
            "ImportError: missing dependency",
            "ImportError: failed to import",
            "ImportError: cannot open shared object file",
            "ImportError: cannot load library",
            "ImportError: undefined symbol",
            "ImportError: bad magic number",
            "ImportError: incompatible library",
            "pkg_resources.DistributionNotFound",
            "pkg_resources.VersionConflict",
            "ModuleNotFoundError:",
            "ImportError:",
            "INTERNALERROR",
            "No module named",
            "Could not find a version that satisfies the requirement",
            "ERROR: Could not find a version that satisfies the requirement",
            "ERROR: No matching distribution found for",
            "ImportError",
            "not configured",
            "ModuleNotFoundError",
            "No module named",
            "missing module named",
            "missing dependency",
            "Failed to import",
            "Could not import",
            "cannot import",
            "cannot open shared object file",
            "undefined symbol",
            "bad magic number",
            "incompatible library",
        ]
        output_lower = output.lower()
        return any(sig.lower() in output_lower for sig in dependency_error_signatures)

    def analyze_pytest_output(self, output) -> tuple[str, bool, int]:
        """
        Main pytest output analyzer - routes to appropriate parser.
        Handles both regular pytest runs and meta-testing scenarios.
        """
        if not isinstance(output, str) or not output.strip():
            return "Invalid pytest output.", False, 0
        
        # Detect if this is meta-testing (multiple test session starts)
        session_starts = list(re.finditer(r'={5,}\s*test session starts\s*={5,}', output, re.IGNORECASE))
        
        if len(session_starts) > 1:
            # Meta-testing scenario - use specialized parser
            return self._analyze_meta_pytest_output(output)
        else:
            # Regular pytest scenario - use original logic
            return self._analyze_regular_pytest_output(output)

    def _analyze_regular_pytest_output(self, output) -> tuple[str, bool, int]:
        """
        Original pytest output parsing logic for regular (non-meta) test runs.
        """
        def extract_short_summary(output_text):
            """Extract the short test summary info section from pytest output."""
            summary_pattern = re.compile(r'={5,}\s*short test summary info\s*={5,}', re.IGNORECASE)
            summary_match = summary_pattern.search(output_text)
            
            if not summary_match:
                return ""
            
            summary_start = summary_match.end()
            
            # Find the end of summary section (look for next section with === or end of output)
            end_pattern = re.compile(r'={5,}.*?={5,}', re.IGNORECASE)
            end_match = end_pattern.search(output_text, summary_start + 1)
            
            if end_match:
                summary_end = end_match.start()
            else:
                summary_end = len(output_text)
            
            summary_content = output_text[summary_start:summary_end].strip()
            
            if summary_content:
                # Filter out xfailed lines from the summary
                filtered_lines = []
                for line in summary_content.splitlines():
                    # Skip lines that contain XFail markers
                    if "XFail:" not in line and "xfail" not in line.lower():
                        filtered_lines.append(line)
                
                if filtered_lines:
                    filtered_summary = "\n".join(filtered_lines)
                    return "\n\n=========================== short test summary info ============================\n" + filtered_summary
            
            return ""
        
        try:

            logging.info(f"PYTEST OUTPUT: {output}")
            # If no proper pytest session, try to analyze what we can
            if "test session starts" not in output:
                # Check if we have any failure indicators
                if "ERROR:" in output or "FAILED" in output or "failed" in output:
                    return f"Tests failed with error output: {output[:200]}...", False, 1
                elif "Successfully ran all tests" in output:
                    return "Successfully ran all tests.", True, 0
                else:
                    return f"Unexpected test output (no session start): {output[:100]}...", False, 0
            
            short_summary = extract_short_summary(output)
            logging.info(f"SHORT SUMMARY PYTEST OUTPUT: {short_summary}")
            # if "most likely due to a circular import" in output:
            #     return "Tests failed due to circular import" + short_summary, True, 0
            # # Check for recursion errors first
            # if "RecursionError" in short_summary or "maximum recursion depth" in short_summary:
            #     return "Tests failed due to RecursionError\n\n" + short_summary, True, 0
            
            # Parse the test summary to distinguish actual failures from expected failures
            # Count FAILED lines directly from short test summary
            failed_count = 0
            xfailed_count = 0
            passed_count = 0
            skipped_count = 0
            xpassed_count = 0

            # Count FAILED lines in the short test summary info
            if "short test summary info" in short_summary:
                failed_names = self._extract_failed_test_names(short_summary)
                failed_count = len(failed_names)
            
            # Also try to parse the summary line for other counts if available
            summary_line_pattern = re.compile(r'={3,}.*?\b\d+\.\d+s\s*(\([^)]+\))?\s*={3,}', re.IGNORECASE)
            summary_match = summary_line_pattern.search(short_summary)

            if summary_match:
                summary_line = summary_match.group()
                
                # Extract all "number word" patterns from the summary line
                # This handles any order and missing sections
                result_patterns = re.findall(r'(\d+)\s+(\w+)', summary_line)
                
                for count, result_type in result_patterns:
                    count = int(count)
                    result_type = result_type.lower()
                    
                    # Only update failed_count from summary if we didn't already count from FAILED lines
                    if result_type == 'failed' or result_type == 'error' and 'xfail' not in summary_line.lower():
                        if failed_count == 0:  # Only use summary count if no FAILED lines were found
                            failed_count = count
                    elif result_type == 'xfailed':
                        xfailed_count = count
                    elif result_type == 'passed':
                        passed_count = count
                    elif result_type == 'skipped':
                        skipped_count = count
                    elif result_type == 'xpassed':
                        xpassed_count = count

            print(failed_count, xfailed_count, passed_count, skipped_count, xpassed_count)
            
            # If no actual failures (only expected failures), consider it success
            if failed_count == 0:
                if xfailed_count > 0:
                    return f"Successfully ran all tests.", True, 0
                return "Successfully ran all tests.", True, 0
            
            # Look for failures section
            failure_sections = []
            failures_pattern = re.compile(r'={5,}\s*FAILURES\s*={5,}', re.IGNORECASE)
            errors_pattern = re.compile(r'={5,}\s*ERRORS\s*={5,}', re.IGNORECASE)

            failures_match = failures_pattern.search(output)
            errors_match = errors_pattern.search(output)

            if failures_match:
                failure_sections.append(('FAILURES', failures_match))
            if errors_match:
                failure_sections.append(('ERRORS', errors_match))

            if not failure_sections:
                return f"Tests failed ({failed_count} failures) but failure details not found in output." + short_summary, False, failed_count
            
            # Use the first section found (either FAILURES or ERRORS)
            failures_start = failure_sections[0][1].start()
            
            # Find the end of failures section
            current_section_type = failure_sections[0][0]  # 'FAILURES' or 'ERRORS'
            ending_patterns = [
                re.compile(r'={5,}\s*short test summary info\s*={5,}', re.IGNORECASE),
                re.compile(r'={5,}\s*warnings summary\s*={5,}', re.IGNORECASE),
            ]

            # Only add the opposite section type as ending pattern
            if current_section_type == 'FAILURES':
                ending_patterns.append(re.compile(r'={5,}\s*ERRORS\s*={5,}', re.IGNORECASE))
            elif current_section_type == 'ERRORS':
                ending_patterns.append(re.compile(r'={5,}\s*FAILURES\s*={5,}', re.IGNORECASE))

            failures_end = len(output)
            for pattern in ending_patterns:
                match = pattern.search(output, failures_start + 20)
                if match:
                    failures_end = min(failures_end, match.start())
            
            # Extract the failures content
            failures_content = output[failures_start:failures_end].strip()
            
            if not failures_content:
                return "No failure details found." + short_summary, False, 0
            
            # Split individual test failures - look for test separator lines
            failure_pattern = re.compile(r'_{5,}\s+(.+?)\s+_{5,}')
            failure_separators = list(failure_pattern.finditer(failures_content))
            
            if not failure_separators:
                return failures_content + short_summary, False, 0  # Return as-is if we can't parse it
            
            test_results = []
            total_failures = len(failure_separators)
            number_to_process = 2
            actual_failures_processed = 0
            excluded_count = 0

            # Extract each individual failure, but limit to first 2 VALID (non-excluded) ones
            for i, separator in enumerate(failure_separators):
                # Stop if we already have 2 valid failures
                if actual_failures_processed >= number_to_process:
                    break
                
                test_name = separator.group(1).strip()
                start_pos = separator.end()
                
                if i + 1 < len(failure_separators):
                    end_pos = failure_separators[i + 1].start()
                else:
                    end_pos = len(failures_content)
                
                failure_content = failures_content[start_pos:end_pos].strip()
                
                # Check if this is an expected failure (xfail) by looking for XFAIL markers
                is_xfail = (
                    'XFAIL' in failure_content.upper() or
                    '@pytest.mark.xfail' in failure_content or
                    'xfail' in test_name.lower()
                )
                
                if is_xfail:
                    excluded_count += 1
                    continue
                
                if failure_content:
                    # Just use the original separator format and content
                    full_failure = separator.group() + '\n' + failure_content
                    
                    # Truncate very long individual failures to keep output manageable
                    max_failure_length = 20000  # characters - enough for meaningful debugging
                    if len(full_failure) > max_failure_length:
                        # Smart truncation: keep the beginning (test name, error) and end (actual failure)
                        # Split the failure to preserve the most important parts
                        lines = full_failure.split('\n')
                        
                        # Always keep first 20 lines (test name, setup, initial context)
                        # And last 15 lines (actual error, assertion failure)
                        if len(lines) > 500:  # Only truncate if significantly long
                            start_lines = lines[:400]
                            end_lines = lines[-100:]
                            middle_count = len(lines) - 400
                            
                            truncated_failure = (
                                '\n'.join(start_lines) + 
                                f'\n\n... (truncated {middle_count} lines of detailed traceback) ...\n\n' +
                                '\n'.join(end_lines)
                            )
                            test_results.append(truncated_failure)
                        else:
                            # Not long enough to need smart truncation, use simple truncation
                            truncated_failure = full_failure[:max_failure_length] + f"\n\n... (truncated, full failure was {len(full_failure)} characters)"
                            test_results.append(truncated_failure)
                    else:
                        test_results.append(full_failure)
                    
                    actual_failures_processed += 1
            
            if not test_results:
                if excluded_count > 0:
                    return "Successfully ran all tests.", True, 0
                    # return f"All failures are expected (xfail). {excluded_count} expected failures found." + short_summary, True, 0
                return "Successfully ran all tests.", True, 0
            
            # Add note if there were more failures
            header = "=================================== FAILURES ==================================="
            result = header + '\n' + '\n'.join(test_results)
            
            # Calculate remaining actual failures (excluding expected failures)
            remaining_actual_failures = failed_count - actual_failures_processed
            
            if remaining_actual_failures > 0:
                result += f"\n\n... and {remaining_actual_failures} more actual failures (showing first {number_to_process} failures only)"
            
            return result + short_summary, True, failed_count
            
        except Exception as e:
            print(f"An error occurred during the analysis: {e}")
            return f"Error parsing pytest output: {str(e)}", False, 0
    

    def _extract_debug_prints_from_pytest(self, pytest_output: str) -> dict[str, list[str]]:
        """
        Extract debug print statements from pytest test execution output.
        Simple and safe version that avoids infinite loops.
        """
        debug_prints = {}
        lines = pytest_output.splitlines()
        
        # Pattern to match test function names
        test_name_pattern = r'^([^:]+::[^:\s]+(?:::[^:\s]+)?)'
        
        current_test = None
        current_prints = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line starts a test
            test_match = re.match(test_name_pattern, line)
            if test_match:
                # Save previous test's prints if any
                if current_test and current_prints:
                    debug_prints[current_test] = current_prints.copy()
                
                # Start new test
                current_test = test_match.group(1)
                current_prints = []
                
                # Check for debug output on the same line
                remainder = line[len(current_test):].strip()
                if remainder and remainder not in ['PASSED', 'FAILED', 'ERROR', 'SKIPPED', 'XFAIL', 'XPASS']:
                    current_prints.append(remainder)
            
            # Check if this is a test result line
            elif line_stripped in ['PASSED', 'FAILED', 'ERROR', 'SKIPPED', 'XFAIL', 'XPASS']:
                # Save current test's prints if any
                if current_test and current_prints:
                    debug_prints[current_test] = current_prints.copy()
                current_test = None
                current_prints = []
            
            # Check if we hit a section divider
            elif re.match(r'^={5,}', line_stripped):
                # Save current test's prints if any
                if current_test and current_prints:
                    debug_prints[current_test] = current_prints.copy()
                current_test = None
                current_prints = []
            
            # If we're currently in a test and this is not empty, it's debug output
            elif current_test and line_stripped:
                current_prints.append(line_stripped)
        
        # Handle any remaining test at the end
        if current_test and current_prints:
            debug_prints[current_test] = current_prints
        
        # Filter out FAILED entries - only return debug prints from actual test execution
        filtered_debug_prints = {k: v for k, v in debug_prints.items() if not k.startswith('FAILED')}
        
        return filtered_debug_prints

    def _extract_failed_test_names(self, pytest_output: str, test_files: Optional[List[str]] = None) -> list[str]:
        """
        Extract FAILED test function names from pytest output.
        
        Args:
            pytest_output: String containing pytest output
            test_files: List of test files to filter by
            
        Returns:
            List of failed test names in format "file/path::test_function"
        """
       

        failed_tests = set()
        
        for line in pytest_output.splitlines():
            if 'FAILED' in line:
                lines = line.split(' ')
                # logging.info(f"FOUND FAILED LINE: {line}")
                clean_name = re.sub(r'\x1b\[[0-9;]*m', '', lines[1])
                clean_name = clean_name.split('[')[0]
                if clean_name != '':
                    failed_tests.add(clean_name)
        
        return list(failed_tests)

    @EnhancedToolManager.tool
    def run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            
            error_type=EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
       

        return observation
    
    @EnhancedToolManager.tool
    def apply_code_edit(self,file_path:str, search:str, replace:str)->str:
        '''
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute
            
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: file '{file_path}' does not exist.")
        
        original=self._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.")
            case 1:
                
                new_content = original.replace(search, replace)
                try:
                        is_error,error=self.check_syntax_error(new_content)
                        if not is_error:
                            self.save_file(file_path, new_content)
                                
                            return "ok, code edit applied successfully"
                        else:
                            error.message="code edit failed. "+error.message
                            raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: syntax error in file {file_path}. {e.message}")
            case num_hits:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
    
    @EnhancedToolManager.tool
    def finish(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return "finish"
        else: 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,qa_response.get("analysis",""))

def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(work_dir)
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        pass
    finally:
        os.chdir(original_cwd)

def set_env_for_agent():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, REPO_DIR, DEFAULT_TIMEOUT, run_id
    
    run_id = os.getenv("RUN_ID") or str(uuid4())
    repo_dir = os.path.abspath(repo_dir)

    sys.path.insert(0, repo_dir)


    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    ensure_git_initialized()

    set_env_for_agent()

    try:
        
        problem_type = check_problem_type(input_dict.get("problem_statement"))
        
        logger.info(f"problem_type: {problem_type}")
        if problem_type == PROBLEM_TYPE_CREATE:
            result = process_create_task(input_dict)
        else:
            result = process_fix_task(input_dict, run_id)
    except Exception as e:
        import traceback
        logger.error(f"Error in agent_main: {e}, {traceback.format_exc()}")
        result = process_fix_task(input_dict, run_id)

    os.system("git reset --hard")

    return result

def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            directory_tree = get_directory_tree()

            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{directory_tree}"}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)

            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            retry += 1
        
        time.sleep(2)
    
    if response == None or response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
        num_files = sum(1 for line in directory_tree.splitlines() if '.' in line)
        if num_files > 20:
            return "FIX"
        else:
            return "CREATE"

    return response


def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re
    
    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:

        ```text
        This is a test.

        This is another test!
        ```text

        Then the text block should be:

        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split('\n')
        processed_lines = []
        
        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                # Mark trailing and leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))
                
                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
            
            processed_lines.append(f"\"{processed_line}\"")
        
        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
            
    # Pattern to match ```text...``` blocks
    pattern = r'```text\n(.*?)\n```'
    
    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)
        
        return f'```text\n{processed_content}\n```'
    
    # Replace all text blocks with processed versions
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction

def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    
    while retry < 10:
        try:
            code_response = EnhancedNetwork.make_request(code_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Code Generation completed")
            
            loop_check_messages = [
                {
                    "role": "system",
                    "content": INFINITE_LOOP_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code."
                }   
            ]
            
            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Infinite Loop Check completed")

            # Clean up the final response (use loop check response as it's the final validated version)
            solution = loop_check_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()
            
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning solution generation failed")
        return ""
    
    return ""

def score_solution_quality(solution: str, problem_statement: str) -> float:
    """
    Advanced quality scoring based on AST analysis and problem-specific criteria
    """
    score = 0.0
    
    # Extract code from markdown if present
    code = solution
    if '```python' in solution:
        code_start = solution.find('```python') + 9
        code_end = solution.find('```', code_start)
        if code_end != -1:
            code = solution[code_start:code_end].strip()
        else:
            code = solution[code_start:].strip()
    
    # 1. SYNTAX VALIDATION (Critical - 30 points)
    try:
        tree = ast.parse(code)
        score += 30.0  # Valid syntax is essential
        
        # Analyze AST structure for quality indicators
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # 2. FUNCTIONAL COMPLETENESS (25 points)
        if functions:
            score += 15.0  # Has functions
            # Check for proper function structure
            for func in functions:
                if func.args.args:  # Has parameters
                    score += 2.0
                if any(isinstance(node, ast.Return) for node in ast.walk(func)):
                    score += 3.0  # Has return statements
                if len(func.body) > 1:  # Substantial function body
                    score += 2.0
        
        if classes:
            score += 10.0  # Has classes
            for cls in classes:
                if any(isinstance(node, ast.FunctionDef) for node in cls.body):
                    score += 3.0  # Class has methods
        
        # 3. IMPORT ANALYSIS (10 points)
        if imports:
            score += 5.0  # Has imports
            # Check for appropriate imports
            import_names = []
            for imp in imports:
                if isinstance(imp, ast.Import):
                    import_names.extend([alias.name for alias in imp.names])
                elif isinstance(imp, ast.ImportFrom):
                    import_names.extend([alias.name for alias in imp.names])
            
            # Bonus for standard library imports
            stdlib_imports = ['os', 'sys', 'json', 're', 'math', 'collections', 'itertools', 'functools']
            if any(imp in stdlib_imports for imp in import_names):
                score += 3.0
            
            # Bonus for problem-specific imports
            if any(imp in ['unittest', 'pytest', 'mock'] for imp in import_names):
                score += 2.0
        
        # 4. CONTROL FLOW ANALYSIS (15 points)
        if_statements = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
        loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
        try_except = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        
        if if_statements:
            score += 5.0  # Has conditional logic
            if len(if_statements) > 1:
                score += 2.0  # Complex conditional logic
        
        if loops:
            score += 5.0  # Has loops
            if len(loops) > 1:
                score += 2.0  # Complex iteration
        
        if try_except:
            score += 8.0  # Has error handling
        
        # 5. PROBLEM-SPECIFIC ANALYSIS (20 points)
        problem_lower = problem_statement.lower()
        code_lower = code.lower()
        
        # Check if solution addresses problem requirements
        if 'test' in problem_lower and 'test' in code_lower:
            score += 5.0
        if 'function' in problem_lower and 'def ' in code_lower:
            score += 5.0
        if 'class' in problem_lower and 'class ' in code_lower:
            score += 5.0
        if 'import' in problem_lower and 'import ' in code_lower:
            score += 3.0
        if 'return' in problem_lower and 'return ' in code_lower:
            score += 2.0
        
        # 6. CODE COMPLEXITY & SOPHISTICATION (10 points)
        # Check for advanced Python features
        if any(isinstance(node, ast.ListComp) for node in ast.walk(tree)):
            score += 2.0  # List comprehensions
        if any(isinstance(node, ast.DictComp) for node in ast.walk(tree)):
            score += 2.0  # Dict comprehensions
        if any(isinstance(node, ast.GeneratorExp) for node in ast.walk(tree)):
            score += 2.0  # Generator expressions
        if any(isinstance(node, ast.Lambda) for node in ast.walk(tree)):
            score += 2.0  # Lambda functions
        if any(isinstance(node, ast.Decorator) for node in ast.walk(tree)):
            score += 2.0  # Decorators
        
    except SyntaxError:
        score -= 50.0  # Severe penalty for syntax errors
        return max(0.0, score)
    except Exception as e:
        logger.warning(f"Error analyzing solution: {str(e)}")
        score += 10.0  # Partial credit for non-syntax issues
    
    # 7. LENGTH AND COMPLETENESS (10 points)
    if len(code) > 200:
        score += 5.0  # Substantial solution
    elif len(code) > 100:
        score += 3.0  # Moderate solution
    elif len(code) < 50:
        score -= 10.0  # Penalty for very short solutions
    
    # 8. DOCUMENTATION AND CLARITY (5 points)
    if '"""' in code or "'''" in code:
        score += 3.0  # Has docstrings
    if '#' in code and code.count('#') > 2:
        score += 2.0  # Has comments
    
    return max(0.0, score)  # Ensure non-negative score


def evaluate_solutions_with_llm(solutions: List[str], problem_statement: str, code_skeleton: str) -> str:
    """
    Advanced LLM evaluation with detailed analysis and multiple validation steps
    """
    models = [QWEN_MODEL_NAME]
    
    # Create comprehensive evaluation prompt
    comparison_prompt = f"""You are a senior Python developer and code reviewer with expertise in software engineering best practices. You need to evaluate and select the HIGHEST QUALITY solution from {len(solutions)} options.

PROBLEM STATEMENT:
{problem_statement}

CODE SKELETON/REQUIREMENTS:
{code_skeleton}

EVALUATION CRITERIA (in order of importance):
1. **CORRECTNESS**: Does the solution correctly implement the requirements?
2. **COMPLETENESS**: Does it handle all specified cases and edge cases?
3. **CODE QUALITY**: Is the code clean, readable, and well-structured?
4. **BEST PRACTICES**: Does it follow Python conventions and best practices?
5. **ERROR HANDLING**: Does it properly handle errors and edge cases?
6. **PERFORMANCE**: Is the solution efficient and optimized?
7. **MAINTAINABILITY**: Is the code easy to understand and modify?

SOLUTIONS TO EVALUATE:

"""
    
    for i, solution in enumerate(solutions, 1):
        comparison_prompt += f"=== SOLUTION {i} ===\n{solution}\n\n"
    
    comparison_prompt += """DETAILED ANALYSIS REQUIRED:
For each solution, consider:
- Does it solve the exact problem described?
- Are there any logical errors or bugs?
- Is the code style consistent and professional?
- Are there any security vulnerabilities?
- Does it handle edge cases properly?
- Is the solution efficient and scalable?

RESPONSE FORMAT:
BEST_SOLUTION: [number]
CONFIDENCE: [high/medium/low]
REASONING: [detailed explanation of why this solution is superior]
STRENGTHS: [key advantages of the chosen solution]
WEAKNESSES: [any potential issues with the chosen solution]
ALTERNATIVES: [brief mention of other viable options and why they're inferior]"""
    
    try:
        comparison_messages = [
            {
                "role": "system", 
                "content": """You are a world-class Python developer and code reviewer with 15+ years of experience. You have a reputation for identifying the highest quality, most robust solutions. You are thorough, analytical, and prioritize correctness and maintainability above all else. You understand that choosing the wrong solution can lead to bugs, security issues, and technical debt."""
            },
            {
                "role": "user",
                "content": comparison_prompt
            }
        ]
        
        response = EnhancedNetwork.make_request(comparison_messages, model=models[0])
        logger.info(f"LLM evaluation response: {response[:200]}...")
        
        # Parse the response to extract the selected solution
        best_solution_match = re.search(r'BEST_SOLUTION:\s*(\d+)', response)
        if best_solution_match:
            selected_index = int(best_solution_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= selected_index < len(solutions):
                logger.info(f"LLM selected solution {selected_index + 1} as the best")
                
                # Extract confidence level
                confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response)
                confidence = confidence_match.group(1) if confidence_match else "unknown"
                logger.info(f"LLM confidence level: {confidence}")
                
                # If confidence is low, consider using scoring as backup
                if confidence.lower() == "low":
                    logger.warning("LLM has low confidence, considering backup selection")
                    # Could implement backup logic here
                
                return solutions[selected_index]
            else:
                logger.warning(f"Invalid solution index {selected_index + 1}, using scoring fallback")
                return solutions[0]  # Fallback to first solution
        else:
            logger.warning("Could not parse LLM response, using scoring fallback")
            return solutions[0]  # Fallback to first solution
            
    except Exception as e:
        logger.error(f"Error in LLM solution comparison: {str(e)}, using scoring fallback")
        return solutions[0]  # Fallback to first solution


def improved_solution_selection(solutions: List[str], problem_statement: str, code_skeleton: str) -> str:
    """
    Quality-first solution selection prioritizing accuracy over speed/cost
    """
    if len(solutions) <= 1:
        return solutions[0] if solutions else ""
    
    logger.info(f"Evaluating {len(solutions)} solutions for quality...")
    
    # Strategy 1: Consensus check (only if we have strong consensus)
    solution_counts = Counter(solutions)
    if len(solution_counts) < len(solutions):  # We have duplicates
        consensus_solution = max(solution_counts.keys(), key=solution_counts.get)
        consensus_count = solution_counts[consensus_solution]
        total_solutions = len(solutions)
        
        # Only use consensus if it's a strong majority (>= 60%)
        if consensus_count >= max(2, total_solutions * 0.6):
            logger.info(f"Strong consensus found: solution appeared {consensus_count}/{total_solutions} times ({consensus_count/total_solutions*100:.1f}%)")
            return consensus_solution
        else:
            logger.info(f"Weak consensus: {consensus_count}/{total_solutions} times, proceeding with detailed analysis")
    
    # Strategy 2: Advanced quality scoring
    scored_solutions = []
    for i, solution in enumerate(solutions, 1):
        score = score_solution_quality(solution, problem_statement)
        scored_solutions.append((solution, score))
        logger.info(f"Solution {i} quality score: {score:.2f}")
    
    # Sort by quality score
    scored_solutions.sort(key=lambda x: x[1], reverse=True)
    
    # Strategy 3: Multi-level LLM validation for quality assurance
    top_solutions = scored_solutions[:3]  # Top 3 solutions
    
    # If top solution has significantly higher score, use it
    if len(top_solutions) > 1:
        score_diff = top_solutions[0][1] - top_solutions[1][1]
        if score_diff >= 15.0:  # Significant quality difference
            logger.info(f"Clear winner: top solution has {score_diff:.2f} point advantage")
            return top_solutions[0][0]
    
    # For close scores or when we want maximum quality assurance, use LLM
    logger.info("Close quality scores detected, using LLM for detailed analysis")
    top_solution_candidates = [sol for sol, _ in top_solutions]
    
    # Use LLM to make the final quality decision
    selected_solution = evaluate_solutions_with_llm(top_solution_candidates, problem_statement, code_skeleton)
    
    # Log the final decision
    selected_score = next((score for sol, score in scored_solutions if sol == selected_solution), 0)
    logger.info(f"Final selection: solution with quality score {selected_score:.2f}")
    
    return selected_solution


def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    
    models = [QWEN_MODEL_NAME]
    
    # Generate three different solutions
    solutions = []
    retry = 0
    
    while len(solutions) < 5 and retry < 20:
        try:
            logger.info(f"Generating solution {len(solutions) + 1}/5")
            
            # Try multi-step reasoning first
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton)
            
            if solution:
                solutions.append(solution)
                logger.info(f"Generated solution {len(solutions)} using multi-step reasoning")
            else:
                # Fallback to single-step approach
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=models[0])
                
                # Clean up the response
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()
                
                if solution:
                    solutions.append(solution)
                    logger.info(f"Generated solution {len(solutions)} using fallback approach")
            
        except Exception as e:
            logger.error(f"Error generating solution {len(solutions) + 1}: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if not solutions:
        logger.error("Failed to generate any solutions")
        return ""
    
    # Use improved solution selection
    logger.info(f"Generated {len(solutions)} solutions, using improved selection strategy")
    return improved_solution_selection(solutions, problem_statement, code_skeleton)


def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    
    created_files = []
    
    if not initial_solution.strip():
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()
        
        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):  # Not a comment
            
            # Write the previous file if we have one
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                # Create directory if needed (for subdirectories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Join content and remove empty lines at start/end
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
            
            # Start new file
            current_filename = stripped_line
            current_content = []
        else:
            # This line is content for the current file
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
    
    return created_files

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Testcase Generation completed")
            
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                }   
            ]
            
            testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Testcase check completed")

            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning testcase generation failed")
        return ""
    
    return ""

def evaluate_testcases_with_llm(testcases: List[str], problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    """
    Advanced LLM evaluation for test case quality with detailed analysis
    """
    models = [QWEN_MODEL_NAME]
    
    # Create comprehensive test evaluation prompt
    comparison_prompt = f"""You are a senior software engineer and testing expert with 15+ years of experience in writing comprehensive, robust test suites. You need to evaluate and select the HIGHEST QUALITY test case from {len(testcases)} options.

PROBLEM STATEMENT:
{problem_statement}

FILES TO TEST:
{files_to_test}

CODE SKELETON/IMPLEMENTATION:
{code_skeleton}

EVALUATION CRITERIA (in order of importance):
1. **TEST COVERAGE**: Does it test all specified functionality and edge cases?
2. **CORRECTNESS**: Are the test assertions accurate and meaningful?
3. **EDGE CASE HANDLING**: Does it test boundary conditions, error cases, and edge scenarios?
4. **TEST STRUCTURE**: Is the test code well-organized and follows best practices?
5. **ASSERTION QUALITY**: Are assertions specific, clear, and comprehensive?
6. **MAINTAINABILITY**: Is the test code readable and easy to maintain?
7. **ROBUSTNESS**: Will the tests catch regressions and bugs effectively?

TEST CASES TO EVALUATE:

"""
    
    for i, testcase in enumerate(testcases, 1):
        comparison_prompt += f"=== TEST CASE {i} ===\n{testcase}\n\n"
    
    comparison_prompt += """DETAILED ANALYSIS REQUIRED:
For each test case, consider:
- Does it comprehensively test the problem requirements?
- Are there sufficient edge cases and boundary conditions?
- Are the assertions meaningful and will catch real bugs?
- Is the test structure professional and maintainable?
- Are there any missing test scenarios?
- Is the test code efficient and well-written?

RESPONSE FORMAT:
BEST_TESTCASE: [number]
CONFIDENCE: [high/medium/low]
REASONING: [detailed explanation of why this test case is superior]
COVERAGE: [assessment of test coverage completeness]
EDGE_CASES: [evaluation of edge case testing]
QUALITY: [assessment of test code quality and best practices]
MISSING: [any important test scenarios that might be missing]"""
    
    try:
        comparison_messages = [
            {
                "role": "system", 
                "content": """You are a world-class software testing expert with 15+ years of experience. You have a reputation for writing comprehensive, bulletproof test suites that catch bugs before they reach production. You understand that poor test quality leads to false confidence and production failures. You prioritize thoroughness, accuracy, and maintainability in test design."""
            },
            {
                "role": "user",
                "content": comparison_prompt
            }
        ]
        
        response = EnhancedNetwork.make_request(comparison_messages, model=models[0])
        logger.info(f"LLM test case evaluation response: {response[:200]}...")
        
        # Parse the response to extract the selected test case
        best_testcase_match = re.search(r'BEST_TESTCASE:\s*(\d+)', response)
        if best_testcase_match:
            selected_index = int(best_testcase_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= selected_index < len(testcases):
                logger.info(f"LLM selected test case {selected_index + 1} as the best")
                
                # Extract confidence level
                confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response)
                confidence = confidence_match.group(1) if confidence_match else "unknown"
                logger.info(f"LLM confidence level: {confidence}")
                
                return testcases[selected_index]
            else:
                logger.warning(f"Invalid test case index {selected_index + 1}, using scoring fallback")
                return testcases[0]  # Fallback to first test case
        else:
            logger.warning("Could not parse LLM response, using scoring fallback")
            return testcases[0]  # Fallback to first test case
            
    except Exception as e:
        logger.error(f"Error in LLM test case comparison: {str(e)}, using scoring fallback")
        return testcases[0]  # Fallback to first test case


def improved_testcase_selection(testcases: List[str], problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    """
    Quality-first test case selection using LLM evaluation
    """
    if len(testcases) <= 1:
        return testcases[0] if testcases else ""
    
    logger.info(f"Evaluating {len(testcases)} test cases for quality...")
    
    # Strategy 1: Consensus check (only if we have strong consensus)
    testcase_counts = Counter(testcases)
    if len(testcase_counts) < len(testcases):  # We have duplicates
        consensus_testcase = max(testcase_counts.keys(), key=testcase_counts.get)
        consensus_count = testcase_counts[consensus_testcase]
        total_testcases = len(testcases)
        
        # Only use consensus if it's a strong majority (>= 60%)
        if consensus_count >= max(2, total_testcases * 0.6):
            logger.info(f"Strong consensus found: test case appeared {consensus_count}/{total_testcases} times ({consensus_count/total_testcases*100:.1f}%)")
            return consensus_testcase
        else:
            logger.info(f"Weak consensus: {consensus_count}/{total_testcases} times, proceeding with LLM analysis")
    
    # Strategy 2: Use LLM for quality evaluation
    logger.info("Using LLM for detailed test case analysis")
    
    # Use LLM to make the final quality decision
    selected_testcase = evaluate_testcases_with_llm(testcases, problem_statement, files_to_test, code_skeleton)
    
    logger.info("Final test case selection completed using LLM evaluation")
    
    return selected_testcase


def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    """
    Generate 5 test case candidates and select the best one using quality-first approach
    """
    models = [QWEN_MODEL_NAME]
    
    # Generate 5 different test case candidates
    testcases = []
    retry = 0
    
    while len(testcases) < 5 and retry < 15:
        try:
            logger.info(f"Generating test case {len(testcases) + 1}/5")
            
            # Try multi-step reasoning first
            testcase = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)
            
            if testcase:
                testcases.append(testcase)
                logger.info(f"Generated test case {len(testcases)} using multi-step reasoning")
            else:
                # Fallback to single-step approach
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate comprehensive test cases with edge cases and boundary conditions."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=models[0])
                
                # Clean up the response
                testcase = response.strip()
                if testcase.startswith('```python'):
                    testcase = testcase[9:]
                if testcase.startswith('```'):
                    testcase = testcase[3:]
                if testcase.endswith('```'):
                    testcase = testcase[:-3]
                testcase = testcase.strip()
                
                if testcase:
                    testcases.append(testcase)
                    logger.info(f"Generated test case {len(testcases)} using fallback approach")
            
        except Exception as e:
            logger.error(f"Error generating test case {len(testcases) + 1}: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if not testcases:
        logger.error("Failed to generate any test cases")
        return ""
    
    # Use improved test case selection
    logger.info(f"Generated {len(testcases)} test cases, using quality-first selection strategy")
    return improved_testcase_selection(testcases, problem_statement, files_to_test, code_skeleton)

def process_create_task(input_dict):
    global run_id
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)
    print(problem_statement)

    code_skeleton = get_code_skeleton()
    start_time = time.time()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton)
    print(initial_solution)
    
    # Extract and write files from the solution
    created_files = extract_and_write_files(initial_solution)
    print(f"Created or Updated {len(created_files)} files: {created_files}")

    
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    print(test_cases)
    # Extract and write files from test cases
    test_files = extract_and_write_files(test_cases)
    print(f"Created or Updated {len(test_files)} files: {test_files}")

    timeout = DEFAULT_TIMEOUT - (time.time()-start_time) - 60

    patch = fix_task_solve_workflow(
        problem_statement=problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=30,
        instance_prompt=CREATE_TASK_INSTANCE_PROMPT_TEMPLATE,
        system_prompt=CREATE_TASK_SYSTEM_PROMPT,
        task_type="create",
        test_paths=None,
        file_paths=None
    )

    if patch is None:
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()

    logger.info(f"Generated patch create:\n{patch}")

    return patch

def get_code_skeleton() -> str:
    # Initialize the result string
    result = ""
    
    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"
    
    return result

def get_directory_tree(start_path: str = '.') -> str:

    tree_lines = []
    
    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            # Get the directory name
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            
            # Add current directory to tree (skip for root directory)
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            
            # Get all items in directory
            try:
                items = os.listdir(path)
                # Filter out hidden directories and files starting with '.'
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                
                # Separate directories and files
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                
                # Process directories first
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                
                # Then process files
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                    
            except PermissionError:
                # Handle directories we can't read
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")
                
        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    
    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)

def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
    current_dir = os.path.dirname(file_path)
    
    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None

def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": FIND_TEST_RUNNER_PROMPT},
            {"role": "user", "content": readme_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "pytest"
    except Exception as e:
        return "pytest"

def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    # Remove extension and make relative to repo
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    # Adjust relative to test runner directory if needed
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, '.')

def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path

def get_test_runner_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"

    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": TEST_RUNNER_MODE_PROMPT},
            {"role": "user", "content": runner_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "FILE"
    except Exception as e:
        return "FILE"

def count_test_cases(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)
    
    except (FileNotFoundError, UnicodeDecodeError):
        return 0

def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None
    
    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        return "pytest", "FILE"

    readme_file_path = find_readme(test_file_path, '.')
    if readme_file_path:
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)

    return test_runner, test_runner_mode

def process_fix_task(input_dict: Dict[str, Any], run_id):
    """Main entry point for task processing and code modification.

    Parameters
    ----------
    input_dict : dict
        Configuration dictionary containing the task specification.
        Required key: 'problem_statement' with task details.
        Optional keys: 'run_id' for tracking purposes.
    """
    # setting environment to include current working directory and lib directory
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split('/')[-1]
    repod_path = repo_path[:-len(repod_dir)-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)

    set_env_for_agent()
    cwd = os.getcwd()
    
    test_runner, test_runner_mode = get_test_runner_and_mode()
    logger.info(f"Detected test runner: {test_runner} with mode: {test_runner_mode}")
    file_paths = None
    test_paths = None
    try:
        test_func_names, _logs_patch_find_workflow = test_find_workflow(
            problem_text,
            timeout=timeout, 
            run_id_1=run_id, 
            instance_id=input_dict.get("instance_id", "")
        )
    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"Error in test_patch_find_workflow{error_info}")
        test_func_names = []
    try:
        patch_text= fix_task_solve_workflow(
            problem_text,
            timeout=timeout,
            run_id_1=run_id,
            test_runner=test_runner,
            test_runner_mode=test_runner_mode,
            file_paths=file_paths,
            test_paths=test_func_names,
            instance_prompt=FIX_TASK_INSTANCE_PROMPT_TEMPLATE,
            system_prompt=FIX_TASK_SYSTEM_PROMPT,
            task_type="fix"
        )
        logger.info(f"SWE patch: {patch_text}")

    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"Error in solve_workflow{error_info}")
    finally:
        os.chdir(cwd)

    return patch_text

TEST_PATCH_FIND_SYSTEM_PROMPT_TEMPLATE_V0 = textwrap.dedent("""
# 🧠 Test Function Finder
You are a code analysis expert tasked with identifying test functions that directly validate the issue described in the problem statement. Follow this structured workflow carefully.

**🔍 Step-by-Step Process**
1. **Problem Analysis** 
   - Parse the problem statement carefully.
   - Identify all affected functions, classes, and modules.
   - Determine expected input/output and behavioral changes.

2. **Test Discovery**
   - Use `search_in_all_files_content` extensively with multiple search strategies and keywords.
   - Use `search_in_specified_file` to confirm or reject test relevance.
   - Explore all possible naming patterns (test_* functions, methods inside Test* classes, parameterized tests, etc.).
   - Be thorough — aim to **exhaustively discover all potentially relevant tests**.

3. **Filtering & Ranking** 
   - Filter out irrelevant tests that do not align with the problem statement.
   - Rank remaining tests by:
     - **Specificity** (directly related to the affected code or bug)
     - **Coverage** (breadth of scenarios)
     - **Isolation** (minimal unrelated setup)
   - Prefer multiple tests if they collectively validate different aspects of the same problem.

4. **Validation**
   - Use `run_repo_tests` to execute **each potentially relevant test** individually.
   - Run as many relevant tests as possible — aim for **maximum diagnostic coverage**.
   - Only include tests that **fail** or clearly exercise the bug described in the problem statement.
   - Remember to execute tests by module path list, e.g.:
     `["path.class.function", "another_path.class.function"]`.

**🛠️ Available Tools**
- `search_in_all_files_content`: Find test patterns and references across the repo.
- `search_in_specified_file`: Verify the context of test definitions.
- `run_repo_tests`: Execute tests to confirm if they expose the issue.
- `get_file_content`: Retrieve the full test function code.
- `test_patch_find_finish`: Finalize and return the list of relevant tests.

**⚠️ Critical Rules**
- Only include tests that are clearly and contextually relevant to the given `problem_statement`.
- Be **comprehensive** — explore all test files and try to **exhaustively identify every relevant case**.
- Do **not** include unrelated or passing tests that have no logical connection to the described issue.
- Prioritize tests that:
  - Contain explicit assertions matching the bug context.
  - Are small, focused, and reproducible.
- If no direct match is found, include the **closest possible candidates** that validate similar behavior.
- Always use exact tool names and input schemas from the documentation.
- Never invent tool names or parameters.

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")


PATCH_FIND_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
[CRITICAL FIRST DECISION FOCUS]

Problem Statement:
{problem_statement}

""")

def test_find_workflow(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "") -> tuple[List[str], List[str]]:
    run_id = run_id_1
    cot=EnhancedCOT(latest_observations_to_keep=40)
    MAX_STEPS_TEST_PATCH_FIND = 40
    tool_manager=FixTaskEnhancedToolManager(
        available_tools=[
            "search_in_all_files_content",
            "get_file_content",
            "search_in_specified_file",
            "test_patch_find_finish",
            "run_repo_tests",
        ],

    )
    logger.info(f"[TEST_PATCH_FIND] Starting test patch find agent execution...")
    system_prompt = TEST_PATCH_FIND_SYSTEM_PROMPT_TEMPLATE_V0.format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
    instance_prompt = PATCH_FIND_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)

    #QA.SYSTEM_PROMPT=QA.SYSTEM_PROMPT.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []

    for step in range(MAX_STEPS_TEST_PATCH_FIND):
        logger.info(f"[TEST_PATCH_FIND] Execution step {step + 1}/{MAX_STEPS_TEST_PATCH_FIND}")
        
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break
        
        logs.append(f"Execution step {step + 1}/{MAX_STEPS_TEST_PATCH_FIND}\n\n")

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
    
        try:
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=GLM_MODEL_NAME,run_id=run_id)
            logs.append(f"next_thought: {next_thought}\n\nnext_tool_name: {next_tool_name}\n\nnext_tool_args: {next_tool_args}\n\n")
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logs.append(f"Inference error: {error_msg}\n\n")
            logger.error(f"[TEST_PATCH_FIND] Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text,total_attempts=total_attempts, inference_error_counter=error_counter,request_data=messages))
            break
        
        logger.info(f"[TEST_PATCH_FIND] About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"[TEST_PATCH_FIND] next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logs.append(f"next_observation: {next_observation}\n\n")
            logger.info(f"[TEST_PATCH_FIND] next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logs.append(f"Tool error: {error_msg}\n\n")
            logger.error(f"[TEST_PATCH_FIND] Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logs.append(f"Tool error: {error_msg}\n\n")
            logger.error(f"[TEST_PATCH_FIND] Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == "test_patch_find_finish" and next_observation == 'finish':
            test_func_names = next_tool_args["test_func_names"]
            logger.info(f'[TEST_PATCH_FIND] [CRITICAL] Workflow called test_patch_find_finish operation with test_func_names: {test_func_names}')
            logs.append(f"Workflow called test_patch_find_finish operation with test_func_names: {test_func_names}\n\n")
            return test_func_names, logs
            
        print(f"[TEST_PATCH_FIND] [CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        # This happens if we exit the loop without breaking (reached MAX_STEPS)
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
        logger.info(f"[TEST_PATCH_FIND] [CRITICAL] Workflow completed after reaching MAX_STEPS ({MAX_STEPS_TEST_PATCH_FIND})")

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str, \
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS, file_paths: List[str], test_paths: List[str] , instance_prompt, system_prompt,task_type:str) -> tuple[str, List[str], List[str]]:
    logger.info(f"Starting {task_type} task workflow: timeout={timeout}s, max_steps={n_max_steps}, run_id={run_id_1}")
    cot=EnhancedCOT(latest_observations_to_keep=500)
    logger.info(f"test_paths found: {test_paths}")
    if task_type == "fix":
        tool_manager=FixTaskEnhancedToolManager(
            test_runner=test_runner,
            test_runner_mode=test_runner_mode,
            available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "search_in_all_files_content",
            "search_in_specified_file",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "finish"
            ],
        )
        instance_prompt = instance_prompt.format(
            problem_statement=problem_statement,
            test_func_names = ', '.join(test_paths) if test_paths else 'None',
        )
        system_prompt = system_prompt.format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
    else:
        tool_manager=FixTaskEnhancedToolManager(
            test_runner=test_runner,
            test_runner_mode=test_runner_mode,
            available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "search_in_all_files_content",
            "search_in_specified_file",
            "run_repo_tests_create",
            "run_code",
            "apply_code_edit",
            "generate_test_function",
            "finish"
            ],
        )
        instance_prompt = instance_prompt.format(problem_statement=problem_statement)
        system_prompt = system_prompt.format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    
    for step in range(n_max_steps):
        logger.info(f"Starting step {step+1}/{n_max_steps} (elapsed: {time.time() - start_time:.1f}s)")
        
        if time.time() - start_time > timeout:
            logger.warning(f"Global timeout reached at step {step+1} ({timeout}s)")
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())

        messages.append({"role": "system", "content": STOP_INSTRUCTION})
    
        if cot.is_thought_repeated():
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
    
        try:
            logger.info(f"Step {step+1}/{n_max_steps}: Getting inference from LLM")
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=GLM_MODEL_NAME, run_id=run_id_1)
            logger.info(f"LLM inference successful: tool={next_tool_name}, thought_length={len(next_thought) if next_thought else 0}")
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text,total_attempts=total_attempts),inference_error_counter=error_counter,request_data=messages)
            break
        
        try:
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
            
            logger.info(f"Executing tool: {next_tool_name} with args: {next_tool_args}")
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()

            logger.info(f"Tool execution completed: {next_tool_name}\n\n Observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logger.error(f"Tool {next_tool_name} failed with EnhancedToolManager.Error: {e.message}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
                logger.error(f"Tool {next_tool_name} failed with TypeError: {str(e)}")
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
                logger.error(f"Tool {next_tool_name} failed with Exception: {repr(e)}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == "finish":
            logger.info("Agent finished successfully")
            break
    else:
        logger.warning(f"Agent reached maximum steps ({n_max_steps}) or timeout")
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
        if n_max_steps < MAX_FIX_TASK_STEPS: # This is create task case and failed with smaller fix steps so try to use original solution supposing generated testcases are wrong
            logger.info("Returning None for create task case")
            return None
    
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Workflow completed after {step+1 if 'step' in locals() else 'unknown'} steps, patch length: {len(patch) if patch else 0} chars")

    return patch