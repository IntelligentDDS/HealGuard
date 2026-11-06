from __future__ import annotations

import re
from pathlib import PurePosixPath


FRAME = re.compile(r"(?m)^(?P<file>\S+\.(?:py|pyx)|<string>):(?P<line>\d+):(?P<detail>[^\n]*)")
HELPERS = (
    "numpy/testing/", "numpy/ma/testutils.py", "pandas/_testing.py",
    "pandas/_testing/", "pandas/util/testing.py", "pandas/_libs/testing.pyx",
    "sklearn/utils/testing.py", "unittest/case.py", "nose/tools/trivial.py",
    "_pytest/raises.py", "_pytest/python_api.py",
)


def is_test_file(file, target_files):
    path = PurePosixPath(file)
    return (file in target_files or any(file.endswith('/' + target) for target in target_files)
            or 'r2e_tests' in path.parts or 'tests' in path.parts or 'test' in path.parts
            or path.name.startswith('test_') or path.name.endswith('_test.py'))


def is_assertion_helper(file):
    normalized = '/' + file.lstrip('./')
    return any('/' + helper in normalized if helper.endswith('/')
               else normalized.endswith('/' + helper) for helper in HELPERS)


def assertion_evidence(text, record):
    """Classify the tested call's completion at the failing test oracle, not the remaining test body."""
    start = re.search(r'(?m)^=+ FAILURES =+$', text)
    if start is None:
        return None, 'missing_failure_section', {}
    end = re.search(r'(?m)^=+ (?:warnings summary|short test summary info|\d+ failed).*$', text[start.end():])
    stop = start.end() + end.start() if end else len(text)
    section = text[start.end():stop]
    blocks = re.split(r'(?m)^_{3,} .+ _{3,}\s*$', section)
    blocks = [block for block in blocks if re.search(r'(?m)^E\s+', block)]
    evidence = {'log_character_range': [start.end(), stop], 'blocks': []}
    if len(blocks) != 1:
        return None, 'multiple_or_missing_failure_blocks', evidence
    block = blocks[0]
    chain = re.split(r'During handling of the above exception, another exception occurred:|The above exception was the direct cause of the following exception:', block)[-1]
    frames = [{'file': match['file'], 'line': int(match['line']), 'detail': match['detail'].strip()}
              for match in FRAME.finditer(chain)]
    hidden = [{'file': match[0], 'function': match[1]}
              for match in re.findall(r"frame = <frame .*?file '([^']+)'.*?code ([^>]+)>", chain)]
    errors = re.findall(r'(?m)^E[ \t]+((?:[\w.]+(?:Error|Exception)|Failed)(?::[^\n]*)?)[ \t]*$', chain)
    terminal = frames[-1]['detail'] if frames else ''
    # Rewritten pytest assertions may only name AssertionError in the terminal location.
    if not errors and re.match(r'AssertionError\b', terminal):
        errors = [terminal]
    evidence['blocks'].append({'frames': frames, 'hidden_frames': hidden, 'terminal_errors': errors})
    if not frames or not errors:
        return None, 'missing_terminal_frame_or_exception', evidence
    error = errors[-1]
    if not (error == 'AssertionError' or error.startswith(('AssertionError:', 'Failed: DID NOT RAISE'))):
        return None, 'not_assertion_failure', evidence
    exception_oracle = bool(re.search(r'\b(?:assert_raises\w*|assertRaises\w*)\s*\(|\bpytest\.raises\s*\(', chain))
    # Ordinary result assertions may also say "does not match"; require an exception oracle.
    if exception_oracle and re.search(r'does not match|Regex pattern did not match|pattern .* not found', error, re.I):
        return 0, 'target_raised_exception_message_mismatch', evidence
    target_files = set()
    if record.get('test_file'):
        target_files.add(record['test_file'].lstrip('./'))
    for field in ('fail_to_pass', 'FAIL_TO_PASS'):
        tests = record.get(field, [])
        if isinstance(tests, str):
            # Paths can still be recognized without interpreting serialized test lists.
            tests = re.findall(r'[\w./-]+\.py(?:::.*?)?(?=[\"\s,\]]|$)', tests)
        for test in tests:
            target_files.add(test.split('::')[0].lstrip('./'))
    files = [frame['file'] for frame in frames if frame['file'] != '<string>']
    files.extend(frame['file'] for frame in hidden)
    patched_files = re.findall(r'(?m)^\+\+\+ b/(.+)$', record.get('patch', ''))
    if any(file == patched or file.endswith('/' + patched) for file in files for patched in patched_files):
        return 0, 'assertion_inside_target_call', evidence
    if not any(is_test_file(file, target_files) for file in files):
        return None, 'no_test_oracle_frame', evidence
    other_files = [file for file in files if not is_test_file(file, target_files) and not is_assertion_helper(file)]
    if other_files:
        return 0, 'assertion_inside_target_call', evidence
    if frames[-1]['file'] == '<string>' and not hidden:
        return None, 'unresolved_instrumentation_frame', evidence
    if error.startswith('Failed: DID NOT RAISE') or (exception_oracle and 'not raised' in error):
        return 1, 'target_returned_expected_exception_missing', evidence
    if 'No warning raised' in error:
        return 1, 'target_returned_expected_warning_missing', evidence
    # Non-oracle frames would expose an assertion thrown while evaluating an argument.
    # The failed oracle establishes completion of that tested invocation, not later invocations.
    highlighted = re.findall(r'(?m)^>\s*(.*)$', chain)
    oracle_call = any(re.search(r'\bassert\b|\bassert\w*\s*\(|\beq_\s*\(|self\.assert\w*\s*\(', line)
                      for line in highlighted)
    helper_frame = any(is_assertion_helper(file) for file in files)
    multiline_oracle = bool(re.search(r'(?m)^\s+(?:self\.)?assert\w*\s*\(', chain))
    if not (oracle_call or helper_frame or multiline_oracle):
        return None, 'unrecognized_oracle_statement', evidence
    return 1, 'target_reached_test_oracle', evidence
