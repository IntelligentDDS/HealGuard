from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFS_ROOT = PROJECT_ROOT / "refs"
RULES_ROOT = PROJECT_ROOT / "scripts" / "sink_rules"
SOURCES_ROOT = RULES_ROOT / "sources"
GENERATED_ROOT = RULES_ROOT / "generated"
OFFICIAL_RULES = RULES_ROOT / "official_rules.yaml"
FINAL_RULES = RULES_ROOT / "final_rules.yaml"
DEFAULT_RULES = FINAL_RULES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def bandit_sources() -> list[Path]:
    root = REFS_ROOT / "bandit"
    files = [root / "setup.cfg", *(root / "bandit").rglob("*.py")]
    return sorted(path for path in files if path.is_file())

def semgrep_sources() -> list[Path]:
    root = REFS_ROOT / 'semgrep-rules' / 'python'
    return sorted([*root.rglob('*.yml'), *root.rglob('*.yaml')])

def pysa_sources() -> list[Path]:
    root = REFS_ROOT / 'pyre-check' / 'stubs' / 'taint'
    return sorted((path for path in root.rglob('*') if path.is_file()))

def codeql_sources() -> list[Path]:
    root = REFS_ROOT / 'codeql' / 'python' / 'ql'
    suffixes = {'.ql', '.qll', '.yml', '.yaml', '.qls', '.qlref', '.qhelp'}
    files = [path for path in root.rglob('*') if path.is_file() and path.suffix in suffixes]
    for name in ('qlpack.yml', 'codeql-pack.yml', 'codeql-pack.lock.yml'):
        files.extend((path for path in root.rglob(name) if path.is_file()))
    return sorted(set(files))

def source_groups() -> dict[str, tuple[Path, list[Path]]]:
    return {'bandit': (REFS_ROOT / 'bandit', bandit_sources()), 'semgrep': (REFS_ROOT / 'semgrep-rules' / 'python', semgrep_sources()), 'pysa': (REFS_ROOT / 'pyre-check' / 'stubs' / 'taint', pysa_sources()), 'codeql': (REFS_ROOT / 'codeql' / 'python' / 'ql', codeql_sources())}

def clear_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

def copy_sources(output: Path) -> dict:
    clear_output(output)
    records = []
    counts = {}
    for tool, (source_root, paths) in source_groups().items():
        counts[tool] = len(paths)
        for source in paths:
            relative_path = source.relative_to(source_root)
            destination = output / tool / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hash = sha256(source)
            destination_hash = sha256(destination)
            if source_hash != destination_hash or source.stat().st_size != destination.stat().st_size:
                raise RuntimeError(f'Copy verification failed: {source}')
            records.append({'tool': tool, 'source': source.relative_to(PROJECT_ROOT).as_posix(), 'destination': destination.relative_to(PROJECT_ROOT).as_posix(), 'size': source.stat().st_size, 'sha256': source_hash})
    manifest = {'counts': counts, 'total': len(records), 'files': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest

SINK_KINDS = {'CODE_EXECUTION', 'COMMAND_PROCESS', 'FILESYSTEM', 'NETWORK_IPC', 'DATABASE', 'SENSITIVE_DATA_ACCESS', 'DATA_DISCLOSURE', 'CREDENTIAL_IDENTITY', 'PERMISSION_AUTHORITY', 'SYSTEM_ENVIRONMENT_CONFIGURATION', 'PERSISTENT_SHARED_STATE', 'SCHEDULING_CONCURRENCY', 'AVAILABILITY_RESOURCE', 'UNKNOWN_EFFECT'}

NATIVE_KIND_MAP = {'CodeInjection': 'CODE_EXECUTION', 'UnsafeDeserialization': 'CODE_EXECUTION', 'CommandInjection': 'COMMAND_PROCESS', 'PathInjection': 'FILESYSTEM', 'TarSlip': 'FILESYSTEM', 'ServerSideRequestForgery': 'NETWORK_IPC', 'UrlRedirect': 'NETWORK_IPC', 'LdapInjection (DN)': 'NETWORK_IPC', 'LdapInjection (Filter)': 'NETWORK_IPC', 'SqlInjection': 'DATABASE', 'NoSqlInjection (string sink)': 'DATABASE', 'NoSqlInjection (dict sink)': 'DATABASE', 'CleartextLogging': 'DATA_DISCLOSURE', 'CleartextStorage': 'PERSISTENT_SHARED_STATE', 'LogInjection': 'DATA_DISCLOSURE', 'ReflectedXss': 'DATA_DISCLOSURE', 'StackTraceExposure': 'DATA_DISCLOSURE', 'WeakSensitiveDataHashing (NormalHashFunction)': 'CREDENTIAL_IDENTITY', 'WeakSensitiveDataHashing (ComputationallyExpensiveHashFunction)': 'CREDENTIAL_IDENTITY', 'PolynomialReDoS': 'AVAILABILITY_RESOURCE', 'RegexInjection': 'AVAILABILITY_RESOURCE', 'XmlBomb': 'AVAILABILITY_RESOURCE', 'XpathInjection': 'CODE_EXECUTION', 'Xxe': 'DATA_DISCLOSURE', 'RemoteCodeExecution': 'CODE_EXECUTION', 'ExecArgSink': 'COMMAND_PROCESS', 'ExecEnvSink': 'COMMAND_PROCESS', 'ExecImportSink': 'CODE_EXECUTION', 'ExecDeserializationSink': 'CODE_EXECUTION', 'FileContentDeserializationSink': 'CODE_EXECUTION', 'FileSystem_ReadWrite': 'FILESYSTEM', 'FileSystem_Other': 'FILESYSTEM', 'HTTPClientRequest': 'NETWORK_IPC', 'HTTPClientRequest_DATA': 'NETWORK_IPC', 'HTTPClientRequest_METADATA': 'NETWORK_IPC', 'HTTPClientRequest_URI': 'NETWORK_IPC', 'SQL': 'DATABASE', 'Logging': 'DATA_DISCLOSURE', 'ReturnedToUser': 'DATA_DISCLOSURE', 'ResponseHeaderName': 'DATA_DISCLOSURE', 'ResponseHeaderValue': 'DATA_DISCLOSURE', 'XSS': 'DATA_DISCLOSURE', 'CookieWrite': 'CREDENTIAL_IDENTITY', 'Authentication': 'CREDENTIAL_IDENTITY', 'DataStorage': 'PERSISTENT_SHARED_STATE', 'MemcacheSink': 'PERSISTENT_SHARED_STATE'}

BANDIT_KIND_MAP = {'B102': 'CODE_EXECUTION', 'B202': 'FILESYSTEM', 'B301': 'CODE_EXECUTION', 'B302': 'CODE_EXECUTION', 'B307': 'CODE_EXECUTION', 'B310': 'NETWORK_IPC', 'B311': 'AVAILABILITY_RESOURCE', 'B312': 'NETWORK_IPC', 'B313': 'CODE_EXECUTION', 'B314': 'CODE_EXECUTION', 'B315': 'CODE_EXECUTION', 'B316': 'CODE_EXECUTION', 'B317': 'CODE_EXECUTION', 'B318': 'CODE_EXECUTION', 'B319': 'CODE_EXECUTION', 'B320': 'CODE_EXECUTION', 'B321': 'CODE_EXECUTION', 'B401': 'NETWORK_IPC', 'B402': 'NETWORK_IPC', 'B506': 'CODE_EXECUTION', 'B601': 'COMMAND_PROCESS', 'B602': 'COMMAND_PROCESS', 'B603': 'COMMAND_PROCESS', 'B604': 'COMMAND_PROCESS', 'B605': 'COMMAND_PROCESS', 'B606': 'COMMAND_PROCESS', 'B607': 'COMMAND_PROCESS', 'B608': 'DATABASE', 'B609': 'COMMAND_PROCESS', 'B610': 'DATABASE', 'B611': 'DATABASE', 'B612': 'NETWORK_IPC', 'B615': 'FILESYSTEM', 'B702': 'CODE_EXECUTION', 'B703': 'DATA_DISCLOSURE', 'B704': 'DATA_DISCLOSURE'}

KEYWORDS = [('COMMAND_PROCESS', ('command injection', 'command-injection', 'shell', 'subprocess', 'os.system', 'process execution')), ('CODE_EXECUTION', ('code injection', 'code-injection', 'deserial', 'pickle', 'marshal', 'yaml.load', 'eval', 'template injection', 'ssti', 'remote code execution', ' rce')), ('FILESYSTEM', ('path injection', 'path-injection', 'path traversal', 'tar slip', 'file system', 'filesystem', 'file permission', 'temp file', 'archive extraction')), ('NETWORK_IPC', ('ssrf', 'request forgery', 'request-forgery', 'socket', 'network', 'http client', 'redirect', 'url redirect', 'ldap', 'thrift')), ('DATABASE', ('sql injection', 'sql-injection', 'nosql', 'database', 'mongodb', 'sqlalchemy', 'rawsql')), ('SENSITIVE_DATA_ACCESS', ('secret', 'sensitive data', 'key vault', 'keyvault', 'privacy', 'credential read')), ('DATA_DISCLOSURE', ('xss', 'cleartext', 'logging', 'log injection', 'response header', 'returned to user', 'email send', 'stack trace', 'exposure')), ('CREDENTIAL_IDENTITY', ('password', 'authentication', 'cookie', 'jwt', 'credential', 'identity', 'session')), ('PERMISSION_AUTHORITY', ('permission', 'chmod', 'chown', 'privilege', 'authorization', 'access control')), ('SYSTEM_ENVIRONMENT_CONFIGURATION', ('configuration', 'environment', 'debug', 'certificate validation', 'tls', 'ssl', 'host key', 'sys.path')), ('PERSISTENT_SHARED_STATE', ('storage', 'cache', 'memcache', 'shared state', 'registry', 'data storage')), ('SCHEDULING_CONCURRENCY', ('thread', 'async', 'concurrency', 'callback', 'scheduler', 'fork')), ('AVAILABILITY_RESOURCE', ('redos', 'denial of service', 'resource', 'timeout', 'xml bomb', 'billion laughs', 'availability'))]

def map_kind(*values: Any) -> str:
    for value in values:
        if str(value) in NATIVE_KIND_MAP:
            return NATIVE_KIND_MAP[str(value)]
        if str(value) in BANDIT_KIND_MAP:
            return BANDIT_KIND_MAP[str(value)]
    text = ' '.join((str(value) for value in values if value is not None)).lower()
    for sink_kind, keywords in KEYWORDS:
        if any((keyword in text for keyword in keywords)):
            return sink_kind
    return 'UNKNOWN_EFFECT'

def git_version(path: Path) -> str:
    result = subprocess.run(['git', '-C', str(path), 'describe', '--tags', '--always'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'

def bandit_rules() -> list[dict[str, Any]]:
    records = []
    for path in sorted((SOURCES_ROOT / 'bandit/bandit/plugins').glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            test_id = ''
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and (decorator.func.attr == 'test_id') and decorator.args and isinstance(decorator.args[0], ast.Constant):
                    test_id = str(decorator.args[0].value)
            if test_id:
                doc = ast.get_docstring(node) or ast.get_docstring(tree) or ''
                records.append({'tool': 'bandit', 'native_rule_id': test_id, 'native_name': node.name, 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': 'ASTTestPlugin', 'sink_kind': map_kind(test_id, node.name, doc), 'evidence': doc})
    blacklist_root = SOURCES_ROOT / 'bandit/bandit/blacklists'
    for path in sorted((blacklist_root / name for name in ('calls.py', 'imports.py'))):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'build_conf_dict':
                continue
            name = ast.literal_eval(node.args[0])
            test_id = ast.literal_eval(node.args[1])
            cwe = ast.unparse(node.args[2])
            qualnames = ast.literal_eval(node.args[3])
            message = constants[node.args[4].id] if isinstance(node.args[4], ast.Name) else ast.literal_eval(node.args[4])
            level = ast.literal_eval(node.args[5]) if len(node.args) == 6 else 'MEDIUM'
            rule = {'name': name, 'id': test_id, 'cwe': cwe, 'message': message, 'qualnames': qualnames, 'level': level}
            node_types = ('Call',) if path.name == 'calls.py' else ('Call', 'Import', 'ImportFrom')
            for node_type in node_types:
                records.append({'tool': 'bandit', 'native_rule_id': test_id, 'native_name': name, 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': f'Blacklist{node_type}', 'sink_kind': map_kind(test_id, name, message, qualnames), 'raw_rule': rule})
    return records

def semgrep_rules() -> list[dict[str, Any]]:
    records = []
    root = SOURCES_ROOT / 'semgrep'
    for path in sorted([*root.rglob('*.yml'), *root.rglob('*.yaml')]):
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        for rule in data.get('rules', []) if isinstance(data, dict) else []:
            metadata = rule.get('metadata') or {}
            records.append({'tool': 'semgrep', 'native_rule_id': rule.get('id', ''), 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': 'SemgrepRule', 'sink_kind': map_kind(rule.get('id'), rule.get('message'), metadata), 'raw_rule': rule})
    return records

def pysa_rules() -> list[dict[str, Any]]:
    records = []
    root = SOURCES_ROOT / 'pysa'
    for path in sorted(root.rglob('taint.config')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for sink in data.get('sinks', []):
            name = sink.get('name', '') if isinstance(sink, dict) else str(sink)
            records.append({'tool': 'pysa', 'native_rule_id': name, 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': 'TaintSink', 'sink_kind': map_kind(name, sink), 'raw_rule': sink})
        for rule in data.get('rules', []):
            records.append({'tool': 'pysa', 'native_rule_id': str(rule.get('code', rule.get('name', ''))), 'native_name': rule.get('name', ''), 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': 'TaintRule', 'sink_kind': map_kind(rule.get('name'), rule.get('sinks'), rule.get('message_format')), 'raw_rule': rule})
    annotation = re.compile('(?:TaintSink|TaintInTaintOut|Sanitize|PartialSink)\\[([^\\]]+)\\]')
    callable_pattern = re.compile('^(?:async\\s+)?def\\s+([^(:]+)|^class\\s+([^(:]+)')
    for path in sorted(root.rglob('*.pysa')):
        current_callable = ''
        for line_number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            callable_match = callable_pattern.match(line.strip())
            if callable_match:
                current_callable = next((group for group in callable_match.groups() if group))
            kinds = annotation.findall(line)
            if kinds:
                records.append({'tool': 'pysa', 'native_rule_id': f'{path.name}:{line_number}', 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'line': line_number, 'source_type': 'PysaModel', 'sink_kind': map_kind(kinds, line), 'native_kinds': kinds, 'native_callable': current_callable, 'raw_model': line})
    return records

def codeql_rules() -> list[dict[str, Any]]:
    records = []
    model_root = SOURCES_ROOT / 'codeql/lib'
    for path in sorted(model_root.rglob('*.model.yml')):
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        for extension in data.get('extensions', []):
            model_type = extension.get('addsTo', {}).get('extensible', '')
            if model_type not in {'sourceModel', 'sinkModel', 'typeModel', 'summaryModel'}:
                continue
            for index, row in enumerate(extension.get('data', [])):
                native_kind = row[-1] if row else ''
                records.append({'tool': 'codeql', 'native_rule_id': f'{path.name}:{model_type}:{index}', 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': model_type, 'sink_kind': map_kind(native_kind, row), 'native_kind': native_kind, 'raw_model': row})
    sinks_path = SOURCES_ROOT / 'codeql/src/meta/alerts/Sinks.qll'
    for native_kind in re.findall('kind\\s*=\\s*"([^"]+)"', sinks_path.read_text(encoding='utf-8')):
        records.append({'tool': 'codeql', 'native_rule_id': native_kind, 'source_path': sinks_path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': 'taintSink', 'sink_kind': map_kind(native_kind), 'native_kind': native_kind})
    for path in sorted((SOURCES_ROOT / 'codeql/src').rglob('*.ql')):
        header = path.read_text(encoding='utf-8', errors='replace')[:6000]
        if '@kind' not in header or ('@tags' not in header and '/Security/' not in path.as_posix()):
            continue
        query_id = re.search('@id\\s+([^\\s*]+)', header)
        name = re.search('@name\\s+([^\\n*]+)', header)
        tags = re.findall('@tags?\\s+([^\\n*]+)', header)
        records.append({'tool': 'codeql', 'native_rule_id': query_id.group(1) if query_id else path.stem, 'native_name': name.group(1).strip() if name else path.stem, 'source_path': path.relative_to(PROJECT_ROOT).as_posix(), 'source_type': 'Query', 'sink_kind': map_kind(path, name.group(1) if name else '', tags), 'tags': tags})
    return records

def add_catalog_keys(records: list[dict[str, Any]]) -> None:
    used_keys: dict[str, int] = {}
    for record in records:
        parts = [record['tool'], record['native_rule_id'], record['source_type'], record['source_path']]
        if record.get('line') is not None:
            parts.append(str(record['line']))
        base_key = '|'.join(parts)
        occurrence = used_keys.get(base_key, 0) + 1
        used_keys[base_key] = occurrence
        record['catalog_key'] = base_key if occurrence == 1 else f'{base_key}|{occurrence}'

def import_official_rules() -> dict[str, Any]:
    records = bandit_rules() + semgrep_rules() + pysa_rules() + codeql_rules()
    add_catalog_keys(records)
    return {
        'versions': {
            name: git_version(REFS_ROOT / path)
            for name, path in {
                'bandit': 'bandit',
                'semgrep': 'semgrep-rules',
                'pysa': 'pyre-check',
                'codeql': 'codeql',
            }.items()
        },
        'records': records,
    }

SECURITY_CATEGORIES = {'security', 'secrets'}

CALL_PATTERN = re.compile('(?<![\\w$])([A-Za-z_][\\w.]*)\\s*\\(')

PYSA_ANNOTATION = re.compile('(TaintSink|TaintInTaintOut|Sanitize|PartialSink)\\[([^\\]]+)\\]')

PYSA_PARAMETER = re.compile('([A-Za-z_]\\w*)\\s*:\\s*(TaintSink|TaintInTaintOut|Sanitize|PartialSink)\\[')

CODEQL_ARGUMENT = re.compile('Argument\\[([^\\]]+)\\]')

def canonical_id(catalog_key: str) -> str:
    digest = hashlib.sha256(catalog_key.encode()).hexdigest()[:20]
    return f'RH.OFFICIAL.{digest}'

def nested_values(value: Any, key: str) -> list[Any]:
    values = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                values.append(current_value)
            values.extend(nested_values(current_value, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(nested_values(item, key))
    return values

def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for value_item in value for item in strings(value_item)]
    if isinstance(value, dict):
        return [item for value_item in value.values() for item in strings(value_item)]
    return []

def semgrep_semantics(record: dict[str, Any]) -> dict[str, Any]:
    rule = record['raw_rule']
    metadata = rule.get('metadata') or {}
    category = metadata.get('category')
    has_taint_sink = bool(rule.get('pattern-sinks'))
    role = 'TAINT_SINK' if has_taint_sink else 'SECURITY_DETECTOR'
    pattern_fields = sorted((key for key in rule if key.startswith('pattern')))
    pattern_text = strings({key: rule[key] for key in pattern_fields})
    call_targets = sorted({match.group(1) for text in pattern_text for match in CALL_PATTERN.finditer(text)})
    focused = sorted(set(strings(nested_values(rule, 'focus-metavariable'))))
    return {'semantic_role': role, 'enabled_for_sink_detection': category in SECURITY_CATEGORIES and record['sink_kind'] != 'UNKNOWN_EFFECT', 'matcher': {'mode': rule.get('mode', 'search'), 'call_targets': call_targets, 'focused_metavariables': focused, 'pattern_fields': pattern_fields}, 'dangerous_ports': focused or ['matched_expression'], 'conditions': {'has_exclusions': any((key.startswith('pattern-not') for key in pattern_fields)), 'has_inside_constraints': any(('inside' in key for key in pattern_fields)), 'has_metavariable_constraints': bool(nested_values(rule, 'metavariable-regex') or nested_values(rule, 'metavariable-comparison')), 'has_sanitizers': bool(rule.get('pattern-sanitizers')), 'has_propagators': bool(rule.get('pattern-propagators'))}}

def bandit_semantics(record: dict[str, Any]) -> dict[str, Any]:
    raw_rule = record.get('raw_rule') or {}
    targets = sorted(set(raw_rule.get('qualnames') or []))
    return {'semantic_role': 'AST_SINK_DETECTOR' if record['sink_kind'] != 'UNKNOWN_EFFECT' else 'NON_SINK_DETECTOR', 'enabled_for_sink_detection': record['sink_kind'] != 'UNKNOWN_EFFECT', 'matcher': {'node_type': record['source_type'], 'call_targets': targets}, 'dangerous_ports': ['official_plugin_defined'], 'conditions': {'implemented_by_official_plugin': record['source_type'] == 'ASTTestPlugin'}}

def codeql_semantics(record: dict[str, Any]) -> dict[str, Any]:
    source_type = record['source_type']
    raw_model = record.get('raw_model') or []
    role_map = {'sinkModel': 'TAINT_SINK', 'sourceModel': 'TAINT_SOURCE', 'summaryModel': 'PROPAGATION_SUMMARY', 'typeModel': 'TYPE_RELATION', 'taintSink': 'SINK_KIND_DECLARATION', 'Query': 'SECURITY_QUERY'}
    matcher: dict[str, Any] = {}
    flow: dict[str, Any] = {}
    dangerous_ports = []
    if source_type in {'sinkModel', 'sourceModel'} and len(raw_model) >= 2:
        matcher = {'type': raw_model[0], 'path': raw_model[1]}
        dangerous_ports = CODEQL_ARGUMENT.findall(str(raw_model[1])) if source_type == 'sinkModel' else []
    elif source_type == 'summaryModel' and len(raw_model) >= 5:
        matcher = {'type': raw_model[0], 'path': raw_model[1]}
        flow = {'input': raw_model[2], 'output': raw_model[3], 'kind': raw_model[4]}
    elif source_type == 'typeModel' and len(raw_model) >= 3:
        flow = {'type': raw_model[0], 'subtype': raw_model[1], 'path': raw_model[2]}
    return {'semantic_role': role_map[source_type], 'enabled_for_sink_detection': source_type in {'sinkModel', 'taintSink'} and record['sink_kind'] != 'UNKNOWN_EFFECT', 'matcher': matcher, 'dangerous_ports': dangerous_ports, 'flow': flow, 'conditions': {}}

def pysa_semantics(record: dict[str, Any]) -> dict[str, Any]:
    source_type = record['source_type']
    model = record.get('raw_model', '')
    annotations = [{'annotation': name, 'kind': kind} for name, kind in PYSA_ANNOTATION.findall(model)]
    roles = {item['annotation'] for item in annotations}
    if 'TaintSink' in roles or 'PartialSink' in roles:
        role = 'TAINT_SINK'
    elif 'TaintInTaintOut' in roles:
        role = 'PROPAGATION_SUMMARY'
    elif 'Sanitize' in roles:
        role = 'SANITIZER'
    elif source_type == 'TaintSink':
        role = 'SINK_KIND_DECLARATION'
    elif source_type == 'TaintRule':
        role = 'TAINT_FLOW_RULE'
    else:
        role = 'MODEL_DEFINITION'
    callable_match = re.search('(?:def|class)\\s+([^(:]+)', model)
    callable_name = callable_match.group(1).strip() if callable_match else record.get('native_callable')
    dangerous_ports = [name for name, annotation in PYSA_PARAMETER.findall(model) if annotation in {'TaintSink', 'PartialSink'}]
    return {'semantic_role': role, 'enabled_for_sink_detection': role == 'TAINT_SINK' and record['sink_kind'] != 'UNKNOWN_EFFECT', 'matcher': {'callable': callable_name, 'annotations': annotations}, 'dangerous_ports': dangerous_ports, 'conditions': {}}

def definition_fingerprint(record: dict[str, Any]) -> str:
    native_definition = record.get('raw_rule', record.get('raw_model'))
    if native_definition is None:
        native_definition = {'catalog_key': record['catalog_key']}
    payload = {'source_type': record['source_type'], 'native_definition': native_definition, 'native_callable': record.get('native_callable')}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(serialized.encode()).hexdigest()

def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalizer = {'bandit': bandit_semantics, 'semgrep': semgrep_semantics, 'pysa': pysa_semantics, 'codeql': codeql_semantics}[record['tool']]
    semantics = normalizer(record)
    role = semantics['semantic_role']
    if role in {'TAINT_SINK', 'AST_SINK_DETECTOR', 'SECURITY_DETECTOR', 'SINK_KIND_DECLARATION'}:
        direct_or_wrapper = 'direct'
    elif role == 'PROPAGATION_SUMMARY':
        direct_or_wrapper = 'wrapper'
    else:
        direct_or_wrapper = 'supporting'
    return {'canonical_rule_id': canonical_id(record['catalog_key']), 'official_rule_reference': record['catalog_key'], 'tool': record['tool'], 'native_rule_id': record['native_rule_id'], 'native_definition_type': record['source_type'], 'native_definition_fingerprint': definition_fingerprint(record), 'source_path': record['source_path'], 'sink_kind': record['sink_kind'], 'direct_or_wrapper': direct_or_wrapper, 'evidence': {'kind': 'vendored_official_definition', 'official_rule_reference': record['catalog_key']}, **semantics}

def normalize_official_rules(official: dict[str, Any]) -> dict[str, Any]:
    return {
        'official_versions': official['versions'],
        'rules': [normalize_record(record) for record in official['records']],
    }

def normalize_project_rule(rule: dict[str, Any]) -> dict[str, Any]:
    matcher = {key: rule[key] for key in ('callee', 'callee_regex', 'method', 'method_regex', 'target_regex') if key in rule}
    dangerous_argument = rule['dangerous_argument']
    dangerous_ports = dangerous_argument if isinstance(dangerous_argument, list) else [dangerous_argument]
    return {'runtime_healing_rule_id': rule['rule_id'], 'rule_origin': 'project', 'sink_kind': rule['sink_kind'], 'semantic_role': 'PROJECT_SINK_RULE' if rule['sink_kind'] != 'UNKNOWN_EFFECT' else 'PROJECT_UNKNOWN_RULE', 'enabled_for_sink_detection': rule['sink_kind'] != 'UNKNOWN_EFFECT', 'direct_or_wrapper': rule['direct_or_wrapper'], 'matcher': matcher, 'dangerous_ports': dangerous_ports, 'conditions': rule['guard'], 'evidence': {'kind': 'project_definition', 'value': rule['evidence']}, 'native_definition': rule}

def semantic_signature(rule: dict[str, Any]) -> str:
    payload = {
        'sink_kind': rule['sink_kind'],
        'matcher': rule.get('matcher', {}),
        'dangerous_ports': rule.get('dangerous_ports', []),
        'conditions': rule.get('conditions', {}),
        'native_definition_fingerprint': rule.get('native_definition_fingerprint'),
        'project_rule_id': rule['runtime_healing_rule_id'] if rule['rule_origin'] == 'project' else None,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def deduplicate_rules(rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        groups.setdefault(semantic_signature(rule), []).append(rule)
    deduplicated = []
    aliases = {}
    for group in groups.values():
        primary = min(group, key=lambda rule: (rule['rule_origin'] != 'project', rule['runtime_healing_rule_id']))
        primary_id = primary['runtime_healing_rule_id']
        member_ids = sorted({rule['runtime_healing_rule_id'] for rule in group})
        for member_id in member_ids:
            aliases[member_id] = primary_id
        merged = {
            **primary,
            'runtime_healing_rule_id': primary_id,
            'aliases': member_ids,
            'origins': sorted({rule['rule_origin'] for rule in group}),
            'tools': sorted({tool for rule in group for tool in rule.get('tools', [rule.get('tool')]) if tool}),
            'official_rule_references': sorted({reference for rule in group for reference in rule.get('official_rule_references', [rule.get('official_rule_reference')]) if reference}),
            'source_paths': sorted({source for rule in group for source in rule.get('source_paths', [rule.get('source_path')]) if source}),
            'evidence': [evidence for rule in group for evidence in (rule['evidence'] if isinstance(rule['evidence'], list) else [rule['evidence']])],
        }
        for key in ('canonical_rule_id', 'official_rule_reference', 'source_path', 'tool'):
            merged.pop(key, None)
        deduplicated.append(merged)
    deduplicated.sort(key=lambda rule: rule['runtime_healing_rule_id'])
    return deduplicated, aliases


def project_rule_supported_by_tool(rule: dict[str, Any], tool: str) -> bool:
    native = rule['native_definition']
    if tool in {'bandit', 'semgrep'}:
        return native['template'] == 'assignment' or any(key in native for key in ('callee', 'callee_regex', 'method', 'method_regex'))
    if tool in {'pysa', 'codeql'}:
        dangerous_argument = native.get('dangerous_argument')
        supported_port = dangerous_argument == 'receiver' or isinstance(dangerous_argument, int) or (
            isinstance(dangerous_argument, list) and all(isinstance(port, int) for port in dangerous_argument)
        )
        return bool(native.get('callee')) and supported_port and native['sink_kind'] != 'UNKNOWN_EFFECT'
    return False


def canonical_target(rule: dict[str, Any]) -> str | None:
    matcher = rule.get('matcher', {})
    for key in ('callee', 'callee_regex', 'method', 'method_regex', 'target_regex', 'callable'):
        if matcher.get(key):
            return f'{key}:{matcher[key]}'
    call_targets = matcher.get('call_targets', [])
    if len(call_targets) == 1:
        return f'callable:{call_targets[0]}'
    model_type = str(matcher.get('type', '')).removesuffix('!').removeprefix('~')
    model_path = str(matcher.get('path', ''))
    member = re.search(r'Member\[([^,\]]+)\]', model_path)
    if model_type and member:
        return f'callable:{model_type}.{member.group(1)}'
    if model_type and model_path.startswith('Call.Argument['):
        return f'callable:{model_type}'
    return None


def canonical_ports(rule: dict[str, Any]) -> list[str]:
    ports = []
    for port in rule.get('dangerous_ports', []):
        value = str(port)
        argument = re.match(r'^(?:argument:)?(\d+)', value, re.IGNORECASE)
        ports.append(f'argument:{argument.group(1)}' if argument else value)
    return sorted(set(ports))


def canonical_merge_key(rule: dict[str, Any]) -> str:
    if 'project' in rule.get('origins', []):
        return f"project:{rule['runtime_healing_rule_id']}"
    return f"official:{rule['runtime_healing_rule_id']}"


def build_all_catalog(tool_rules: list[dict[str, Any]], sink_kinds: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rule in tool_rules:
        groups.setdefault(canonical_merge_key(rule), []).append(rule)
    canonical_rules = []
    aliases = {}
    for group in groups.values():
        primary = min(group, key=lambda rule: ('project' not in rule.get('origins', []), rule['runtime_healing_rule_id']))
        member_ids = sorted({rule['runtime_healing_rule_id'] for rule in group})
        canonical_id = primary['runtime_healing_rule_id'] if len(member_ids) == 1 or 'project' in primary.get('origins', []) else canonical_id_for_group(member_ids)
        for member_id in member_ids:
            aliases[member_id] = canonical_id
        canonical_rules.append({
            'runtime_healing_rule_id': canonical_id,
            'sink_kind': primary['sink_kind'],
            'target': canonical_target(primary),
            'dangerous_ports': canonical_ports(primary),
            'conditions': primary.get('conditions', {}),
            'supported_tools': sorted({tool for rule in group for tool in rule.get('tools', [])}),
            'aliases': member_ids,
            'implementations': [
                {
                    'rule_id': rule['runtime_healing_rule_id'],
                    'tools': rule.get('tools', []),
                    'matcher': rule.get('matcher', {}),
                    'dangerous_ports': rule.get('dangerous_ports', []),
                    'official_rule_references': rule.get('official_rule_references', []),
                    'source_paths': rule.get('source_paths', []),
                    'evidence': rule.get('evidence', []),
                }
                for rule in group
            ],
        })
    canonical_rules.sort(key=lambda rule: rule['runtime_healing_rule_id'])
    return {
        'version': 1,
        'tools': ['bandit', 'semgrep', 'pysa', 'codeql'],
        'sink_kinds': sink_kinds,
        'counts': {
            'source_total': len(tool_rules),
            'total': len(canonical_rules),
            'duplicates_removed': len(tool_rules) - len(canonical_rules),
            'enabled_for_sink_detection': len(canonical_rules),
        },
        'aliases': aliases,
        'rules': canonical_rules,
    }


def canonical_id_for_group(member_ids: list[str]) -> str:
    digest = hashlib.sha256('\n'.join(member_ids).encode()).hexdigest()[:20]
    return f'RH.MERGED.{digest}'


def build_rule_catalogs(final_rules: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    normalized_human = [normalize_project_rule(rule) for rule in final_rules['rules']]
    all_tool_rules = []
    summary = {}
    for tool in ('bandit', 'semgrep', 'pysa', 'codeql'):
        source_rules = [
            {**rule, 'tool': tool}
            for rule in normalized_human
            if rule['enabled_for_sink_detection'] and project_rule_supported_by_tool(rule, tool)
        ]
        rules, aliases = deduplicate_rules(source_rules)
        payload = {
            'version': 3,
            'tool': tool,
            'source': 'final_rules',
            'sink_kinds': final_rules['sink_kinds'],
            'counts': {
                'source_total': len(source_rules),
                'total': len(rules),
                'duplicates_removed': len(source_rules) - len(rules),
                'human': len(rules),
                'official': 0,
                'enabled_for_sink_detection': len(rules),
            },
            'aliases': aliases,
            'rules': rules,
        }
        output = output_dir / f'{tool}_rules.json'
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary[tool] = {**payload['counts'], 'output': output.as_posix()}
        all_tool_rules.extend(rules)
    total_payload = build_all_catalog(all_tool_rules, final_rules['sink_kinds'])
    total_payload['source'] = 'final_rules'
    total_path = output_dir / 'all_rules.json'
    total_path.write_text(json.dumps(total_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary['all'] = {**total_payload['counts'], 'output': total_path.as_posix()}
    summary['final'] = {
        'source_total': len(final_rules['rules']),
        'total': len(final_rules['rules']),
        'enabled_for_sink_detection': sum(rule['sink_kind'] != 'UNKNOWN_EFFECT' for rule in final_rules['rules']),
        'output': FINAL_RULES.as_posix(),
    }
    return summary

VALID_KINDS = {'CODE_EXECUTION', 'COMMAND_PROCESS', 'FILESYSTEM', 'NETWORK_IPC', 'DATABASE', 'SENSITIVE_DATA_ACCESS', 'DATA_DISCLOSURE', 'CREDENTIAL_IDENTITY', 'PERMISSION_AUTHORITY', 'SYSTEM_ENVIRONMENT_CONFIGURATION', 'PERSISTENT_SHARED_STATE', 'SCHEDULING_CONCURRENCY', 'AVAILABILITY_RESOURCE', 'UNKNOWN_EFFECT'}

CALL_TEMPLATES = {'direct_call', 'conditional_call', 'family_call', 'framework_family', 'method_call'}

REQUIRED_FIELDS = {'rule_id', 'sink_kind', 'template', 'fully_qualified_callee_or_event', 'receiver', 'dangerous_argument', 'guard', 'direct_or_wrapper', 'evidence'}

def load_rules(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if path.resolve() == FINAL_RULES.resolve():
        if data.get('source') != 'human_review':
            raise ValueError('final_rules.yaml must declare source: human_review')
        if data.get('reviewed_from') != OFFICIAL_RULES.relative_to(PROJECT_ROOT).as_posix():
            raise ValueError('final_rules.yaml must reference official_rules.yaml')
    if set(data.get('sink_kinds', [])) != VALID_KINDS:
        raise ValueError('sink_kinds must contain the complete Runtime Healing taxonomy')
    rule_ids = set()
    targets = set()
    effects_by_kind: dict[str, set[str]] = {}
    for rule in data.get('rules', []):
        missing = REQUIRED_FIELDS - rule.keys()
        if missing:
            raise ValueError(f"{rule.get('rule_id', '<unknown>')} missing fields: {sorted(missing)}")
        if rule['rule_id'] in rule_ids:
            raise ValueError(f"duplicate rule_id: {rule['rule_id']}")
        review = rule.get('review')
        if not isinstance(review, dict) or review.get('status') != 'confirmed':
            raise ValueError(f"{rule['rule_id']} must have confirmed human review")
        if rule['sink_kind'] not in VALID_KINDS:
            raise ValueError(f"invalid sink_kind: {rule['sink_kind']}")
        if rule['template'] in CALL_TEMPLATES and (not any((key in rule for key in ('callee', 'callee_regex', 'method', 'method_regex')))):
            raise ValueError(f"{rule['rule_id']} has no call matcher")
        dangerous_argument = rule['dangerous_argument']
        valid_port = dangerous_argument in {'receiver', 'all'} if isinstance(dangerous_argument, str) else isinstance(dangerous_argument, int) or (
            isinstance(dangerous_argument, list) and dangerous_argument and all(isinstance(port, int) and port >= 0 for port in dangerous_argument)
        )
        if not valid_port:
            raise ValueError(f"{rule['rule_id']} has invalid dangerous_argument: {dangerous_argument}")
        effect_key = rule.get('effect_key')
        if not effect_key:
            raise ValueError(f"{rule['rule_id']} missing effect_key")
        kind_effects = effects_by_kind.setdefault(rule['sink_kind'], set())
        if effect_key in kind_effects:
            raise ValueError(f"duplicate effect_key in {rule['sink_kind']}: {effect_key}")
        target = rule['fully_qualified_callee_or_event']
        if target in targets:
            raise ValueError(f"duplicate target across human rules: {target}")
        rule_ids.add(rule['rule_id'])
        kind_effects.add(effect_key)
        targets.add(target)
    return data

def module_and_path(callee: str, dangerous_argument: Any) -> tuple[str, str] | None:
    parts = callee.split('.')
    module, members = ('builtins', parts[1:]) if callee.startswith('builtins.') else (parts[0], parts[1:]) if len(parts) > 1 else ('builtins', parts)
    if dangerous_argument == 'receiver':
        argument = 'Argument[self]'
    elif isinstance(dangerous_argument, int):
        argument = f'Argument[{dangerous_argument}]'
    else:
        return None
    return (module, '.'.join([*(f'Member[{member}]' for member in members), argument]))

def semgrep_call_pattern(rule: dict[str, Any]) -> dict[str, Any] | None:
    if rule.get('callee'):
        return {'pattern': f"{rule['callee'].removeprefix('builtins.')}(...)"}
    if rule.get('method'):
        return {'pattern': f"$RECEIVER.{rule['method']}(...)"}
    regex = rule.get('callee_regex') or rule.get('method_regex')
    if regex:
        expression = regex.removeprefix('^').removesuffix('$')
        return {'pattern-regex': f'\\b(?:{expression})\\s*\\('}
    return None

def write_filtered_official_rules(
    official: dict[str, Any], canonical: dict[str, Any], output: Path
) -> dict[str, int]:
    raw_by_reference = {record['catalog_key']: record for record in official['records']}
    candidates = []
    for rule in canonical['rules']:
        if not rule['enabled_for_sink_detection']:
            continue
        reference = rule['official_rule_reference']
        candidates.append({
            'candidate_id': rule['canonical_rule_id'],
            'tool': rule['tool'],
            'sink_kind': rule['sink_kind'],
            'semantic_role': rule['semantic_role'],
            'matcher': rule.get('matcher', {}),
            'dangerous_ports': rule.get('dangerous_ports', []),
            'conditions': rule.get('conditions', {}),
            'official_rule_reference': reference,
            'official_evidence': raw_by_reference[reference],
        })
    payload = {
        'version': 1,
        'source': 'filtered_official_rules',
        'purpose': 'Human review input only; these candidates are not Guard sinks.',
        'official_versions': official['versions'],
        'filter': 'enabled_for_sink_detection from normalized official semantics',
        'candidate_count': len(candidates),
        'candidates': candidates,
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=160),
        encoding='utf-8',
    )
    return {'official_total': len(official['records']), 'filtered_candidates': len(candidates)}


def native_rules(final_rules: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = list(final_rules['rules'])
    coverage = {
        'source': 'final_rules',
        'human_total': len(rules),
        'generated_total': len(rules),
        'official_rules_converted': 0,
    }
    return rules, coverage


def compile_semgrep(rules: list[dict[str, Any]], output: Path) -> None:
    compiled = []
    for rule in rules:
        matcher = semgrep_call_pattern(rule) if rule['template'] in CALL_TEMPLATES else None
        if rule['template'] == 'assignment':
            target = rule.get('target_regex', '')
            matcher = {'pattern-regex': f"\\b(?:{target.removeprefix('^').removesuffix('$')})(?:\\[[^\\n]+\\])?\\s*="}
        if not matcher:
            continue
        compiled.append({'id': rule['rule_id'].lower(), 'message': f"{rule['sink_kind']} sink: {rule['rule_id']}", 'languages': ['python'], 'severity': 'WARNING', 'metadata': {'category': 'security', 'sink_kind': rule['sink_kind'], 'runtime_healing_rule_id': rule['rule_id'], 'evidence': rule['evidence']}, **matcher})
    (output / 'semgrep.yaml').write_text(yaml.safe_dump({'rules': compiled}, sort_keys=False), encoding='utf-8')

def compile_codeql(rules: list[dict[str, Any]], output: Path) -> None:
    rows = []
    metadata = []
    for rule in rules:
        if not rule.get('callee') or rule['sink_kind'] == 'UNKNOWN_EFFECT':
            continue
        model = module_and_path(rule['callee'], rule.get('dangerous_argument'))
        if not model:
            continue
        module, path = model
        rows.append([module, path, rule['sink_kind']])
        metadata.append({'module': module, 'path': path, 'sink_kind': rule['sink_kind'], 'rule_id': rule['rule_id']})
    model_dir = output / 'codeql' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / 'codeql' / 'codeql-pack.yml').write_text("name: agenticheal/runtime-healing-models\nversion: 0.0.1\nlibrary: true\nextensionTargets:\n  codeql/python-all: '*'\ndataExtensions:\n  - models/**/*.yml\n", encoding='utf-8')
    extension = {'extensions': [{'addsTo': {'pack': 'codeql/python-all', 'extensible': 'sinkModel'}, 'data': rows}]}
    (model_dir / 'runtime-healing.model.yml').write_text(yaml.safe_dump(extension, sort_keys=False), encoding='utf-8')
    (output / 'codeql-model-map.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

def compile_pysa(data: dict[str, Any], output: Path) -> None:
    sinks = [{'name': kind, 'comment': f'Runtime Healing {kind} sink'} for kind in data['sink_kinds'] if kind != 'UNKNOWN_EFFECT']
    sources = [{'name': 'RuntimeHealingInput', 'comment': 'Values observed or modified by handling code'}]
    taint_rules = [{'name': f'Runtime Healing {kind}', 'code': 7000 + index, 'sources': ['RuntimeHealingInput'], 'sinks': [kind], 'message_format': f'Handling input reaches {kind}'} for index, kind in enumerate(data['sink_kinds'][:-1])]
    pysa_dir = output / 'pysa'
    pysa_dir.mkdir(parents=True, exist_ok=True)
    (pysa_dir / 'taint.config').write_text(json.dumps({'sources': sources, 'sinks': sinks, 'features': [], 'rules': taint_rules}, indent=2) + '\n', encoding='utf-8')
    models = []
    for rule in data['rules']:
        if not rule.get('callee') or rule['sink_kind'] == 'UNKNOWN_EFFECT':
            continue
        argument = rule.get('dangerous_argument')
        if argument == 'receiver':
            signature = f"self: TaintSink[{rule['sink_kind']}], *args, **kwargs"
        elif isinstance(argument, int):
            parameters = [f'arg{index}' for index in range(argument)] + [f"arg{argument}: TaintSink[{rule['sink_kind']}]", '*args', '**kwargs']
            signature = ', '.join(parameters)
        elif isinstance(argument, list) and argument:
            last = max(argument)
            parameters = [
                f"arg{index}: TaintSink[{rule['sink_kind']}]" if index in argument else f'arg{index}'
                for index in range(last + 1)
            ] + ['*args', '**kwargs']
            signature = ', '.join(parameters)
        else:
            continue
        models.append(f"def {rule['callee'].removeprefix('builtins.')}({signature}): ...")
    (pysa_dir / 'runtime_healing.pysa').write_text('\n'.join(models) + '\n', encoding='utf-8')

def compile_bandit(rules: list[dict[str, Any]], output: Path) -> None:
    package = output / 'bandit_plugin'
    module = package / 'runtime_healing_bandit'
    module.mkdir(parents=True, exist_ok=True)
    (module / 'rules.json').write_text(json.dumps(rules, indent=2) + '\n', encoding='utf-8')
    plugin = 'import ast\nimport json\nimport re\nfrom pathlib import Path\n\nimport bandit\nfrom bandit.core import test_properties as test\n\nRULES = json.loads(Path(__file__).with_name("rules.json").read_text())\n\ndef call_matches(rule, callee):\n    if rule.get("callee"):\n        target = rule["callee"].removeprefix("builtins.")\n        return callee in {target, rule["callee"]}\n    method_name = callee.rsplit(".", 1)[-1]\n    method_pattern = rule.get("method_regex")\n    if rule.get("method") == method_name or (method_pattern and re.fullmatch(method_pattern, method_name)):\n        receiver = rule.get("receiver")\n        return receiver not in {"socket", "sys.path"} or receiver in callee\n    pattern = rule.get("callee_regex")\n    return bool(pattern and re.fullmatch(pattern, callee))\n\ndef dotted_name(node):\n    if isinstance(node, ast.Name):\n        return node.id\n    if isinstance(node, ast.Attribute):\n        prefix = dotted_name(node.value)\n        return f"{prefix}.{node.attr}" if prefix else node.attr\n    if isinstance(node, ast.Subscript):\n        return dotted_name(node.value)\n    return ""\n\ndef issue(rule):\n    return bandit.Issue(severity=bandit.MEDIUM, confidence=bandit.HIGH, text=f"{rule[\'sink_kind\']}|{rule[\'rule_id\']}|{rule[\'evidence\']}")\n\n@test.checks("Call")\n@test.test_id("RH001")\ndef runtime_healing_call(context):\n    callee = context.call_function_name_qual or context.call_function_name or ""\n    rule = next((item for item in RULES if item["template"] != "assignment" and call_matches(item, callee)), None)\n    return issue(rule) if rule else None\n\n@test.checks("Assign", "AnnAssign", "AugAssign")\n@test.test_id("RH002")\ndef runtime_healing_assignment(context):\n    node = context.node\n    targets = node.targets if isinstance(node, ast.Assign) else [node.target]\n    rule = next((item for item in RULES if item["template"] == "assignment" and any(re.match(item["target_regex"], dotted_name(target)) for target in targets)), None)\n    return issue(rule) if rule else None\n'
    (module / '__init__.py').write_text(plugin, encoding='utf-8')
    (package / 'pyproject.toml').write_text('[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "runtime-healing-bandit"\nversion = "0.0.1"\ndependencies = []\n\n[project.entry-points."bandit.plugins"]\nruntime_healing_call = "runtime_healing_bandit:runtime_healing_call"\nruntime_healing_assignment = "runtime_healing_bandit:runtime_healing_assignment"\n\n[tool.setuptools.package-data]\nruntime_healing_bandit = ["rules.json"]\n', encoding='utf-8')
    (package / 'setup.py').write_text('from setuptools import setup\nsetup()\n', encoding='utf-8')

def artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            records.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
    return records


def validate_outputs(catalog_dir: Path, native_dir: Path) -> None:
    catalog_paths = sorted(catalog_dir.glob("*.json"))
    if len(catalog_paths) != 5:
        raise RuntimeError(f"Expected five rule catalogs, found {len(catalog_paths)}")
    for path in catalog_paths:
        json.loads(path.read_text(encoding="utf-8"))
    yaml.safe_load((native_dir / "semgrep.yaml").read_text(encoding="utf-8"))


def build_sink_rules(rules_path: Path, sources_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = copy_sources(sources_dir)
    data = load_rules(rules_path)
    clear_output(output_dir)
    catalog_dir = output_dir / "catalogs"
    internal_dir = output_dir / "internal"
    native_dir = output_dir / "native"
    catalog_dir.mkdir(parents=True)
    internal_dir.mkdir(parents=True)
    native_dir.mkdir(parents=True)

    official = import_official_rules()
    canonical = normalize_official_rules(official)
    (internal_dir / "official_catalog.json").write_text(
        json.dumps(official, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (internal_dir / "normalized_catalog.json").write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    official_filter = write_filtered_official_rules(official, canonical, OFFICIAL_RULES)
    catalogs = build_rule_catalogs(data, catalog_dir)
    generated_rules, conversion_coverage = native_rules(data)
    compile_bandit(generated_rules, native_dir)
    compile_semgrep(generated_rules, native_dir)
    compile_codeql(generated_rules, native_dir)
    compile_pysa({**data, "rules": generated_rules}, native_dir)
    (internal_dir / "shared_conversion_coverage.json").write_text(
        json.dumps(conversion_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_outputs(catalog_dir, native_dir)

    manifest = {
        "source_files": {"total": source_manifest["total"], "counts": source_manifest["counts"]},
        "sink_kinds": len(data["sink_kinds"]),
        "official_rules": len(official["records"]),
        "normalized_rules": len(canonical["rules"]),
        "official_filter": official_filter,
        "rule_catalogs": catalogs,
        "shared_conversion": conversion_coverage,
        "artifacts": artifact_records(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Runtime Healing sink rules")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--sources", type=Path, default=SOURCES_ROOT)
    parser.add_argument("--output", type=Path, default=GENERATED_ROOT)
    args = parser.parse_args()
    manifest = build_sink_rules(args.rules.resolve(), args.sources.resolve(), args.output.resolve())
    summary = {
        "sources": manifest["source_files"],
        "official_rules": manifest["official_rules"],
        "normalized_rules": manifest["normalized_rules"],
        "artifacts": len(manifest["artifacts"]),
        "output": args.output.resolve().as_posix(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
