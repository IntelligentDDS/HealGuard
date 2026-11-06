import sys


REPOSITORY_EXCLUDED_DIRECTORIES = {
    ".git", ".eggs", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "__pycache__", "__pypackages__", "_examples", "_vendor", "asv_bench",
    "bench", "benchmark", "benchmarks", "build", "dependencies", "depends", "deps", "dist",
    "doc", "docs", "env", "example", "examples", "extern", "external", "node_modules",
    "site-packages", "test", "testing", "tests", "Tests", "third_party", "thirdparty",
    "tutorial", "tutorials", "vendor", "venv",
}

REPOSITORY_EXCLUDED_PATTERNS = {
    "*.egg-info",
    "*.pyx",
    "*.pyx.in",
    "*.pxd",
    "*.pxi",
}

REPOSITORY_EXCLUDED_SUFFIXES = (".pyx", ".pyx.in", ".pxd", ".pxi")

HEAL_EXCLUDED_DIRECTORIES = {
    *REPOSITORY_EXCLUDED_DIRECTORIES,
}


def is_repository_path_excluded(parts, is_directory: bool = False) -> bool:
    parts = tuple(parts)
    directory_parts = parts if is_directory else parts[:-1]
    return any(
        part in REPOSITORY_EXCLUDED_DIRECTORIES
        or part.endswith(".egg-info")
        or part.lower().startswith(("vendor-", "vendored-", "vendor_", "vendored_"))
        for part in directory_parts
    ) or bool(not is_directory and parts and parts[-1].lower().endswith(REPOSITORY_EXCLUDED_SUFFIXES))


def is_python_source(path) -> bool:
    if path.suffix == ".py":
        return True
    try:
        with path.open("rb") as source_file:
            first_line = source_file.readline(256)
    except OSError as error:
        print(f"Failed to inspect possible Python source {path}: {error}", file=sys.stderr)
        return False
    return first_line.startswith(b"#!") and b"python" in first_line.lower()


def iter_repository_python_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if is_repository_path_excluded(relative.parts) or path.is_symlink():
            continue
        try:
            if not path.is_file():
                continue
        except OSError as error:
            print(f"Failed to inspect repository path {path}: {error}", file=sys.stderr)
            continue
        if is_python_source(path):
            yield path
