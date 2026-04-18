import argparse
import ast
import io
import json
import os
import re
import tokenize

import yaml

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

DATASET_ROOT_CANDIDATES = {
    "flashinfer-trace": [
        os.path.join(REPO_TOP_PATH, "datasets", "flashinfer-trace"),
        os.path.join(REPO_TOP_PATH, "flashinfer-trace"),
    ],
    "mlsys26-contest": [
        os.path.join(REPO_TOP_PATH, "datasets", "mlsys26-contest"),
        os.path.join(REPO_TOP_PATH, "mlsys26-contest"),
    ],
}

CUTE_DSL_SOURCE_SCAFFOLDING = """## Compile-Safe CuTe DSL Scaffold

Use this source structure for every generated solution so exploration starts
from a known-valid CuTe import, launch, stream, and cache pattern. The code
block below is valid Python; adapt the same patterns to the real kernel instead
of inventing import paths or launch syntax:

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

_COMPILED = {}


def _tensor_cache_part(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        tensor.device.type,
        tensor.device.index,
    )


def _cache_key(*tensors, **params):
    return tuple(_tensor_cache_part(t) for t in tensors) + tuple(sorted(params.items()))


@cute.kernel
def compile_probe_kernel():
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    bdim = cute.arch.block_dim()[0]
    cute.arch.sync_threads()


@cute.jit
def launch_compile_probe(stream):
    compile_probe_kernel().launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        stream=stream,
    )


def _get_compiled(key, jit_fn, *compile_args):
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(jit_fn, *compile_args)
        _COMPILED[key] = compiled
    return compiled


def _compile_probe_once():
    stream = cutlass.cuda.default_stream()
    compiled = _get_compiled(("compile_probe",), launch_compile_probe, stream)
    compiled(stream)
    cutlass.cuda.stream_sync(stream)
```

For the real implementation, define task-specific `@cute.kernel` and `@cute.jit`
functions at module scope, convert tensors with `from_dlpack` only when needed,
launch with `real_kernel(args...).launch(grid=..., block=..., stream=stream)`,
compile through `cute.compile`, cache by `_cache_key`, and return outputs from a
plain Python `run` wrapper. The probe above is only a minimal API shape example;
do not call a separate probe from the final `run` path because benchmark warmup
and timed iterations should compile and execute the task kernel itself.
"""

CUTE_SOURCE_IMPORT_REPLACEMENTS = {
    "from cutlass.cute import from_dlpack": (
        "from cutlass.cute.runtime import from_dlpack"
    ),
    "from cutlass import from_dlpack": "from cutlass.cute.runtime import from_dlpack",
    "from cutlass.cuda import default_stream": "",
    "from cutlass.cute.cuda import default_stream": "",
}

CUTE_SOURCE_API_REPLACEMENTS = {
    "stream.synchronize()": "cutlass.cuda.stream_sync(stream)",
    "cute.threadIdx.x": "cute.arch.thread_idx()[0]",
    "cute.threadIdx.y": "cute.arch.thread_idx()[1]",
    "cute.threadIdx.z": "cute.arch.thread_idx()[2]",
    "cute.blockIdx.x": "cute.arch.block_idx()[0]",
    "cute.blockIdx.y": "cute.arch.block_idx()[1]",
    "cute.blockIdx.z": "cute.arch.block_idx()[2]",
    "cute.blockDim.x": "cute.arch.block_dim()[0]",
    "cute.blockDim.y": "cute.arch.block_dim()[1]",
    "cute.blockDim.z": "cute.arch.block_dim()[2]",
    "cute.gridDim.x": "cute.arch.grid_dim()[0]",
    "cute.gridDim.y": "cute.arch.grid_dim()[1]",
    "cute.gridDim.z": "cute.arch.grid_dim()[2]",
    "cute.syncthreads()": "cute.arch.sync_threads()",
}

CUTE_SOURCE_FORBIDDEN_PATTERNS = {
    "cute.load(": (
        "cute.load is not available in the installed CuTe DSL; convert inputs "
        "with from_dlpack or direct JIT arguments and use CuTe tensor indexing."
    ),
    "cute.store(": (
        "cute.store is not available in the installed CuTe DSL; write through "
        "CuTe tensor indexing or supported copy/tensor APIs."
    ),
    "cute.shared_memory": (
        "cute.shared_memory is not available as a direct helper in this CuTe "
        "DSL environment; use documented CuTe tensor/layout or nvgpu memory "
        "APIs only when constructing shared-memory tiles correctly."
    ),
    "cute.arch.shared_memory": (
        "cute.arch.shared_memory is not available in this CuTe DSL environment; "
        "avoid direct shared-memory helpers unless using documented nvgpu/CuTe "
        "memory APIs correctly."
    ),
    "cute.StaticBuffer": (
        "cute.StaticBuffer does not exist in this CuTe DSL version; "
        "use standard CuTe tensor/layout APIs for shared memory tiles."
    ),
    "cute.alloc_shared": (
        "cute.alloc_shared does not exist in this CuTe DSL version; "
        "use documented CuTe tensor/layout or nvgpu memory APIs."
    ),
    "cute.arch.ld_global": (
        "cute.arch.ld_global* helpers do not exist in this CuTe DSL version; "
        "use CuTe tensor indexing for global memory access."
    ),
    "cute.arch.st_global": (
        "cute.arch.st_global* helpers do not exist in this CuTe DSL version; "
        "use CuTe tensor indexing for global memory stores."
    ),
    "cute.range_dynamic": (
        "cute.range_dynamic does not exist; use Python range() for static "
        "loops or CuTe DSL loop constructs."
    ),
    "cute.tensor_view": (
        "cute.tensor_view does not exist; use from_dlpack to convert "
        "torch tensors to CuTe tensors."
    ),
    "cute.maximum": (
        "cute.maximum does not exist in CuTe DSL; use Python max() for "
        "host code or implement element-wise max in kernel logic."
    ),
    "cute.make_shape": (
        "cute.make_shape does not exist; use Python tuples for grid/block "
        "dimensions and CuTe layout APIs for tensor shapes."
    ),
    "cute.make_layout": (
        "cute.make_layout does not exist; use CuTe layout APIs from the "
        "documented cutlass.cute namespace."
    ),
}

CUTE_SOURCE_FORBIDDEN_REGEXES = {
    r"\b\w+\s*\[[^\]\n]*grid[^\]\n]*block[^\]\n]*stream[^\]\n]*\]\s*\(": (
        "Triton/CUDA-style bracket kernel launch is not valid CuTe DSL; call "
        "kernel(args...).launch(grid=..., block=..., smem=..., stream=...)."
    ),
}


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        code = code_match.group(1).strip()
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()
        return code

    return output_string


def _strip_cute_type_annotations(source: str) -> str:
    """Remove Python type annotations from @cute.kernel and @cute.jit function params.

    CuTe DSL passes MLIR-wrapped types (e.g. Int32) at compile time.  If the
    function signature carries a Python annotation like ``num_tokens: int``,
    the DSL runtime raises ``DSLRuntimeError: expects argument … to be
    <class 'int'>, but got Int32``.  Stripping annotations avoids this.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Collect line ranges of parameters to fix (1-indexed)
    lines = source.splitlines(keepends=True)

    def _is_cute_decorated(node: ast.FunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                if dec.value.id == "cute" and dec.attr in ("kernel", "jit"):
                    return True
        return False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_cute_decorated(node):
            continue
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation is not None:
                # Remove `: <type>` from the parameter line
                lineno = arg.lineno - 1  # 0-indexed
                line = lines[lineno]
                # Replace `param_name: type_name` with `param_name`
                lines[lineno] = re.sub(
                    rf"\b{re.escape(arg.arg)}\s*:\s*\w+", arg.arg, line
                )
        # Also strip return annotation (may be on a different line than def)
        if node.returns is not None:
            ret_lineno = node.returns.lineno - 1
            lines[ret_lineno] = re.sub(r"\)\s*->.*?:", "):", lines[ret_lineno])

    return "".join(lines)


def normalize_cute_source(source: str) -> str:
    """Normalize common CuTe DSL import/stream mistakes in generated code."""
    # Strip any XML edit/reasoning tags that leaked from LLM output.
    # Remove reasoning blocks entirely (content is never valid code).
    normalized = re.sub(
        r"<reasoning_\d+>.*?</reasoning_\d+>\s*", "", source, flags=re.DOTALL
    )
    # Remove old_str/new_str tag lines (keep content between them, as it's code).
    normalized = re.sub(
        r"</?(?:old_str|new_str)_\d+>[ \t]*\n?", "", normalized
    )
    # Strip type annotations from @cute.kernel / @cute.jit functions to avoid
    # DSL Int32-vs-int mismatches at MLIR compile time.
    normalized = _strip_cute_type_annotations(normalized)
    for old, new in CUTE_SOURCE_IMPORT_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    for old, new in CUTE_SOURCE_API_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(
        r"(?<![\w.])default_stream\(\)",
        "cutlass.cuda.default_stream()",
        normalized,
    )

    lines = []
    seen_runtime_dlpack_import = False
    for line in normalized.splitlines():
        if line == "from cutlass.cute.runtime import from_dlpack":
            if seen_runtime_dlpack_import:
                continue
            seen_runtime_dlpack_import = True
        lines.append(line)
    normalized = "\n".join(lines)

    has_cutlass_import = re.search(r"^import\s+cutlass(\s|$)", normalized, re.MULTILINE)
    if "cutlass.cuda." in normalized and not has_cutlass_import:
        normalized = "import cutlass\n" + normalized

    return normalized


def _source_without_strings_and_comments(source: str) -> str:
    """Return source text suitable for simple static scans."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return "".join(
            " " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string
            for token in tokens
        )
    except tokenize.TokenError:
        return source


def _line_number(source: str, index: int) -> int:
    return source.count("\n", 0, max(index, 0)) + 1


def _is_cute_kernel_decorator(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "kernel"
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "cute"
    )


def _find_cute_kernel_ast_issues(source: str) -> list[str]:
    """Find unsupported AST patterns inside @cute.kernel functions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_cute_kernel_decorator(dec) for dec in node.decorator_list):
            continue

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                issues.append(
                    f"line {child.lineno}: return is not allowed inside @cute.kernel; "
                    "guard work with predicates instead of early exits."
                )
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "int"
            ):
                issues.append(
                    f"line {child.lineno}: int(...) inside @cute.kernel often "
                    "tries to convert dynamic DSL values to Python integers; "
                    "keep values in CuTe/DSL scalar form."
                )

    return issues


def find_cute_source_issues(source: str) -> list[str]:
    """Find known CuTe DSL API mistakes before spending a GPU compile."""
    scan_source = _source_without_strings_and_comments(source)
    issues = []

    for pattern, message in CUTE_SOURCE_FORBIDDEN_PATTERNS.items():
        index = scan_source.find(pattern)
        if index != -1:
            issues.append(f"line {_line_number(scan_source, index)}: {message}")

    for pattern, message in CUTE_SOURCE_FORBIDDEN_REGEXES.items():
        match = re.search(pattern, scan_source)
        if match:
            issues.append(f"line {_line_number(scan_source, match.start())}: {message}")

    issues.extend(_find_cute_kernel_ast_issues(source))
    return issues


def prepare_cute_source_for_eval(source: str) -> tuple[str, list[str]]:
    """Normalize generated CuTe source and return static preflight issues."""
    normalized = normalize_cute_source(source)
    return normalized, find_cute_source_issues(normalized)


def str_replace(
    file_content: str,
    old_str: str,
    new_str: str | None,
    encoding: str = "utf-8",
) -> str:
    """
    Implement the str_replace command, which replaces old_str with new_str in the file content.

    Args:
        file_content: The original file content
        old_str: String to replace
        new_str: Replacement string
        encoding: The encoding to use (auto-detected by decorator)
    """
    new_str = new_str or ""

    pattern = re.escape(old_str)
    occurrences = [
        (
            file_content.count("\n", 0, match.start()) + 1,
            match.group(),
            match.start(),
        )
        for match in re.finditer(pattern, file_content)
    ]

    if not occurrences:
        old_str = old_str.strip()
        new_str = new_str.strip()
        pattern = re.escape(old_str)
        occurrences = [
            (
                file_content.count("\n", 0, match.start()) + 1,
                match.group(),
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]
        if not occurrences:
            print(
                f"[Warning] No replacement was performed, old_str\n ```\n{old_str}\n```\ndid not appear verbatim in the file."
            )
            return file_content
    if len(occurrences) > 1:
        line_numbers = sorted(set(line for line, _, _ in occurrences))
        print(
            f"[Warning] No replacement was performed. Multiple occurrences of old_str\n ```\n{old_str}\n```\nin lines {line_numbers}. Please ensure it is unique."
        )
        return file_content

    replacement_line, matched_text, idx = occurrences[0]
    new_file_content = (
        file_content[:idx] + new_str + file_content[idx + len(matched_text) :]
    )
    return new_file_content


def _strip_tag_content(text: str) -> str:
    """Strip exactly one leading and one trailing newline from tag content."""
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def extract_edits(output: str):
    """Extract str_replace edits from LLM output."""
    edits = []
    seen_ids = set()
    for line in output.split("\n"):
        if line.strip().startswith("<old_str_"):
            try:
                idx = int(line.strip().split("_")[2].split(">")[0])
                if idx in seen_ids:
                    continue
                seen_ids.add(idx)
                raw_old_str = output.split("<old_str_" + str(idx) + ">")[1].split(
                    "</old_str_" + str(idx) + ">"
                )[0]
                raw_new_str = output.split("<new_str_" + str(idx) + ">")[1].split(
                    "</new_str_" + str(idx) + ">"
                )[0]
                edits.append(
                    (_strip_tag_content(raw_old_str), _strip_tag_content(raw_new_str))
                )
            except Exception:
                continue
    return edits


# Dataset paths


def get_dataset_root(test_source: str) -> str:
    """Return dataset root path for the given test source."""
    if test_source not in DATASET_ROOT_CANDIDATES:
        raise ValueError(
            f"Unknown test_source: {test_source}, expected one of {list(DATASET_ROOT_CANDIDATES)}"
        )

    for candidate in DATASET_ROOT_CANDIDATES[test_source]:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Dataset root for '{test_source}' not found. Checked: "
        + ", ".join(DATASET_ROOT_CANDIDATES[test_source])
    )


def construct_flashinfer_trace_dataset(
    op_type: str,
    dataset_root: str = None,
) -> list[str]:
    """Return sorted list of problem names for given op_type."""
    if dataset_root is None:
        dataset_root = get_dataset_root("flashinfer-trace")
    op_dir = os.path.join(dataset_root, "definitions", op_type)
    return sorted(f[:-5] for f in os.listdir(op_dir) if f.endswith(".json"))


def load_flashinfer_trace_definition(
    op_type: str,
    problem_name: str,
    dataset_root: str = None,
) -> dict:
    """Load definition JSON for given op_type and problem_name."""
    if dataset_root is None:
        dataset_root = get_dataset_root("flashinfer-trace")
    path = os.path.join(dataset_root, "definitions", op_type, f"{problem_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def load_test_source(test_source: str, level, problem_id):
    """
    Load reference architecture source code.
    level=op_type (str), problem_id=problem_name (str)
    """
    dataset_root = get_dataset_root(test_source)
    definition = load_flashinfer_trace_definition(
        str(level), str(problem_id), dataset_root=dataset_root
    )
    return str(problem_id), definition


def load_tasks_from_test_list(
    tasks_path: str,
    test_source: str = "mlsys26-contest",
) -> list[dict]:
    """
    Load tasks from a test list file.

    Args:
        tasks_path: Path to the tasks file
        test_source: Dataset source identifier

    Returns:
        List of task dicts with 'level' and 'problem_id' keys
    """
    dataset_root = get_dataset_root(test_source)

    with open(tasks_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    tasks = []
    for line in lines:
        parts = line.split(" ", 1)
        level = parts[0]
        if len(parts) == 1:
            problems = construct_flashinfer_trace_dataset(
                level, dataset_root=dataset_root
            )
        else:
            problems = [p.strip() for p in parts[1].split(",") if p.strip()]
        tasks.extend({"level": level, "problem_id": str(p)} for p in problems)

    return tasks


def load_config_from_yaml(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Load configuration from YAML file and update args.
    Command line arguments take precedence over YAML values.
    """
    if args.config is None:
        return args

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Set YAML values as new defaults, then re-parse so CLI args override them
    parser.set_defaults(**{k: v for k, v in config_dict.items() if hasattr(args, k)})
    return parser.parse_args(argv)
