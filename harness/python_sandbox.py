"""Restricted in-memory Python execution sandbox for computing and executing trajectories."""

from __future__ import annotations

import ast
import io
import math
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.base import WORKSPACE_SPEC

FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pathlib",
        "urllib",
        "requests",
        "httpx",
        "mujoco",
        "PIL",
        "cv2",
        "importlib",
        "builtins",
        "ctypes",
        "multiprocessing",
        "threading",
        "pickle",
    }
)

FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__globals__",
        "__code__",
        "__closure__",
        "__loader__",
        "__spec__",
    }
)


@dataclass
class SandboxExecutionResult:
    """Result returned by RestrictedPythonSandbox.run()."""

    success: bool
    stdout: str
    error_message: str | None
    strokes_executed: int
    waypoints_executed: int
    ink_pixels_added: int
    spatial_hallucinations: int
    code_hallucinations: int
    warnings: list[str]


def _validate_ast_safety(code: str) -> None:
    """Parse code AST and reject forbidden imports or dunder introspection."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in FORBIDDEN_MODULES:
                    raise PermissionError(
                        f"Import of module '{root_mod}' is forbidden in the robot math sandbox. "
                        "Use pre-loaded 'np' (numpy), 'math', 'draw_primitives', and 'draw_waypoints'."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in FORBIDDEN_MODULES:
                    raise PermissionError(
                        f"Import from module '{root_mod}' is forbidden in the robot math sandbox."
                    )
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                raise PermissionError(f"Access to attribute '{node.attr}' is forbidden.")
        elif isinstance(node, ast.Name) and node.id in ("open", "eval", "exec", "compile"):
            raise PermissionError(f"Calling builtin '{node.id}' is forbidden in the sandbox.")


def _safe_import(name: str, globals_dict: Any = None, locals_dict: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """Allow importing only math, numpy, or typing inside the sandbox."""
    del globals_dict, locals_dict, level
    root = name.split(".")[0]
    if root not in ("math", "numpy", "typing", "dataclasses"):
        raise PermissionError(
            f"Import of '{name}' is not permitted. Only 'math' and 'numpy' ('np') are allowed."
        )
    return __import__(name, fromlist=fromlist)


class RestrictedPythonSandbox:
    """In-memory Python execution sandbox with numpy, math, and robot drawing callbacks."""

    def __init__(
        self,
        draw_primitives_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
        draw_waypoints_fn: Callable[[list[list[float]] | np.ndarray], dict[str, Any]],
    ) -> None:
        self._draw_primitives_fn = draw_primitives_fn
        self._draw_waypoints_fn = draw_waypoints_fn

    def run(self, code: str) -> SandboxExecutionResult:
        """Validate and execute user/LLM Python code and return telemetry."""
        stdout_buf = io.StringIO()
        strokes_executed = 0
        waypoints_executed = 0
        ink_pixels_added = 0
        spatial_hallucinations = 0
        warnings: list[str] = []

        def _wrapped_draw_primitives(strokes: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal strokes_executed, waypoints_executed, ink_pixels_added, spatial_hallucinations
            res = self._draw_primitives_fn(strokes)
            strokes_executed += len(strokes)
            waypoints_executed += int(res.get("waypoints_commanded", 0))
            ink_pixels_added += int(res.get("ink_pixels_added", 0))
            spatial_hallucinations += int(res.get("spatial_hallucinations", 0))
            warnings.extend(res.get("warnings", []))
            return res

        def _wrapped_draw_waypoints(waypoints_xyz: list[list[float]] | np.ndarray) -> dict[str, Any]:
            nonlocal waypoints_executed, ink_pixels_added, spatial_hallucinations
            res = self._draw_waypoints_fn(waypoints_xyz)
            waypoints_executed += int(res.get("waypoints_commanded", 0))
            ink_pixels_added += int(res.get("ink_pixels_added", 0))
            spatial_hallucinations += int(res.get("spatial_hallucinations", 0))
            warnings.extend(res.get("warnings", []))
            return res

        safe_builtins: dict[str, Any] = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "__import__": _safe_import,
        }

        sandbox_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "np": np,
            "numpy": np,
            "math": math,
            "WORKSPACE_SPEC": dict(WORKSPACE_SPEC),
            "draw_primitives": _wrapped_draw_primitives,
            "draw_waypoints": _wrapped_draw_waypoints,
        }

        try:
            _validate_ast_safety(code)
            with redirect_stdout(stdout_buf):
                exec(compile(code, "<pelican_sandbox>", "exec"), sandbox_globals)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            tb_short = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return SandboxExecutionResult(
                success=False,
                stdout=stdout_buf.getvalue()[:2000],
                error_message=tb_short,
                strokes_executed=strokes_executed,
                waypoints_executed=waypoints_executed,
                ink_pixels_added=ink_pixels_added,
                spatial_hallucinations=spatial_hallucinations,
                code_hallucinations=1,
                warnings=warnings,
            )

        return SandboxExecutionResult(
            success=True,
            stdout=stdout_buf.getvalue()[:2000],
            error_message=None,
            strokes_executed=strokes_executed,
            waypoints_executed=waypoints_executed,
            ink_pixels_added=ink_pixels_added,
            spatial_hallucinations=spatial_hallucinations,
            code_hallucinations=0,
            warnings=warnings,
        )
