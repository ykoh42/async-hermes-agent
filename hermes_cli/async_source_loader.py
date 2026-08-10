"""Async source-module loading for plugin packages.

Python's normal source loader reads ``.py`` files synchronously while an
import statement is executing.  Plugin discovery runs inside the agent event
loop, so that implicit read is not an acceptable boundary.  This module reads
the package sources asynchronously first, then lets the normal import
protocol execute compiled source from memory.  Relative imports therefore
keep their normal lazy execution and package metadata without performing a
second synchronous filesystem read.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import aiofiles
import aiofiles.os


@dataclass(frozen=True)
class _SourceRecord:
    source: str
    filename: str
    is_package: bool
    search_path: str | None = None
    is_namespace: bool = False


class _MemorySourceLoader(importlib.abc.Loader):
    """Execute one source module whose bytes were read before import."""

    def __init__(self, fullname: str, record: _SourceRecord) -> None:
        self._fullname = fullname
        self._record = record

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return None

    def exec_module(self, module: ModuleType) -> None:
        code = compile(
            self._record.source,
            self._record.filename,
            "exec",
        )
        exec(code, module.__dict__)


class _MemorySourceFinder(importlib.abc.MetaPathFinder):
    """Resolve only the source modules belonging to one loaded package."""

    def __init__(self, records: dict[str, _SourceRecord]) -> None:
        self._records = records

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        record = self._records.get(fullname)
        if record is None:
            return None
        if record.is_namespace:
            spec = importlib.machinery.ModuleSpec(
                fullname,
                loader=None,
                is_package=True,
            )
            spec.submodule_search_locations = [record.search_path or record.filename]
            return spec
        loader = _MemorySourceLoader(fullname, record)
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=record.filename,
            is_package=record.is_package,
        )
        if (
            spec is not None
            and spec.submodule_search_locations is not None
            and record.search_path is not None
        ):
            spec.submodule_search_locations[:] = [record.search_path]
        return spec


async def _read_source(path: Path) -> str:
    async with aiofiles.open(path, mode="rb") as handle:
        raw = await handle.read()
    return importlib.util.decode_source(raw)


async def locate_source_module(
    module_name: str,
    *,
    distribution=None,  # type: ignore[no-untyped-def]
) -> tuple[Path, bool] | None:
    """Find a source module using async stat calls only.

    ``importlib.util.find_spec`` performs synchronous filesystem traversal and
    may execute parent package imports.  Entry-point discovery only needs the
    source path, so inspect the owning distribution (when available) and then
    ``sys.path`` directly.  The boolean in the result indicates a package
    ``__init__.py`` rather than a single module file.
    """
    existing = sys.modules.get(module_name)
    existing_file = getattr(existing, "__file__", None)
    if existing_file:
        path = Path(existing_file)
        if path.suffix == ".py" and await aiofiles.os.path.isfile(path):
            return path, path.name == "__init__.py"

    relative = Path(*module_name.split("."))
    roots: list[Path] = []
    if distribution is not None:
        try:
            located = distribution.locate_file("")
            roots.append(Path(located))
        except (AttributeError, TypeError, ValueError, OSError):
            pass
    roots.extend(Path(entry or ".") for entry in sys.path)

    seen: set[str] = set()
    for root in roots:
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        module_file = root / f"{relative}.py"
        if await aiofiles.os.path.isfile(module_file):
            return module_file, False
        package_file = root / relative / "__init__.py"
        if await aiofiles.os.path.isfile(package_file):
            return package_file, True
    return None


async def _package_source_records(
    module_name: str,
    init_file: Path,
) -> dict[str, _SourceRecord]:
    """Read a package's complete Python source tree before execution.

    The normal import machinery is still used to execute the already-read
    source, but every package and module that can be reached through a
    relative import is registered with the in-memory finder first.  Limiting
    this to one directory would let ``from .subpkg import helper`` fall back
    to ``SourceFileLoader`` and synchronously read the subpackage later in a
    turn.
    """
    records: dict[str, _SourceRecord] = {}

    async def collect(package_name: str, package_init: Path) -> None:
        records[package_name] = _SourceRecord(
            source=await _read_source(package_init),
            filename=str(package_init),
            is_package=True,
            search_path=str(package_init.parent),
        )
        package_dir = package_init.parent
        for child_name in sorted(await aiofiles.os.listdir(package_dir)):
            if child_name.startswith((".", "__pycache__")):
                continue
            child = package_dir / child_name
            if child_name == "__init__.py":
                continue
            if child_name.endswith(".py"):
                if not await aiofiles.os.path.isfile(child):
                    continue
                child_module = f"{package_name}.{child.stem}"
                records[child_module] = _SourceRecord(
                    source=await _read_source(child),
                    filename=str(child),
                    is_package=False,
                )
                continue
            if not await aiofiles.os.path.isdir(child):
                continue
            nested_init = child / "__init__.py"
            if not await aiofiles.os.path.isfile(nested_init):
                continue
            await collect(f"{package_name}.{child_name}", nested_init)

    await collect(module_name, init_file)
    return records


async def _package_alias_records(
    alias_name: str,
    init_file: Path,
) -> dict[str, _SourceRecord]:
    """Read one package tree under its normal import name.

    Bundled plugin entry modules are executed in the isolated
    ``hermes_plugins`` namespace, but some upstream plugins intentionally use
    absolute imports from their canonical ``plugins.<category>`` package.
    Register those source records with the same in-memory finder, including
    their package parents, so the absolute import preserves its original
    module identity without returning to ``SourceFileLoader``.
    """
    records = await _package_source_records(alias_name, init_file)
    alias_parts = alias_name.split(".")
    package_dir = init_file.parent
    for depth in range(1, len(alias_parts)):
        parent_name = ".".join(alias_parts[:-depth])
        parent_dir = package_dir.parents[depth - 1]
        parent_init = parent_dir / "__init__.py"
        has_init = await aiofiles.os.path.isfile(parent_init)
        if has_init:
            source = await _read_source(parent_init)
            filename = str(parent_init)
        elif await aiofiles.os.path.isdir(parent_dir):
            source = ""
            filename = str(parent_dir)
        else:
            continue
        records[parent_name] = _SourceRecord(
            source=source,
            filename=filename,
            is_package=True,
            search_path=str(parent_dir),
            is_namespace=not has_init,
        )
    return records


async def load_source_package(
    module_name: str,
    init_file: Path,
    *,
    source_alias: str | None = None,
) -> ModuleType:
    """Load a directory package after an async source-read phase.

    The returned package remains backed by its in-memory finder.  This matters
    for plugins that import a helper module only later during a tool call: the
    lazy relative import is still non-blocking instead of falling back to
    ``SourceFileLoader`` after discovery has completed.
    """
    records = await _package_source_records(module_name, init_file)
    if source_alias is not None:
        records.update(await _package_alias_records(source_alias, init_file))
    finder = _MemorySourceFinder(records)
    modules_before = set(sys.modules)
    sys.meta_path.insert(0, finder)
    try:
        package_record = records[module_name]
        spec = importlib.util.spec_from_loader(
            module_name,
            _MemorySourceLoader(module_name, package_record),
            origin=package_record.filename,
            is_package=True,
        )
        if spec is None:
            raise ImportError(f"Cannot create module spec for {init_file}")
        if spec.submodule_search_locations is not None:
            spec.submodule_search_locations[:] = [str(init_file.parent)]
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(init_file.parent)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except BaseException:
        for loaded_name in tuple(sys.modules):
            if loaded_name not in modules_before and loaded_name in records:
                sys.modules.pop(loaded_name, None)
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        raise
    # Keep the finder alive for later lazy relative imports.  The package
    # namespace is unique per discovery source, and its source cache is bounded
    # to the package files read above.
    setattr(module, "__hermes_async_source_finder__", finder)
    return module


async def load_source_module(
    module_name: str,
    source_file: Path,
    *,
    package_dir: Path | None = None,
) -> ModuleType:
    """Load a standalone module with an async source-read boundary.

    ``package_dir`` may be supplied for entry-point modules.  Its sibling
    ``.py`` files are read into the same finder so a later relative import
    remains native async.  The caller is responsible for registering the
    parent package, matching normal import semantics.
    """
    parent_name, _, leaf_name = module_name.rpartition(".")
    records = {
        module_name: _SourceRecord(
            source=await _read_source(source_file),
            filename=str(source_file),
            is_package=False,
        )
    }
    if package_dir is not None and parent_name:
        package_init = package_dir / "__init__.py"
        if await aiofiles.os.path.isfile(package_init):
            records.update(
                await _package_source_records(parent_name, package_init)
            )
            # The explicitly requested module wins if it was also collected as
            # a sibling of the parent package.
            records[module_name] = _SourceRecord(
                source=await _read_source(source_file),
                filename=str(source_file),
                is_package=False,
            )
        else:
            for child_name in sorted(await aiofiles.os.listdir(package_dir)):
                child = package_dir / child_name
                if child_name == source_file.name or child.suffix != ".py":
                    continue
                if not await aiofiles.os.path.isfile(child):
                    continue
                records[f"{parent_name}.{child.stem}"] = _SourceRecord(
                    source=await _read_source(child),
                    filename=str(child),
                    is_package=False,
                )

    finder = _MemorySourceFinder(records)
    modules_before = set(sys.modules)
    sys.meta_path.insert(0, finder)
    try:
        record = records[module_name]
        spec = importlib.util.spec_from_loader(
            module_name,
            _MemorySourceLoader(module_name, record),
            origin=record.filename,
            is_package=False,
        )
        if spec is None:
            raise ImportError(f"Cannot create module spec for {source_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except BaseException:
        for loaded_name in tuple(sys.modules):
            if loaded_name in modules_before:
                continue
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name, None)
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        raise
    setattr(module, "__hermes_async_source_finder__", finder)
    return module


def unload_source_finder(module: ModuleType) -> None:
    """Remove a finder retained by a previously loaded source module."""
    finder = getattr(module, "__hermes_async_source_finder__", None)
    if finder is None:
        return
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass


__all__ = [
    "locate_source_module",
    "load_source_module",
    "load_source_package",
    "unload_source_finder",
]
