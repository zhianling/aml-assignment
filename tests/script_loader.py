import importlib.util
import sys
from types import ModuleType

from birdcls.config import REPO_ROOT


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module, so tests can call its functions.

    The module is registered in ``sys.modules`` because dataclasses look their module up there.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
