"""Import shim for the censored-EM module.

Preferred path is the normal package import. In a minimal dev environment the
top-level ``mobius`` package pulls in rdkit / openmm / ray / prody at import
time, none of which the censored-EM code actually needs, so we fall back to
loading the module file directly inside a synthetic package. This keeps the
maths and unit tests runnable in CI containers that do not carry the full
scientific stack.
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

__all__ = ["CensoredEMGPModel", "EMResult", "CGP_MODULE", "IMPORTED_VIA_PACKAGE"]

_MODPATH = "mobius.surrogate_models.censored_gaussian_process"
IMPORTED_VIA_PACKAGE = True


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses needs this before exec
    spec.loader.exec_module(module)
    return module


def _load_standalone():
    here = pathlib.Path(__file__).resolve()
    root = next(
        (p for p in here.parents if (p / "mobius" / "surrogate_models").is_dir()),
        None,
    )
    if root is None:  # pragma: no cover
        raise ImportError(f"Could not locate the mobius package relative to {here}")

    pkg_dir = root / "mobius"
    sub_dir = pkg_dir / "surrogate_models"

    pkg = types.ModuleType("_mobius_shim")
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["_mobius_shim"] = pkg

    sub = types.ModuleType("_mobius_shim.surrogate_models")
    sub.__path__ = [str(sub_dir)]
    sys.modules["_mobius_shim.surrogate_models"] = sub

    _load("_mobius_shim.surrogate_models.surrogate_model",
          sub_dir / "surrogate_model.py")
    return _load("_mobius_shim.surrogate_models.censored_gaussian_process",
                 sub_dir / "censored_gaussian_process.py")


try:  # pragma: no cover - environment dependent
    CGP_MODULE = importlib.import_module(_MODPATH)
except Exception:  # noqa: BLE001 - any dependency failure should fall through
    IMPORTED_VIA_PACKAGE = False
    CGP_MODULE = _load_standalone()

CensoredEMGPModel = CGP_MODULE.CensoredEMGPModel
EMResult = CGP_MODULE.EMResult


def load_gp_model():
    """Return ``mobius``'s ``GPModel``, via the shim if the package won't import.

    Used by the slow integration tests so they still exercise a real
    GPyTorch/BoTorch surrogate in a container without rdkit/openmm.
    """
    if IMPORTED_VIA_PACKAGE:
        from mobius.surrogate_models.gaussian_process import GPModel
        return GPModel

    name = "_mobius_shim.surrogate_models.gaussian_process"
    if name in sys.modules:
        return sys.modules[name].GPModel

    # GPModel only needs ProgressBar out of mobius.utils, which drags in rdkit.
    if "_mobius_shim.utils" not in sys.modules:
        stub = types.ModuleType("_mobius_shim.utils")

        class _ProgressBar:  # pragma: no cover - display only
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                pass

        stub.ProgressBar = _ProgressBar
        sys.modules["_mobius_shim.utils"] = stub

    sub_dir = pathlib.Path(sys.modules["_mobius_shim.surrogate_models"].__path__[0])
    return _load(name, sub_dir / "gaussian_process.py").GPModel
