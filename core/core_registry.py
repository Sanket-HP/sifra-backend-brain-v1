# ============================================================
# SIFRA CORE REGISTRY
# Ensures SINGLE SifraCore instance per process
# ============================================================

from core.sifra_core import SifraCore

_CORE = None


def get_core():
    global _CORE
    if _CORE is None:
        _CORE = SifraCore()
    return _CORE
