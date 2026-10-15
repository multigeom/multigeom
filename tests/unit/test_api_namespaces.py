"""
The public API is grouped in three subpackages, and the main functions are also
available from `multigeom`.
"""

from __future__ import annotations

import multigeom
from multigeom import analysis, encoders, geometry


def test_each_group_has_its_functions():
    assert {"load_encoder", "NeuralEncoder", "validate_encoder"} <= set(
        encoders.__all__
    )
    assert {"compute_fisher", "compute_g"} <= set(geometry.__all__)
    assert {"subspace_alignment", "bures_wasserstein"} <= set(analysis.__all__)


def test_the_top_level_names_are_the_same_objects():
    assert multigeom.load_encoder is encoders.load_encoder
    assert multigeom.NeuralEncoder is encoders.NeuralEncoder
    assert multigeom.validate_encoder is encoders.validate_encoder
    assert multigeom.compute_fisher is geometry.compute_fisher
    assert multigeom.compute_g is geometry.compute_g
    assert multigeom.subspace_alignment is analysis.subspace_alignment
    assert multigeom.bures_wasserstein is analysis.bures_wasserstein


def test_the_old_vague_name_is_gone_and_the_analyses_left_geometry():
    assert not hasattr(multigeom, "align") and not hasattr(multigeom, "estimate_g")
    assert not hasattr(geometry, "subspace_alignment") and not hasattr(
        geometry, "bures_wasserstein"
    )


def test_importing_the_package_stays_light():
    import subprocess
    import sys

    code = (
        "import sys, multigeom; assert 'torchvision' not in sys.modules and "
        "'matplotlib' not in sys.modules"
    )
    assert (
        subprocess.run([sys.executable, "-c", code], capture_output=True).returncode
        == 0
    )


def test_the_builtin_readouts_are_registered_whatever_the_import_order():
    import subprocess
    import sys

    code = (
        "from multigeom.encoders.neural import NeuralEncoder; from "
        "multigeom.encoders.registry import READOUTS, BACKBONES; "
        "assert {'gaussian', 'linear', 'factorized'} <= set(READOUTS) and 'small_cnn' "
        "in BACKBONES, (list(READOUTS), list(BACKBONES))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
