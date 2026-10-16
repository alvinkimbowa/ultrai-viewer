"""
Monogenic-signal front end of monounet models, computed with SciPy.

These models start with fixed FFT filtering: the image spectrum is multiplied by a
log-Gabor filter bank and by a Riesz multiplier, and both products are transformed
back. ONNX Runtime's DFT is slow (most of the model's run time), so the model is cut
just after the inverse transforms: SciPy does the filtering and ONNX Runtime runs the
rest of the network from there.
"""

import mmap

import numpy as np
import scipy.fft

# Log-Gabor filter bank, [1, scales, H, W].
FILTERS = "encoder.mono2d.lifted_tensor_0"
# Riesz multiplier, [H, W, 2] as (real, imaginary).
RIESZ = "encoder.mono2d.lifted_tensor_1"
# Inverse transforms of the band-passed spectra (even part) and of the band-passed
# spectra times the Riesz multiplier (odd parts), each [1, 1, scales, H, W, 2].
EVEN = "_fft_c2c_1"
ODD = "_fft_c2c_2"

class MonogenicFrontEnd:
    def __init__(self, filters: np.ndarray, riesz: np.ndarray):
        self._filters = filters[0].astype(np.float32)
        self._riesz = (riesz[..., 0] + 1j * riesz[..., 1]).astype(np.complex64)

    def features(self, model_input: np.ndarray) -> dict[str, np.ndarray]:
        """The network's EVEN and ODD inputs for a [1, 1, H, W] float32 image."""
        spectrum = scipy.fft.fft2(model_input[0, 0], norm="ortho", workers=-1)
        scales = len(self._filters)
        planes = np.empty((2 * scales,) + spectrum.shape, dtype=np.complex64)
        np.multiply(spectrum, self._filters, out=planes[:scales])
        np.multiply(planes[:scales], self._riesz, out=planes[scales:])
        planes = scipy.fft.ifft2(planes, workers=-1, overwrite_x=True)
        # complex64 is stored as (real, imaginary) float32 pairs: the layout the
        # network expects, so the view needs no copy.
        out = planes.view(np.float32).reshape(planes.shape + (2,))
        return {EVEN: out[None, None, :scales], ODD: out[None, None, scales:]}


def split_monogenic(model_path: str):
    """(network as ONNX bytes, MonogenicFrontEnd) for a monounet model, or None for
    any other model. The network takes the image plus EVEN and ODD as inputs."""
    # Parsing a large model takes a while; a byte search for the filter bank's name
    # rules out other models first.
    with open(model_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data.find(FILTERS.encode()) < 0:
            return None

    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    produced = {name for node in graph.node for name in node.output}
    if not {FILTERS, RIESZ} <= initializers.keys() or not {EVEN, ODD} <= produced:
        return None

    image_name = graph.input[0].name
    outputs = [o.name for o in graph.output]
    network = onnx.utils.Extractor(model).extract_model([image_name, EVEN, ODD], outputs)
    if any(node.op_type == "DFT" for node in network.graph.node):
        return None

    front_end = MonogenicFrontEnd(
        numpy_helper.to_array(initializers[FILTERS]),
        numpy_helper.to_array(initializers[RIESZ]),
    )
    return network.SerializeToString(), front_end
