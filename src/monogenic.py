"""
Monogenic-signal front end of monounet models, computed with SciPy.

These models start with fixed FFT filtering: the image spectrum is multiplied by a
log-Gabor filter bank and by a Riesz multiplier, and both products are transformed
back. Some of them first standardise the image to zero mean and unit standard
deviation. ONNX Runtime's DFT is slow (most of the model's run time), so the model is
cut just after the inverse transforms: SciPy does the filtering and ONNX Runtime runs
the rest of the network from there.
"""

import mmap

import numpy as np
import scipy.fft

# Ends of the filter bank's name, such as "encoder.mono2d.lifted_tensor_0". The
# "lifted_tensor_0" tensor is the log-Gabor filter bank, [1, scales, H, W], and
# "lifted_tensor_1" next to it the Riesz multiplier, [H, W, 2] as (real, imaginary).
FILTERS = (b".mono2d.lifted_tensor_0", b".mono.lifted_tensor_0")
NAME_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
# Inverse transforms of the band-passed spectra (even part) and of the band-passed
# spectra times the Riesz multiplier (odd parts), each [1, 1, scales, H, W, 2].
EVEN = "_fft_c2c_1"
ODD = "_fft_c2c_2"

class MonogenicFrontEnd:
    def __init__(self, filters: np.ndarray, riesz: np.ndarray, min_std: float | None = None):
        self._filters = filters[0].astype(np.float32)
        self._riesz = (riesz[..., 0] + 1j * riesz[..., 1]).astype(np.complex64)
        # Lower bound on the standard deviation the image is divided by, or None for a
        # model that filters the image as it is.
        self._min_std = min_std

    def features(self, model_input: np.ndarray) -> dict[str, np.ndarray]:
        """The network's EVEN and ODD inputs for a [1, 1, H, W] float32 image."""
        image = model_input[0, 0]
        if self._min_std is not None:
            image = image - image.mean()
            image /= max(float(np.sqrt(np.mean(image * image))), self._min_std)
        spectrum = scipy.fft.fft2(image, norm="ortho", workers=-1)
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
        filters_name = _full_name(data, FILTERS)
    if filters_name is None:
        return None
    riesz_name = filters_name.removesuffix("0") + "1"

    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    produced = {name for node in graph.node for name in node.output}
    if not {filters_name, riesz_name} <= initializers.keys() or not {EVEN, ODD} <= produced:
        return None

    image_name = graph.input[0].name
    outputs = [o.name for o in graph.output]
    network = onnx.utils.Extractor(model).extract_model([image_name, EVEN, ODD], outputs)
    if any(node.op_type == "DFT" for node in network.graph.node):
        return None

    front_end = MonogenicFrontEnd(
        numpy_helper.to_array(initializers[filters_name]),
        numpy_helper.to_array(initializers[riesz_name]),
        _standardisation_min_std(graph, image_name),
    )
    return network.SerializeToString(), front_end


def _full_name(data, endings) -> str | None:
    """The first name in the file that ends in one of the endings."""
    for ending in endings:
        end = data.find(ending)
        if end < 0:
            continue
        start = end
        while start > 0 and data[start - 1] in NAME_BYTES:
            start -= 1
        return data[start : end + len(ending)].decode()
    return None


def _standardisation_min_std(graph, image_name: str) -> float | None:
    """The lower bound on the standard deviation when the model standardises the image
    before filtering it, as (image - mean) / clip(std, min), or None when it does not."""
    from onnx import AttributeProto, numpy_helper

    producers = {out: node for node in graph.node for out in node.output}
    values = {t.name: t for t in graph.initializer}
    values.update(
        (node.output[0], node.attribute[0].t)
        for node in graph.node
        if node.op_type == "Constant" and node.attribute and node.attribute[0].type == AttributeProto.TENSOR
    )

    def made_by(name, op_type):
        node = producers.get(name)
        return node if node is not None and node.op_type == op_type else None

    for node in graph.node:
        if node.op_type != "Div":
            continue
        centred = made_by(node.input[0], "Sub")
        clip = made_by(node.input[1], "Clip")
        if centred is None or clip is None or centred.input[0] != image_name:
            continue
        mean = made_by(centred.input[1], "ReduceMean")
        if mean is None or mean.input[0] != image_name:
            continue
        if len(clip.input) < 2 or clip.input[1] not in values:
            return None
        return float(numpy_helper.to_array(values[clip.input[1]]))
    return None
