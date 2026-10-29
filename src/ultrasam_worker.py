"""
UltraSam helper process for UltrAi Viewer.

UltraSam is SAM ViT-B trained on ultrasound. Its checkpoint is kept in
MMDetection's layout; this script maps it onto Meta's segment_anything modules,
so it runs with PyTorch and segment_anything alone. The viewer has no PyTorch,
so it runs this script with the Python of an environment that has them, and
talks to it through the pipes: one JSON request per line on stdin, one JSON
reply per line on stdout. Every reply carries the "id" of its request. Anything
else the libraries print goes to stderr.

Requests:
    points  {"image", "key", "points", "labels", "out", "mask"?}  the mask that
            include (1) and exclude (0) points, and a box given as corner points
            labelled 2 and 3, select on an 8-bit RGB PNG; points are its pixels.
            "mask", a PNG of the image's size, prompts as well. The image
            embedding is kept while requests carry the same key.
    load    {}  loads the model, so that the first prompt need not wait for it
    quit

A prompt is answered as UltraSam answers it: one mask from the prompt, then one
refinement that prompts with that mask and its bounding box, or the box given,
in place of the points, and keeps the best of three masks. A prompt mask enters
the first pass as UltraSam's refinement takes a mask: as mask logits, with its
bounding box when no box is given.
"""

import json
import os
import pickle
import sys
import traceback
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The viewer's modules sit next to this script; generic names among them
# (labels, measurements) must not shadow modules PyTorch and its dependencies import.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != SCRIPT_DIR]

PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SIDE = 1024
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
NO_MASK, NEGATIVE, POSITIVE, CORNER_A, CORNER_B, NOT_A_POINT = range(6)
EMBEDDING_OF_LABEL = {0: NEGATIVE, 1: POSITIVE, 2: CORNER_A, 3: CORNER_B}
# The magnitude of the logits a drawn mask is given, inside and outside it.
PROMPT_MASK_LOGIT = 8.0


def send(message):
    PROTOCOL.write(json.dumps(message) + "\n")
    PROTOCOL.flush()


class _StubType(type):
    def __getattr__(cls, name):
        return cls


class _Stub(metaclass=_StubType):
    """Stands in for the MMEngine objects, and the methods of them, pickled
    beside the weights."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("mmengine", "mmdet", "mmcv", "mmpretrain", "endosam"):
            return _Stub
        return super().find_class(module, name)


def mmdet_weights(path):
    pickle_module = types.SimpleNamespace(
        __name__="pickle", Unpickler=_Unpickler, load=pickle.load
    )
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=pickle_module
    )
    return checkpoint.get("state_dict", checkpoint)


def _rename(key, pairs):
    for old, new in pairs:
        key = key.replace(old, new)
    return key


def segment_anything_weights(weights):
    """The UltraSam weights under segment_anything's names."""
    state = {}
    for key, value in weights.items():
        if key.startswith("backbone."):
            state["image_encoder." + _rename(key[len("backbone."):], (
                ("patch_embed.projection", "patch_embed.proj"),
                ("ffn.layers.0.0", "mlp.lin1"),
                ("ffn.layers.1", "mlp.lin2"),
                ("layers.", "blocks."),
                ("ln1", "norm1"),
                ("ln2", "norm2"),
                ("channel_reduction", "neck"),
            ))] = value
        elif key.startswith("bbox_head."):
            state["mask_decoder." + key[len("bbox_head."):]] = value
        elif key.startswith("decoder."):
            name = "mask_decoder.transformer." + _rename(key[len("decoder."):], (
                (".attn.", "."),
                ("mlp.layers.0.0", "mlp.lin1"),
                ("mlp.layers.1", "mlp.lin2"),
                ("post_norm", "norm_final_attn"),
            ))
            if name.endswith("_proj_weight"):
                state[name[: -len("_weight")] + ".weight"] = value
            elif name.endswith("in_proj_bias"):
                stem = name[: -len("in_proj_bias")]
                for part, bias in zip(("q", "k", "v"), value.chunk(3)):
                    state[f"{stem}{part}_proj.bias"] = bias
            else:
                state[name] = value
        elif key == "prompt_encoder.label_encoder.label_embedding.weight":
            state["prompt_encoder.no_mask_embed.weight"] = value[NO_MASK:NO_MASK + 1]
            for index in range(4):
                state[f"prompt_encoder.point_embeddings.{index}.weight"] = value[
                    NEGATIVE + index:NEGATIVE + index + 1
                ]
            state["prompt_encoder.not_a_point_embed.weight"] = value[NOT_A_POINT:NOT_A_POINT + 1]
            state["mask_decoder.mask_tokens.weight"] = value[6:10]
            state["mask_decoder.iou_token.weight"] = value[10:11]
        else:
            state[key] = value
    return state


class UltraSam:
    def __init__(self, checkpoint):
        from segment_anything import sam_model_registry

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sam_model_registry["vit_b"](checkpoint=None)
        model.load_state_dict(segment_anything_weights(mmdet_weights(checkpoint)), strict=True)
        self.model = model.to(self.device).eval()
        # CUDA sets itself up on the first forward pass; doing it here keeps that
        # wait out of the first prompt.
        with torch.inference_mode():
            self.model.image_encoder(torch.zeros(1, 3, SIDE, SIDE, device=self.device))
        self._key = None
        self._features = None
        self._scale = None
        self._resized = None
        self._shape = None

    def _embed(self, image, key):
        if key is not None and key == self._key:
            return
        height, width = image.shape[:2]
        scale = SIDE / max(height, width)
        resized = (int(width * scale + 0.5), int(height * scale + 0.5))
        pixels = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
        pixels = (pixels.astype(np.float32) - MEAN) / STD
        padded = np.zeros((SIDE, SIDE, 3), dtype=np.float32)
        padded[: resized[1], : resized[0]] = pixels
        tensor = torch.from_numpy(padded).permute(2, 0, 1)[None].to(self.device)
        self._features = self.model.image_encoder(tensor)
        self._scale = (resized[0] / width, resized[1] / height)
        self._resized = resized
        self._shape = (height, width)
        self._key = key

    def _decode(self, coords, embeddings, mask_logits, multimask):
        """Mask logits at 256 x 256 and the predicted IoU of the mask chosen."""
        encoder = self.model.prompt_encoder
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)[None]
        indices = torch.tensor(embeddings, device=self.device)
        position = encoder.pe_layer.forward_with_coords(coords, (SIDE, SIDE))
        position[:, indices == NOT_A_POINT] = 0.0
        table = torch.cat(
            [encoder.no_mask_embed.weight]
            + [embedding.weight for embedding in encoder.point_embeddings]
            + [encoder.not_a_point_embed.weight]
        )
        sparse = table[indices][None] + position
        if mask_logits is None:
            dense = encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, *encoder.image_embedding_size
            )
        else:
            dense = encoder.mask_downscaling(mask_logits)
        logits, iou = self.model.mask_decoder.predict_masks(
            image_embeddings=self._features,
            image_pe=encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )
        if not multimask:
            return logits[:, 0:1], float(iou[0, 0])
        best = int(iou[0, 1:].argmax()) + 1
        return logits[:, best:best + 1], float(iou[0, best])

    def _mask(self, logits):
        """The mask of 256 x 256 logits at the size of the image, made as
        UltraSam makes it: the thresholded mask is resized, and any pixel it
        touches is kept."""
        mask = (logits > 0).float()
        mask = F.interpolate(mask, (SIDE, SIDE), mode="bilinear", align_corners=False)
        mask = mask[..., : self._resized[1], : self._resized[0]]
        mask = F.interpolate(mask, self._shape, mode="bilinear", align_corners=False)
        return (mask[0, 0] > 0).cpu().numpy()

    def _mask_logits(self, mask):
        """A mask of the image as the 256 x 256 logits the prompt encoder takes."""
        resized = cv2.resize(
            mask.astype(np.float32), self._resized, interpolation=cv2.INTER_LINEAR
        )
        padded = np.zeros((SIDE, SIDE), dtype=np.float32)
        padded[: self._resized[1], : self._resized[0]] = resized
        small = cv2.resize(padded, (SIDE // 4, SIDE // 4), interpolation=cv2.INTER_AREA)
        logits = np.where(small > 0.5, PROMPT_MASK_LOGIT, -PROMPT_MASK_LOGIT)
        return torch.from_numpy(logits.astype(np.float32))[None, None].to(self.device)

    @torch.inference_mode()
    def points(self, image, key, points, labels, prompt_mask=None):
        self._embed(image, key)
        sx, sy = self._scale
        scaled = [[x * sx, y * sy] for x, y in points]
        embeddings = [EMBEDDING_OF_LABEL[int(label)] for label in labels]
        box = [p for p, label in zip(scaled, labels) if label in (2, 3)]
        clicks = [(p, e) for p, e, label in zip(scaled, embeddings, labels) if label in (0, 1)]
        dense = None
        if prompt_mask is not None and prompt_mask.any():
            dense = self._mask_logits(prompt_mask)
            if not box:
                ys, xs = np.nonzero(prompt_mask)
                box = [
                    [float(xs.min()) * sx, float(ys.min()) * sy],
                    [float(xs.max() + 1) * sx, float(ys.max() + 1) * sy],
                ]
        coords = [p for p, _ in clicks] + box
        indices = [e for _, e in clicks] + [CORNER_A, CORNER_B][: len(box)]
        if not box:
            coords.append([0.0, 0.0])
            indices.append(NOT_A_POINT)
        logits, _ = self._decode(coords, indices, dense, multimask=False)
        mask = self._mask(logits)
        if not mask.any():
            return mask
        if not box:
            ys, xs = np.nonzero(mask)
            box = [
                [float(xs.min()) * sx, float(ys.min()) * sy],
                [float(xs.max() + 1) * sx, float(ys.max() + 1) * sy],
            ]
        logits, _ = self._decode(box, [CORNER_A, CORNER_B], logits, multimask=True)
        return self._mask(logits)


def main():
    checkpoint = sys.argv[1]
    sam = None
    send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        cmd, reply_id = request.get("cmd"), request.get("id")
        if cmd == "quit":
            return
        try:
            if cmd not in ("points", "load"):
                raise RuntimeError(f"Unknown request {cmd!r}")
            if sam is None:
                sam = UltraSam(checkpoint)
            if cmd == "load":
                send({"id": reply_id, "ok": True})
                continue
            image = cv2.imread(request["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {request['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prompt_mask = None
            if request.get("mask"):
                prompt_mask = cv2.imread(request["mask"], cv2.IMREAD_GRAYSCALE) > 127
            mask = sam.points(
                image, request.get("key"), request["points"], request["labels"], prompt_mask
            )
            cv2.imwrite(request["out"], mask.astype(np.uint8) * 255)
            send({"id": reply_id, "ok": True, "area": int(mask.sum())})
        except Exception as exc:
            traceback.print_exc()
            send({"id": reply_id, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
