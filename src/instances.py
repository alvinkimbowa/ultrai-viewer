"""
Instance masks: one binary mask per labelled structure, saved one PNG each.

The file layout is the one the web annotator writes, so either app reads the
other's masks. Under a results folder, an image or video has a folder named
after its file without the ending:

    <stem>/<class>_<NNN>.png                  masks of an image
    <stem>/frame_<NNNNNN>_<class>_<NNN>.png   masks of a video frame

<class> is the label with every run of characters other than a-z and 0-9
turned into "_", and <NNN> numbers the instances of one class on one image or
frame from 1.
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np

from .rotations import rotate_mask, unrotate_mask

MASK_FILE = re.compile(r"^(?:frame_(\d{6})_)?(.+)_(\d{3,})\.png$")


def clean_class(name):
    return str(name or "").strip().lower()


def safe_class(name):
    return re.sub(r"[^a-z0-9]+", "_", clean_class(name)).strip("_") or "unlabeled"


def display_class(name):
    label = clean_class(name)
    return "LFCN" if label == "lfcn" else label.title()


def instance_title(instance):
    return f"{display_class(instance['class'])} #{instance['id']}"


def make_instance(class_name, instance_id, mask):
    return {
        "class": clean_class(class_name),
        "id": int(instance_id),
        "mask": (np.asarray(mask) > 0).astype(np.uint8),
    }


def copy_instances(instances):
    return [make_instance(i["class"], i["id"], i["mask"]) for i in instances]


def next_instance_id(instances, class_name):
    """The lowest id above every id the class already has."""
    class_name = clean_class(class_name)
    taken = [i["id"] for i in instances if i["class"] == class_name]
    return max(taken, default=0) + 1


def same_instances(first, second):
    """True when two lists hold the same classes, ids and pixels, in any order."""
    if len(first) != len(second):
        return False
    by_key = {(i["class"], i["id"]): i["mask"] for i in second}
    for instance in first:
        other = by_key.get((instance["class"], instance["id"]))
        if other is None or other.shape != instance["mask"].shape:
            return False
        if not np.array_equal(other > 0, instance["mask"] > 0):
            return False
    return True


def write_mask_png(path, mask):
    """Save a 0/255 PNG, written under another name and then moved into place so
    that a mask being saved is never read half-written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    try:
        if not cv2.imwrite(str(partial), (np.asarray(mask) > 0).astype(np.uint8) * 255):
            raise OSError(f"Failed to save mask: {path.name}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def annotated_frames(folder):
    """Frame numbers that have at least one mask file in a video's folder."""
    frames = set()
    try:
        names = os.listdir(folder)
    except OSError:
        return frames
    for name in names:
        match = MASK_FILE.match(name)
        if match and match.group(1) is not None:
            frames.add(int(match.group(1)))
    return frames


class MaskSet:
    """The mask files of one image (frame None) or one video frame."""

    def __init__(self, folder, frame=None):
        self.folder = Path(folder)
        self.frame = None if frame is None or frame == "" else int(frame)

    def __eq__(self, other):
        return (
            isinstance(other, MaskSet)
            and self.folder == other.folder
            and self.frame == other.frame
        )

    def __hash__(self):
        return hash((self.folder, self.frame))

    def file_name(self, class_name, instance_id):
        prefix = "" if self.frame is None else f"frame_{self.frame:06d}_"
        return f"{prefix}{safe_class(class_name)}_{int(instance_id):03d}.png"

    def files(self):
        """(class, id, path) of each mask file, ordered by class and id."""
        try:
            names = sorted(os.listdir(self.folder))
        except OSError:
            return []
        found = []
        for name in names:
            match = MASK_FILE.match(name)
            if match is None:
                continue
            frame = None if match.group(1) is None else int(match.group(1))
            if frame != self.frame:
                continue
            path = self.folder / name
            if path.is_file():
                found.append(
                    (match.group(2).replace("_", " "), int(match.group(3)), path)
                )
        return found

    def exists(self):
        return bool(self.files())

    def read(self, angle=0.0):
        """The masks as instances, turned the way the image is viewed."""
        instances = []
        for class_name, instance_id, path in self.files():
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            mask = raw >= (128 if raw.max() > 1 else 1)
            if angle:
                mask = rotate_mask(mask, angle)
            instances.append(make_instance(class_name, instance_id, mask))
        return instances

    def stored_shape(self):
        """(height, width) of the saved masks, or None when there are none."""
        for _, _, path in self.files():
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                return raw.shape[:2]
        return None

    def write(self, instances, angle=0.0, original_shape=None, classes=None):
        """Save instances as viewed at angle onto the stored image, whose (height,
        width) is original_shape, and delete the files of instances not among
        them. With classes, only files of those classes are deleted."""
        keep = set()
        for instance in instances:
            mask = instance["mask"] > 0
            if angle:
                mask = unrotate_mask(mask, angle, original_shape)
            name = self.file_name(instance["class"], instance["id"])
            write_mask_png(self.folder / name, mask)
            keep.add(name)
        limit = None if classes is None else {safe_class(c) for c in classes}
        for class_name, _, path in self.files():
            if path.name in keep:
                continue
            if limit is None or safe_class(class_name) in limit:
                path.unlink(missing_ok=True)

    def remove(self):
        for _, _, path in self.files():
            path.unlink(missing_ok=True)

    def snapshot(self):
        """{file name: bytes} of every mask file, for putting them back later."""
        return {path.name: path.read_bytes() for _, _, path in self.files()}

    def restore(self, snapshot):
        for _, _, path in self.files():
            if path.name not in snapshot:
                path.unlink()
        for name, data in snapshot.items():
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def clip_instances(instances, box):
    """The instances emptied outside an ROI box (left, right, top, bottom), with
    those left empty dropped."""
    clipped = []
    for instance in instances:
        mask = instance["mask"]
        if box is not None:
            left, right, top, bottom = box
            inside = np.zeros_like(mask)
            inside[top:bottom, left:right] = mask[top:bottom, left:right]
            mask = inside
        if mask.any():
            clipped.append(make_instance(instance["class"], instance["id"], mask))
    return clipped


def instances_from_labels(labels, class_names):
    """Turn a map of class indices into instances, one per connected piece.

    class_names maps a class index to its label; indices it leaves out are not
    turned into instances.
    """
    labels = np.asarray(labels)
    instances = []
    for index in sorted(int(v) for v in np.unique(labels) if v > 0):
        name = class_names.get(index)
        if not name:
            continue
        count, pieces = cv2.connectedComponents(
            (labels == index).astype(np.uint8), connectivity=8
        )
        for piece in range(1, count):
            mask = pieces == piece
            instances.append(
                make_instance(name, next_instance_id(instances, name), mask)
            )
    return instances
