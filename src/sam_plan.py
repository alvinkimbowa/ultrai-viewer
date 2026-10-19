"""
How SAM 3.1 tracks one object through a video, shared by the viewer, which
counts the steps for its progress bar, and the helper process, which runs them.
It imports nothing, so that both can load it.

The tracker fails with more than PROMPTS_PER_TRACK prompt frames in one track,
so an object with more is tracked in stretches: each holds at most that many
consecutive prompt frames, shares its last prompt frame with the next one, and
saves the frames from its first prompt frame up to the next stretch's first.
"""

PROMPTS_PER_TRACK = 16


def track_plan(prompt_frames, start, direction, frame_count):
    """The stretches tracking one object takes, each a dict of:

    prompts: its prompt frames, in order
    passes: (first frame, last frame, reverse) of each pass from start that
        gives the frames saved; a pass visits both ends
    saved: the frames whose masks it gives

    direction is "both", "forward" or "backward" from start.
    """
    frames = sorted(set(int(f) for f in prompt_frames))
    if direction == "forward":
        wanted = range(start, frame_count)
    elif direction == "backward":
        wanted = range(0, start + 1)
    else:
        wanted = range(frame_count)
    step = PROMPTS_PER_TRACK - 1
    groups = [frames[i:i + PROMPTS_PER_TRACK] for i in range(0, max(1, len(frames) - 1), step)]
    plan = []
    for index, group in enumerate(groups):
        low = 0 if index == 0 else group[0]
        high = frame_count - 1 if index == len(groups) - 1 else groups[index + 1][0] - 1
        saved = [f for f in wanted if low <= f <= high]
        if not saved:
            continue
        passes = []
        after = [f for f in saved if f >= start]
        before = [f for f in saved if f < start]
        if after:
            passes.append((after[0], after[-1], False))
        if before:
            passes.append((before[-1], before[0], True))
        plan.append({"prompts": group, "passes": passes, "saved": saved})
    return plan


def stretch_steps(stretch):
    """Frames a stretch visits: from its first prompt frame to its last, each
    stretch between two prompt frames visiting both, then its passes."""
    prompts = stretch["prompts"]
    seeding = prompts[-1] - prompts[0] + len(prompts) - 1
    return seeding + sum(abs(last - first) + 1 for first, last, _ in stretch["passes"])


def propagation_steps(prompt_frames, start, direction, frame_count):
    """The progress steps of a propagation; prompt_frames holds the prompt
    frames of each object."""
    return sum(
        stretch_steps(stretch)
        for frames in prompt_frames
        for stretch in track_plan(frames, start, direction, frame_count)
    )
