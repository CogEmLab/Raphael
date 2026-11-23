"""Map DeepLabCut pixel coordinates into arena cm, one homography per video.

Every video has its own crop, so there is no global pixel-to-cm scale. The
DLC model tracks the four floor corners of the arena (named by where they
sit in the image) as unique bodyparts, and optionally a reference landmark
on one wall (e.g. a door); their per-video medians fix a homography onto
the arena rectangle.

Convention: the long axis runs along x and the arena spans `arena_cm`
(x by y). With a reference landmark, the arena is rotated so the landmark's
wall is x = 0. Only rotations are applied after the rectangle fit, never a
mirror, so left/right handedness is whatever the camera saw.
"""

import numpy as np

#: Default landmark names (unique bodyparts in the DLC project).
CORNERS = ('CornerTL', 'CornerTR', 'CornerBR', 'CornerBL')
DOOR = 'Door'


def homography(src, dst):
    """3x3 projective transform taking 4+ src points onto dst (DLT)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    matrix = vt[-1].reshape(3, 3)
    return matrix / matrix[2, 2]


def apply(matrix, x, y):
    """Apply a homography to coordinate arrays of any matching shape."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    points = np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    mapped = matrix @ points
    return (
        (mapped[0] / mapped[2]).reshape(x.shape),
        (mapped[1] / mapped[2]).reshape(x.shape),
    )


def landmark_medians(landmarks, pcutoff=0.9):
    """Median (x, y) per landmark over frames where it is confidently seen.

    `landmarks` maps name -> (x, y, likelihood) arrays in pixels.
    """
    medians = {}
    for name, (x, y, likelihood) in landmarks.items():
        good = np.asarray(likelihood) >= pcutoff
        if good.sum() == 0:
            continue
        medians[name] = (
            float(np.median(np.asarray(x)[good])),
            float(np.median(np.asarray(y)[good])),
        )
    return medians


def fit_arena(medians, arena_cm, corners=CORNERS, door=DOOR):
    """Homography from image pixels to arena cm (`arena_cm` = (long, short)
    side lengths, any order), with the `door` landmark's wall at x = 0 if
    it was tracked.

    Returns (matrix, info). Raises ValueError if a corner is missing, since a
    three-point fit would silently shear the whole session.
    """
    missing = [name for name in corners if name not in medians]
    if missing:
        raise ValueError(f'cannot fit arena, corners not tracked: {missing}')
    image = np.array([medians[name] for name in corners])
    top = np.linalg.norm(image[1] - image[0])
    side = np.linalg.norm(image[3] - image[0])
    long_cm, short_cm = max(arena_cm), min(arena_cm)
    width, height = (long_cm, short_cm) if top >= side else (short_cm, long_cm)
    rectangle = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = homography(image, rectangle)

    info = {
        'corners_px': {n: medians[n] for n in corners},
        'door_wall': 'unknown',
    }
    if door and door in medians:
        door_u, door_v = apply(matrix, *medians[door])
        distances = {
            'left': abs(door_u),
            'right': abs(door_u - width),
            'top': abs(door_v),
            'bottom': abs(door_v - height),
        }
        wall = min(distances, key=distances.get)
        info['door_wall'] = wall
        info['door_offset_cm'] = float(distances[wall])
        matrix = _rotation_to_left(wall, width, height) @ matrix
        if wall in ('top', 'bottom') and width >= height:
            info['warning'] = 'door found on a long wall'
    return matrix, info


def _rotation_to_left(wall, width, height):
    """Rigid rotation of the fitted rectangle that puts `wall` at x = 0."""
    if wall == 'left':
        return np.eye(3)
    if wall == 'right':  # 180 degrees
        return np.array([[-1, 0, width], [0, -1, height], [0, 0, 1.0]])
    if wall == 'top':  # 90 degrees clockwise: (u, v) -> (v, width - u)
        return np.array([[0, 1, 0], [-1, 0, width], [0, 0, 1.0]])
    # bottom, 90 degrees counter-clockwise: (u, v) -> (height - v, u)
    return np.array([[0, -1, height], [1, 0, 0], [0, 0, 1.0]])
