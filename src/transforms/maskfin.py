"""
Phase 4: Mask finalization.

Extracts body part annotations (areolas, nipples, belly, etc.) from the
GAN-detected mask and composites them onto the refined mask to produce
the final mask used for nude generation.
"""

import numpy as np
import cv2
import random

from .annotation import BodyPart


def create_maskfin(maskref, maskdet):
    """Create the final mask by overlaying detected body parts on maskref.

    Args:
        maskref: Refined mask from phase 2.
        maskdet: Anatomical detail mask from the maskref-to-maskdet GAN phase.

    Returns:
        maskfin image (512x512 BGR), or None if no body parts detected.
    """
    # Solid green base for drawing body part details
    details = np.zeros((512, 512, 3), np.uint8)
    details[:, :, :] = (0, 255, 0)  # BGR

    bodypart_list = _extract_annotations(maskdet)

    if not bodypart_list:
        return None

    for obj in bodypart_list:
        if obj.w < obj.h:
            a_max = int(obj.h / 2)
            a_min = int(obj.w / 2)
            angle = 0
        else:
            a_max = int(obj.w / 2)
            a_min = int(obj.h / 2)
            angle = 90

        x, y = int(obj.x), int(obj.y)

        colors = {
            "tit": (0, 205, 0),
            "aur": (0, 0, 255),
            "nip": (255, 255, 255),
            "belly": (255, 0, 255),
            "vag": (255, 0, 0),
        }

        if obj.name == "hair":
            xmin = x - int(obj.w / 2)
            ymin = y - int(obj.h / 2)
            xmax = x + int(obj.w / 2)
            ymax = y + int(obj.h / 2)
            cv2.rectangle(details, (xmin, ymin), (xmax, ymax),
                          (100, 100, 100), -1)
        elif obj.name in colors:
            cv2.ellipse(details, (x, y), (a_max, a_min), angle,
                        0, 360, colors[obj.name], -1)

    # Composite details onto maskref
    f1 = np.asarray([0, 250, 0])
    f2 = np.asarray([10, 255, 10])
    green_mask = cv2.bitwise_not(cv2.inRange(maskref, f1, f2))
    green_mask_inv = cv2.bitwise_not(green_mask)

    res1 = cv2.bitwise_and(maskref, maskref, mask=green_mask)
    res2 = cv2.bitwise_and(details, details, mask=green_mask_inv)

    return cv2.add(res1, res2)


# ---------------------------------------------------------------------------
# Annotation extraction
# ---------------------------------------------------------------------------

def _extract_annotations(maskdet):
    """Detect and classify body parts from the maskdet image.

    Returns:
        List of BodyPart objects.
    """
    tits_list = _find_body_part(maskdet, "tit")
    aur_list = _find_body_part(maskdet, "aur")
    vag_list = _find_body_part(maskdet, "vag")
    belly_list = _find_body_part(maskdet, "belly")

    # Filter by dimension (area and aspect ratio)
    aur_list = _filter_dim_parts(aur_list, 100, 1000, 0.5, 3)
    tits_list = _filter_dim_parts(tits_list, 1000, 60000, 0.2, 3)
    vag_list = _filter_dim_parts(vag_list, 10, 1000, 0.2, 3)
    belly_list = _filter_dim_parts(belly_list, 10, 1000, 0.2, 3)

    # Keep at most 2 parts per type
    aur_list = _filter_couple(aur_list)
    tits_list = _filter_couple(tits_list)

    # Detect and resolve missing parts
    problem = _detect_missing_problem(tits_list, aur_list)
    if problem in [3, 6, 7, 8]:
        _resolve_missing_problems(tits_list, aur_list, problem)

    nip_list = _infer_nip(aur_list)
    hair_list = _infer_hair(vag_list)

    return tits_list + aur_list + nip_list + vag_list + hair_list + belly_list


def _find_body_part(image, part_name):
    """Find contours of a body part by color filtering and ellipse fitting.

    Args:
        image: BGR maskdet image.
        part_name: One of 'tit', 'aur', 'vag', 'belly'.

    Returns:
        List of BodyPart objects found.
    """
    bodypart_list = []

    color_filters = {
        "tit": [
            (np.asarray([0, 0, 0]), np.asarray([10, 10, 10])),
            (np.asarray([0, 0, 250]), np.asarray([0, 0, 255])),
        ],
        "aur": [(np.asarray([0, 0, 250]), np.asarray([0, 0, 255]))],
        "vag": [(np.asarray([250, 0, 0]), np.asarray([255, 0, 0]))],
        "belly": [(np.asarray([250, 0, 250]), np.asarray([255, 0, 255]))],
    }

    filters = color_filters.get(part_name, [])
    if not filters:
        return bodypart_list

    # Build combined color mask
    color_mask = cv2.inRange(image, *filters[0])
    for f1, f2 in filters[1:]:
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(image, f1, f2))

    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if len(cnt) <= 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        x = ellipse[0][0]
        y = ellipse[0][1]
        a_min = ellipse[1][0]
        a_max = ellipse[1][1]

        if ellipse[2] == 0:
            h, w = a_max, a_min
        else:
            h, w = a_min, a_max

        # Normalize small body parts
        if part_name in ("belly", "vag"):
            if w < 15:
                w *= 2
            if h < 15:
                h *= 2

        xmin = int(x - w / 2)
        xmax = int(x + w / 2)
        ymin = int(y - h / 2)
        ymax = int(y + h / 2)

        bodypart_list.append(
            BodyPart(part_name, xmin, ymin, xmax, ymax, x, y, w, h)
        )

    return bodypart_list


def _filter_dim_parts(bp_list, min_area, max_area, min_ar, max_ar):
    """Filter body parts by area and aspect ratio."""
    filtered = []
    for obj in bp_list:
        area = obj.w * obj.h
        if min_area < area < max_area:
            ar = obj.w / obj.h
            if min_ar < ar < max_ar:
                filtered.append(obj)
    return filtered


def _filter_couple(bp_list):
    """Keep at most 2 body parts, selecting the closest y-pair."""
    if len(bp_list) <= 2:
        return bp_list

    min_a, min_b = 0, 1
    min_diff = abs(bp_list[0].y - bp_list[1].y)

    for a in range(len(bp_list)):
        for b in range(len(bp_list)):
            if a != b:
                diff = abs(bp_list[a].y - bp_list[b].y)
                if diff < min_diff:
                    min_diff = diff
                    min_a, min_b = a, b

    return [bp_list[min_a], bp_list[min_b]]


def _detect_missing_problem(tits_list, aur_list):
    """Detect missing tit/areola combinations.

    Returns:
        Problem code (1-8), or -1 for unexpected counts.
    """
    t = len(tits_list)
    a = len(aur_list)

    table = {
        (0, 0): 1, (0, 1): 2, (0, 2): 3,
        (1, 0): 4, (1, 1): 5, (1, 2): 6,
        (2, 0): 7, (2, 1): 8,
    }
    return table.get((t, a), -1)


def _resolve_missing_problems(tits_list, aur_list, code):
    """Infer missing body parts from existing ones."""
    def _make(part_name, x, y, w, h):
        return BodyPart(
            part_name,
            int(x - w / 2), int(y - h / 2),
            int(x + w / 2), int(y + h / 2),
            x, y, w, h,
        )

    if code == 3:
        # No tits, 2 aurs -> generate both tits from aurs
        for aur in aur_list:
            factor = random.randint(2, 5)
            new_w = aur.w * factor
            tits_list.append(_make("tit", aur.x, aur.y, new_w, new_w))

    elif code == 6:
        # 1 tit, 2 aurs -> generate missing tit
        d1 = abs(tits_list[0].x - aur_list[0].x)
        d2 = abs(tits_list[0].x - aur_list[1].x)
        src = aur_list[0] if d1 > d2 else aur_list[1]
        w = tits_list[0].w
        tits_list.append(_make("tit", src.x, src.y, w, w))

    elif code == 7:
        # 2 tits, 0 aurs -> generate both aurs from tits
        for tit in tits_list:
            new_w = tit.w * random.uniform(0.03, 0.1)
            aur_list.append(_make("aur", tit.x, tit.y, new_w, new_w))

    elif code == 8:
        # 2 tits, 1 aur -> generate missing aur
        d1 = abs(aur_list[0].x - tits_list[0].x)
        d2 = abs(aur_list[0].x - tits_list[1].x)
        src = tits_list[0] if d1 > d2 else tits_list[1]
        w = aur_list[0].w
        aur_list.append(_make("aur", src.x, src.y, w, w))


def _infer_nip(aur_list):
    """Infer nipple positions from areola locations."""
    nip_list = []
    for aur in aur_list:
        nip_dim = int(5 + aur.w * random.uniform(0.03, 0.09))
        nip_list.append(
            BodyPart(
                "nip",
                int(aur.x - nip_dim / 2), int(aur.y - nip_dim / 2),
                int(aur.x + nip_dim / 2), int(aur.y + nip_dim / 2),
                aur.x, aur.y, nip_dim, nip_dim,
            )
        )
    return nip_list


def _infer_hair(vag_list):
    """Infer pubic hair region from vaginal detections."""
    hair_list = []
    if random.uniform(0.0, 1.0) > 0.3:
        for vag in vag_list:
            hair_w = vag.w * random.uniform(0.4, 1.5)
            hair_h = vag.h * random.uniform(0.4, 1.5)
            x = vag.x
            y = vag.y - hair_h / 2 - vag.h / 2
            hair_list.append(
                BodyPart(
                    "hair",
                    int(x - hair_w / 2), int(y - hair_h / 2),
                    int(x + hair_w / 2), int(y + hair_h / 2),
                    x, y, hair_w, hair_h,
                )
            )
    return hair_list
