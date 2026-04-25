"""
Pipeline orchestrator for the NeoNude transformation.

The pipeline runs 6 alternating OpenCV + GAN phases:

    Phase 0: dress -> correct    [OpenCV]  Color correction
    Phase 1: correct -> mask     [GAN]     Clothing mask generation
    Phase 2: mask -> maskref     [OpenCV]  Mask refinement
    Phase 3: maskref -> maskdet  [GAN]     Anatomical detail detection
    Phase 4: maskdet -> maskfin  [OpenCV]  Mask finalization
    Phase 5: maskfin -> nude     [GAN]     Final image generation
"""

import cv2

from .config import Options
from .model import DataLoader, DeepModel, tensor2im
from .transforms import correct_color, create_maskref, create_maskfin

PHASES = [
    "dress_to_correct",
    "correct_to_mask",
    "mask_to_maskref",
    "maskref_to_maskdet",
    "maskdet_to_maskfin",
    "maskfin_to_nude",
]

GAN_PHASES = {"correct_to_mask", "maskref_to_maskdet", "maskfin_to_nude"}

TARGET_SIZE = (512, 512)


def process(cv_img):
    """Run the full transformation pipeline on an input image.

    Automatically resizes input to 512x512 for processing and restores
    the original dimensions on the output.

    Args:
        cv_img: Input BGR image (OpenCV / numpy array), any size.

    Returns:
        Transformed BGR image at original dimensions, or None if failed.
    """
    original_size = (cv_img.shape[1], cv_img.shape[0])

    # Resize to 512x512 for the pipeline
    if original_size != (TARGET_SIZE[1], TARGET_SIZE[0]):
        dress = cv2.resize(cv_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    else:
        dress = cv_img

    correct = None
    mask = None
    maskref = None
    maskfin = None
    maskdet = None
    nude = None

    for phase in PHASES:
        print(f"[Pipeline] Phase: {phase}")

        if phase in GAN_PHASES:
            opt = Options()
            opt.update_for_phase(phase)

            # Select input image for this phase
            phase_inputs = {
                "correct_to_mask": correct,
                "maskref_to_maskdet": maskref,
                "maskfin_to_nude": maskfin,
            }

            data_loader = DataLoader(opt, phase_inputs[phase])
            dataset = data_loader.load_data()

            model = DeepModel()
            model.initialize(opt)

            for _, data in enumerate(dataset):
                generated = model.inference(data["label"], data["inst"])
                im = tensor2im(generated.data[0])
                result = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                if phase == "correct_to_mask":
                    mask = result
                elif phase == "maskref_to_maskdet":
                    maskdet = result
                elif phase == "maskfin_to_nude":
                    nude = result

        elif phase == "dress_to_correct":
            correct = correct_color(dress)

        elif phase == "mask_to_maskref":
            maskref = create_maskref(mask, correct)

        elif phase == "maskdet_to_maskfin":
            maskfin = create_maskfin(maskref, maskdet)

    # Restore original dimensions
    if nude is not None and original_size != (TARGET_SIZE[1], TARGET_SIZE[0]):
        nude = cv2.resize(nude, original_size, interpolation=cv2.INTER_LINEAR)

    return nude
