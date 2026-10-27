"""Showcase InsightFace's optional RGB face-liveness addon on one image,
together with gender and age estimation.

Usage:
    python examples/liveness_demo.py input.jpg
    python examples/liveness_demo.py input.jpg --mode observe --threshold 0.9 --output out.jpg

The liveness score is the model's estimated probability that a detected face is
live, not an identity match or a guarantee against spoofing. Calibrate the
threshold for the model and execution provider used in your application.
"""
import argparse
import logging

import cv2
from insightface.app import FaceAnalysis


def main():
    """Parse options, run face detection and liveness, and save an annotated image.

    The command-line options control the input image, liveness policy, decision
    threshold, output image path, and execution device:

    * ``image``: input image path. The image must be readable by OpenCV.
    * ``--mode``: ``normal`` skips recognition for faces not classified as live;
      ``observe`` reports liveness but does not use it to block recognition.
    * ``--threshold``: live-score cutoff from 0 to 1; scores equal to the cutoff
      count as live. The addon's default is 0.8.
    * ``--output``: destination for the image annotated with face boxes and labels.
    * ``--ctx``: execution provider context; use -1 for CPU or a GPU id for GPU.
    * ``--debug``: show InsightFace debug logs, such as which model files were
      found and their input shapes and normalization values.
    """
    parser = argparse.ArgumentParser(description="InsightFace liveness demo")
    parser.add_argument(
        "image",
        help="Input image path (for example, friends.jpg)",
    )
    parser.add_argument("--mode", choices=["normal", "observe"], default="normal",
                        help="normal blocks recognition for non-live/rejected faces; observe does not")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum live score in [0, 1]; equality passes (default: 0.8)",
    )
    parser.add_argument(
        "--output",
        default="liveness_result.jpg",
        help="Path for the annotated output image (default: liveness_result.jpg)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="GPU id for inference, or -1 to use the CPU (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print InsightFace debug log messages",
    )
    args = parser.parse_args()

    # InsightFace logs through the standard logging module under the
    # "insightface" logger hierarchy but does not attach handlers itself.
    # basicConfig adds a console handler; only InsightFace is raised to DEBUG
    # so other libraries stay quiet.
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    if args.debug:
        logging.getLogger("insightface").setLevel(logging.DEBUG)

    # OpenCV reads color images in BGR order, which is the format expected by
    # FaceAnalysis. Fail explicitly rather than passing an invalid image onward.
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    # The liveness addon is optional and must be selected explicitly. In normal
    # mode it gates recognition; observe mode is useful for inspecting scores
    # while still computing recognition embeddings for every detected face.
    app = FaceAnalysis(
        name="buffalo_l",
        addons=["liveness"],
        liveness_mode=args.mode,
        liveness_threshold=args.threshold,
    )
    # prepare() initializes the selected models and chooses CPU/GPU execution.
    # det_size controls the detector's input resolution.
    app.prepare(ctx_id=args.ctx, det_size=(640, 640))

    # get() detects faces, evaluates liveness, and (subject to mode) runs the
    # other selected tasks. The face list includes faces even when liveness
    # rejects recognition, so their locations can still be displayed.
    faces = app.get(image)
    print(f"Detected {len(faces)} face(s) in {args.image} (mode={args.mode}, threshold={args.threshold})")

    for i, face in enumerate(faces):
        result = face.liveness
        # A missing result means liveness did not run. An input_rejected result
        # means the aligned face crop lacked enough source-image area; it is not
        # a "fake" classification. For status "ok", is_live applies threshold.
        if result is None:
            label, color = "N/A", (128, 128, 128)
        elif result.status == "input_rejected":
            label, color = "REJECTED", (0, 165, 255)
        elif result.is_live:
            label, color = f"LIVE {result.live_score:.2f}", (0, 200, 0)
        else:
            label, color = f"FAKE {result.live_score:.2f}", (0, 0, 255)

        # buffalo_l includes a gender/age model (genderage.onnx). It runs for
        # every detected face, because normal liveness mode only skips the
        # recognition task. face.sex is "M"/"F" and face.age is an integer
        # estimate in years; both are None if the model was not loaded.
        if face.sex is not None and face.age is not None:
            demographics = f"{face.sex}, {face.age}"
        else:
            demographics = "gender/age N/A"

        # Recognition can be absent, for example when normal mode blocks it for
        # a fake or rejected face. This is separate from the liveness result.
        has_embedding = face.embedding is not None
        print(f"  face {i}: {label}  {demographics}  (embedding computed: {has_embedding})")
        if result is not None and result.status == "input_rejected":
            print(f"    reason: {result.reason}")

        # Draw the detector's bounding box with the liveness outcome above it
        # and gender/age below it. OpenCV color tuples are BGR: green, red, orange.
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, demographics, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write the visualization so the results can be reviewed after inference.
    cv2.imwrite(args.output, image)
    print(f"Annotated image saved to {args.output}")


if __name__ == "__main__":
    main()
