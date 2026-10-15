"""Minimal get -> match -> update demonstration; input handling stays outside the SDK."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from insightface.app import PersonAnalysis
from insightface.utils import face_align


def read_image(path):
    image = cv2.imdecode(np.fromfile(Path(path).expanduser(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode the input image')
    return image


def draw(frame, matches):
    output = frame.copy()
    for match in matches:
        person = match.observation
        box = person.body_bbox if person.body_bbox is not None else person.face.bbox
        x1, y1, x2, y2 = map(int, box)
        color = (60, 190, 70) if match.person_id is not None else (0, 180, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = str(match.person_id) if match.person_id is not None else 'unmatched'
        if match.matched_by:
            label += ' (%s %.2f)' % (match.matched_by, match.similarity)
        # OpenCV text is ASCII-oriented; the desktop GUI supports Unicode labels.
        cv2.putText(output, label.encode('ascii', errors='replace').decode(),
                    (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    scale = min(1., 1280 / max(output.shape[:2]))
    return cv2.resize(output, None, fx=scale, fy=scale) if scale < 1 else output


def save_aligned_faces(frame, observations, output_dir, frame_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError('Aligned output path is not a directory: %s' % output_dir)
    saved = []
    for face_index, person in enumerate(observations):
        face = person.face
        if face is None or face.embedding is None:
            continue
        aligned = face_align.norm_crop(frame, landmark=face.kps, image_size=112)
        output_path = output_dir / ('frame_%06d_face_%03d.bmp' % (frame_index, face_index))
        encoded, data = cv2.imencode('.bmp', aligned)
        if not encoded:
            raise RuntimeError('Cannot encode aligned face as BMP: %s' % output_path)
        try:
            data.tofile(output_path)
        except OSError as error:
            raise OSError('Cannot write aligned face: %s' % output_path) from error
        saved.append(output_path)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image')
    source.add_argument('--video')
    source.add_argument('--camera', type=int)
    source.add_argument('--rtsp', help='RTSP URL; no automatic reconnect in this example')
    parser.add_argument('--model', choices=('cheetah_s', 'cheetah_l'), default='cheetah_s')
    parser.add_argument('--root', default='~/.insightface')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--reference', action='append', default=[], metavar='NAME=PHOTO')
    parser.add_argument('--no-update', action='store_true')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0, help='0 reads until EOF or stop')
    parser.add_argument('--aligned-output-dir', type=Path,
                        help='save each 112x112 face crop used for an embedding as BMP')
    args = parser.parse_args()
    if args.max_frames < 0 or args.camera is not None and args.camera < 0:
        parser.error('frame limit and camera index must be nonnegative')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu else ['CPUExecutionProvider']
    capture = None
    try:
        with PersonAnalysis(name=args.model, root=args.root, providers=providers) as app:
            app.prepare()
            for reference in args.reference:
                name, separator, path = reference.partition('=')
                if not separator or not name or not path:
                    parser.error('--reference requires NAME=PHOTO')
                registration = app.register(name, path)
                if not registration.accepted:
                    raise ValueError('No reference accepted for %s: %s' % (name, registration.rejected))
            if not args.image:
                value = args.camera if args.camera is not None else args.video or args.rtsp
                capture = cv2.VideoCapture(value)
                if not capture.isOpened():
                    raise RuntimeError('Cannot open the input source')
            index = 0
            while True:
                if args.image:
                    frame = read_image(args.image)
                else:
                    ok, frame = capture.read()
                    if not ok:
                        if index == 0:
                            raise RuntimeError('The input opened but did not return a readable frame')
                        break
                observations = app.get(frame)
                if args.aligned_output_dir is not None:
                    save_aligned_faces(frame, observations, args.aligned_output_dir.expanduser(), index)
                matches = app.match(observations)
                changes = None if args.no_update else asdict(app.update(matches))
                print(json.dumps({'frame': index, 'matches': [
                    {'person_id': m.person_id, 'matched_by': m.matched_by, 'similarity': m.similarity}
                    for m in matches], 'reference_changes': changes}, ensure_ascii=False))
                index += 1
                if args.display:
                    cv2.imshow('PersonAnalysis demo', draw(frame, matches))
                    if cv2.waitKey(0 if args.image else 1) & 0xff in (27, ord('q')):
                        break
                if args.image or args.max_frames and index >= args.max_frames:
                    break
    finally:
        if capture is not None:
            capture.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
