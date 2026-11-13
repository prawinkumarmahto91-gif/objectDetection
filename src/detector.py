import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import Counter


class ObjectDetector:

    def __init__(self, model_path):

        self.model = YOLO(str(model_path))

    def detect_image(self, image, confidence=0.25):

        results = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        annotated_image = result.plot()

        class_ids = (
            result.boxes.cls.cpu().numpy().astype(int)
        )

        counts = Counter(
            self.model.names[class_id]
            for class_id in class_ids
        )

        return annotated_image, counts, len(result.boxes)

    def detect_video(
        self,
        input_path,
        output_path,
        confidence=0.25
    ):

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise ValueError("Cannot open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not out.isOpened():
            cap.release()
            raise ValueError("Cannot create output video.")

        frame_count = 0
        total_detections = 0

        try:

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                results = self.model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )

                result = results[0]

                annotated_frame = result.plot()

                out.write(annotated_frame)

                frame_count += 1

                total_detections += len(result.boxes)

        finally:

            cap.release()
            out.release()

        return frame_count, total_detections