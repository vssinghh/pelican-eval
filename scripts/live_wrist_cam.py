"""Live OpenCV preview window for tuning the manual focus ring on Camera 2 (wrist camera)."""

from __future__ import annotations

import cv2


def main() -> None:
    cap = cv2.VideoCapture(2)
    win_name = "SO-ARM101 Wrist Camera (Camera 2) Focus Tuner - Press Q to exit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    peak_focus = 1.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute focus sharpness inside the central/jaw region of interest
        h, w = gray.shape
        roi = gray[int(h * 0.25) : int(h * 0.90), int(w * 0.25) : int(w * 0.80)]
        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        if sharpness > peak_focus:
            peak_focus = sharpness

        display = frame.copy()
        cv2.rectangle(
            display,
            (int(w * 0.25), int(h * 0.25)),
            (int(w * 0.80), int(h * 0.90)),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display,
            f"Focus Sharpness: {sharpness:.1f} (Peak: {peak_focus:.1f}) | Twist lens ring to maximize | Press Q to close",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        cv2.imshow(win_name, display)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
