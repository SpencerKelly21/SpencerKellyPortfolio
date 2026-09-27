"""
HSV Colour Tuner  —  companion to air_hockey_robot.py
=====================================================
Adjust the sliders until your object appears SOLID WHITE in the mask
and everything else is BLACK.

End-Effector


Controls
--------
  q  — quit and print final HSV bounds
  s  — save bounds to hsv_values.py  (can be imported directly)
  r  — reset all HSV sliders to full range
"""

import cv2
import numpy as np
from picamera2 import Picamera2


def nothing(_):
    pass


MORPH_ITERATIONS = 2

# Default boundary values — kept in sync with air_hockey_robot.py
DEFAULT_TOP_WALL  = 55
DEFAULT_BOT_WALL  = 415
DEFAULT_LEFT_HBOT = 410
DEFAULT_MARGIN    = 85


def build_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Applies the same morphology pipeline as the main robot script."""
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask,  None, iterations=MORPH_ITERATIONS)
    mask = cv2.dilate(mask, None, iterations=MORPH_ITERATIONS)
    return mask


def get_hsv_values(win: str) -> tuple[np.ndarray, np.ndarray]:
    h_min = cv2.getTrackbarPos("H Min", win)
    s_min = cv2.getTrackbarPos("S Min", win)
    v_min = cv2.getTrackbarPos("V Min", win)
    h_max = cv2.getTrackbarPos("H Max", win)
    s_max = cv2.getTrackbarPos("S Max", win)
    v_max = cv2.getTrackbarPos("V Max", win)
    return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])


def get_bounds(win: str) -> dict:
    """Read boundary sliders and return the same 'b' dict the main robot uses."""
    top    = cv2.getTrackbarPos("Top Wall",   win)
    bot    = cv2.getTrackbarPos("Bot Wall",   win)
    mid    = cv2.getTrackbarPos("Mid Line",   win)
    margin = cv2.getTrackbarPos("Arm Margin", win)

    # Same safety clamps as the main script
    top    = max(0,        min(top, 230))
    bot    = max(top + 10, min(bot, 479))
    mid    = max(0,        min(mid, 630))
    margin = max(0,        min(margin, (bot - top) // 2 - 5))

    return {
        "top":      top,
        "bot":      bot,
        "mid":      mid,
        "top_hbot": top + margin,
        "bot_hbot": bot - margin,
    }


def draw_arena(frame: np.ndarray, b: dict) -> None:
    """Draws the same arena overlay lines as air_hockey_robot.py."""
    w    = frame.shape[1]
    gray = (200, 200, 200)
    cyan = (255, 220, 0)

    cv2.line(frame, (0,        b["top"]),      (w,  b["top"]),      gray, 1)  # top wall
    cv2.line(frame, (0,        b["bot"]),      (w,  b["bot"]),      gray, 1)  # bot wall
    cv2.line(frame, (b["mid"], 0),             (b["mid"], 480),     gray, 1)  # mid-court
    cv2.line(frame, (b["mid"], b["top_hbot"]), (w,  b["top_hbot"]), cyan, 1)  # safe zone top
    cv2.line(frame, (b["mid"], b["bot_hbot"]), (w,  b["bot_hbot"]), cyan, 1)  # safe zone bot


def save_values(lower: np.ndarray, upper: np.ndarray, filename: str = "hsv_values.py"):
    with open(filename, "w") as f:
        f.write("import numpy as np\n\n")
        f.write(f"LOWER_BOUND = np.array([{lower[0]}, {lower[1]}, {lower[2]}])\n")
        f.write(f"UPPER_BOUND = np.array([{upper[0]}, {upper[1]}, {upper[2]}])\n")
    print(f"[Saved] HSV bounds written to '{filename}'")


def print_values(lower: np.ndarray, upper: np.ndarray):
    print("\n--- COPY THESE INTO YOUR MAIN CODE ---")
    print(f"LOWER_BOUND = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
    print(f"UPPER_BOUND = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
    print("--------------------------------------\n")


def main():
    # --- Camera ---
    cam = Picamera2()
    config = cam.create_preview_configuration({"size": (640, 480), "format": "RGB888"})
    cam.configure(config)
    cam.start()

    # --- HSV trackbar window ---
    HSV_WIN = "HSV Tuner"
    cv2.namedWindow(HSV_WIN)
    cv2.createTrackbar("H Min", HSV_WIN,   0, 179, nothing)
    cv2.createTrackbar("S Min", HSV_WIN,   0, 255, nothing)
    cv2.createTrackbar("V Min", HSV_WIN,   0, 255, nothing)
    cv2.createTrackbar("H Max", HSV_WIN, 179, 179, nothing)
    cv2.createTrackbar("S Max", HSV_WIN, 255, 255, nothing)
    cv2.createTrackbar("V Max", HSV_WIN, 255, 255, nothing)

    # --- Boundary trackbar window (identical to main robot) ---
    BOUND_WIN = "Boundaries"
    cv2.namedWindow(BOUND_WIN)
    cv2.resizeWindow(BOUND_WIN, 500, 220)
    cv2.createTrackbar("Top Wall",   BOUND_WIN, DEFAULT_TOP_WALL,  240, nothing)
    cv2.createTrackbar("Bot Wall",   BOUND_WIN, DEFAULT_BOT_WALL,  480, nothing)
    cv2.createTrackbar("Mid Line",   BOUND_WIN, DEFAULT_LEFT_HBOT, 640, nothing)
    cv2.createTrackbar("Arm Margin", BOUND_WIN, DEFAULT_MARGIN,    150, nothing)

    print(__doc__)

    while True:
        frame = cam.capture_array()

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower, upper = get_hsv_values(HSV_WIN)
        b            = get_bounds(BOUND_WIN)
        mask         = build_mask(hsv, lower, upper)

        # --- Contour overlay ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = frame.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(largest)
            M       = cv2.moments(largest)
            cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(overlay, f"Area: {int(area)}px²  Center: ({cx},{cy})",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # --- Arena lines drawn on top of contour overlay ---
        draw_arena(overlay, b)

        # --- HUD ---
        hud = f"H[{lower[0]}-{upper[0]}]  S[{lower[1]}-{upper[1]}]  V[{lower[2]}-{upper[2]}]"
        cv2.putText(overlay, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        if lower[0] <= 10 or upper[0] >= 165:
            cv2.putText(overlay, "TIP: Red wraps around HSV — also tune 165-179 range",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1)

        # --- Side-by-side: camera+overlay | mask ---
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([overlay, mask_bgr])
        cv2.imshow("Camera | Mask (object = white)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print_values(lower, upper)
            break
        elif key == ord('s'):
            print_values(lower, upper)
            save_values(lower, upper)
        elif key == ord('r'):
            cv2.setTrackbarPos("H Min", HSV_WIN,   0)
            cv2.setTrackbarPos("S Min", HSV_WIN,   0)
            cv2.setTrackbarPos("V Min", HSV_WIN,   0)
            cv2.setTrackbarPos("H Max", HSV_WIN, 179)
            cv2.setTrackbarPos("S Max", HSV_WIN, 255)
            cv2.setTrackbarPos("V Max", HSV_WIN, 255)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
