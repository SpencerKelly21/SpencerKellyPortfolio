"""
air_hockey_complete_J7.py

Controls: 
  q     quit 
  space freeze/unfreeze motors 
  o     toggle visual overlay
  r     start/stop recording telemetry to CSV
"""

import cv2
import time
import numpy as np
import csv
from datetime import datetime

try:
    from dynamixel_sdk import *
    DYNAMIXEL_AVAILABLE = True
except ImportError:
    print("[WARN] dynamixel_sdk not found. Run:  pip install dynamixel-sdk")
    print("[WARN] VISION-ONLY mode: tracking + overlay only, the loop cannot")
    print("       converge because there is no arm to move.")
    DYNAMIXEL_AVAILABLE = False

# Hardware config
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
PROFILE_VELOCITY = 230            # must be >= servo MAX_SPEED (230 tracks ~13.7 cm/s)

DXL_ID_2, DXL_ID_3 = 2, 3
MOTOR_A, MOTOR_B  = DXL_ID_2, DXL_ID_3

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_LED_TOGGLE = 65
EXTENDED_POSITION_MODE = 4
DEG_PER_TICK = 360.0 / 4096


# Kinmatics config
DEG_PER_CM_Y = 23.0
DEG_PER_CM_X = 23.0
SIGN_X = +1
SIGN_Y = +1
X_MIN_CM, X_MAX_CM = -9.6,  9.6
Y_MIN_CM, Y_MAX_CM = -11.2, 8.75

# Defending settings
DEFENDER_X = 455      # pixel x of the defender line
GOAL_TOP, GOAL_BOT = 120, 340   # clamp the predicted intercept to the goal opening

# the defender line
DEFENDER_Y_CM = -9.5

# Vision Config
TOP_WALL, BOT_WALL = 55, 415

LOWER_PUCK = np.array([0, 160, 124])
UPPER_PUCK = np.array([179, 255, 255])
LOWER_PADDLE = np.array([40, 50, 17])
UPPER_PADDLE = np.array([105, 255, 255])

# Blue END-EFFECTOR marker — PASTE YOUR TUNED VALUES from visual_servo_tuner.py
LOWER_EE = np.array([109, 176, 0])
UPPER_EE = np.array([132, 255, 255])

# Servo Config (tuning) 
KP = 0.325    # cm/s per pixel of error
KI = 0.004    # leave ~0 in velocity form (P already removes steady-state error)
KD = 0.055   # damping

SIGN_GAIN_X = -1   # screen-vertical -> robot X. Grounded in v4 calibration; rarely wrong.
SIGN_GAIN_Y = -1   # screen-horizontal -> robot Y. VERIFY in the tuner ('y' to flip).

MAX_SPEED_CM_S = 600.0
INTEGRAL_LIMIT = 200.0
DEADBAND_PX    = 5
DT_CAP_S       = 0.10

SERVO_FORWARD_AXIS = False    # True = 2-DOF (x-axis moves)

# Light smoothing on the predicted-intercept TARGET pixel (prediction is noisy).
TARGET_EMA_ALPHA = 0.65


# Target smoothing
class EMASetpointFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.current = None

    def reset(self):
        self.current = None

    def update(self, raw):
        if self.current is None:
            self.current = float(raw)
        else:
            self.current = self.alpha * raw + (1.0 - self.alpha) * self.current
        return self.current



# Motor Controller
class MotorController:
    def __init__(self):
        self.enabled = False
        self.HOME_A = self.HOME_B = None
        if not DYNAMIXEL_AVAILABLE:
            return

        self.portHandler   = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        if not self.portHandler.openPort():
            print(f"[ERROR] Could not open port {DEVICENAME}.");  return
        if not self.portHandler.setBaudRate(BAUDRATE):
            print("[ERROR] Could not set baud rate.");            return

        for mid in (MOTOR_A, MOTOR_B):
            self.packetHandler.write1ByteTxRx(self.portHandler, mid, ADDR_TORQUE_ENABLE, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, mid, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
            self.packetHandler.write4ByteTxRx(self.portHandler, mid, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)
            self.packetHandler.write1ByteTxRx(self.portHandler, mid, ADDR_TORQUE_ENABLE, 1)
            self.packetHandler.write1ByteTxRx(self.portHandler, mid, ADDR_LED_TOGGLE, 1)

        self.enabled = True
        print("[OK] Motors initialised.")
        self.set_home()
        print(f"[INIT] Pre-positioning to defender Y: {DEFENDER_Y_CM:.2f} cm")
        self.move_to_xy(0.0, DEFENDER_Y_CM)

    def read_deg(self, motor_id):
        raw, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, ADDR_PRESENT_POSITION)
        if raw > 2147483647:
            raw -= 4294967296
        return raw * DEG_PER_TICK

    def move_to_deg(self, motor_id, angle_deg):
        ticks = int(angle_deg / DEG_PER_TICK)
        self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, ADDR_GOAL_POSITION, ticks)

    def set_home(self):
        self.HOME_A = self.read_deg(MOTOR_A)
        self.HOME_B = self.read_deg(MOTOR_B)
        print(f"[HOME] A={self.HOME_A:.1f}°  B={self.HOME_B:.1f}°  -> (x=0, y=0)")

    def move_to_xy(self, x_cm, y_cm):
        if not self.enabled:
            return
        x_cmd = min(max(x_cm, X_MIN_CM), X_MAX_CM)
        y_cmd = min(max(y_cm, Y_MIN_CM), Y_MAX_CM)
        x_deg = SIGN_X * x_cmd * DEG_PER_CM_X
        y_deg = SIGN_Y * y_cmd * DEG_PER_CM_Y
        self.move_to_deg(MOTOR_A, self.HOME_A + x_deg + y_deg)
        self.move_to_deg(MOTOR_B, self.HOME_B + x_deg - y_deg)

    def read_xy(self):
        if not self.enabled:
            return 0.0, 0.0
        da = self.read_deg(MOTOR_A) - self.HOME_A
        db = self.read_deg(MOTOR_B) - self.HOME_B
        x_deg = (da + db) / 2.0
        y_deg = (da - db) / 2.0
        return x_deg / (SIGN_X * DEG_PER_CM_X), y_deg / (SIGN_Y * DEG_PER_CM_Y)

    def shutdown(self):
        if self.enabled:
            for mid in (MOTOR_A, MOTOR_B):
                self.packetHandler.write1ByteTxRx(self.portHandler, mid, ADDR_TORQUE_ENABLE, 0)
                self.packetHandler.write1ByteTxRx(self.portHandler, mid, ADDR_LED_TOGGLE, 0)
            self.portHandler.closePort()


# Task-space controller
class VisualServoController:

    def __init__(self, motors, enable_y=True):
        self.motors  = motors
        self.Kp, self.Ki, self.Kd = KP, KI, KD
        self.sign_x, self.sign_y  = SIGN_GAIN_X, SIGN_GAIN_Y
        self.enable_y = enable_y
        self.frozen   = False
        self.reseed(self.motors.read_xy())

    def reseed(self, xy):
        """Start the integrator at the current arm position (bumpless)."""
        self.sp_x, self.sp_y = float(xy[0]), float(xy[1])
        self.integ_x = self.integ_y = 0.0
        self.have_prev = False
        self.e_x_prev = self.e_y_prev = 0.0

    def step(self, measured_px, target_px, dt):
        """One tick. Returns (setpoint_xy, err_vec_px, err_mag_px, status)."""
        dt = min(max(dt, 1e-3), DT_CAP_S)
        sp = (self.sp_x, self.sp_y)

        if target_px is None:
            return sp, None, None, "NO TARGET"
        if measured_px is None:
            self.have_prev = False
            return sp, None, None, "EE LOST"

        err_sx = target_px[0] - measured_px[0]      # screen horizontal -> robot Y
        err_sy = target_px[1] - measured_px[1]      # screen vertical   -> robot X
        e_x, e_y = float(err_sy), float(err_sx)
        err_mag = float(np.hypot(err_sx, err_sy)) if self.enable_y else abs(float(err_sy))

        if self.frozen:
            self.have_prev = False
            return sp, (err_sx, err_sy), err_mag, "FROZEN"

        if err_mag < DEADBAND_PX:
            self.have_prev = False
            self.motors.move_to_xy(self.sp_x, self.sp_y)
            return (self.sp_x, self.sp_y), (err_sx, err_sy), err_mag, "CONVERGED"

        # X axis (always on)
        self.integ_x = float(np.clip(self.integ_x + e_x * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
        d_x = (e_x - self.e_x_prev) / dt if self.have_prev else 0.0
        v_x = self.sign_x * (self.Kp * e_x + self.Ki * self.integ_x + self.Kd * d_x)
        v_x = float(np.clip(v_x, -MAX_SPEED_CM_S, MAX_SPEED_CM_S))
        self.sp_x = float(np.clip(self.sp_x + v_x * dt, X_MIN_CM, X_MAX_CM))

        # Y axis (optional)
        if self.enable_y:
            self.integ_y = float(np.clip(self.integ_y + e_y * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
            d_y = (e_y - self.e_y_prev) / dt if self.have_prev else 0.0
            v_y = self.sign_y * (self.Kp * e_y + self.Ki * self.integ_y + self.Kd * d_y)
            v_y = float(np.clip(v_y, -MAX_SPEED_CM_S, MAX_SPEED_CM_S))
            self.sp_y = float(np.clip(self.sp_y + v_y * dt, Y_MIN_CM, Y_MAX_CM))

        self.have_prev = True
        self.e_x_prev, self.e_y_prev = e_x, e_y
        self.motors.move_to_xy(self.sp_x, self.sp_y)
        return (self.sp_x, self.sp_y), (err_sx, err_sy), err_mag, "MOVING"


# Bounce predicition
def predict_intercept_with_bounces(puck_x, puck_y, paddle_x, paddle_y,
                                   target_x, top_wall, bot_wall):
    dx = puck_x - paddle_x
    dy = puck_y - paddle_y
    if dx == 0:
        return None, []

    segments = []
    cx, cy   = float(puck_x), float(puck_y)
    cdx, cdy = float(dx), float(dy)

    for _ in range(10):
        if cdx == 0:
            break
        t_to_target = (target_x - cx) / cdx
        if t_to_target <= 0:
            return int(cy), segments

        if cdy < 0:
            t_wall, wall_y = (top_wall - cy) / cdy, top_wall
        elif cdy > 0:
            t_wall, wall_y = (bot_wall - cy) / cdy, bot_wall
        else:
            t_wall, wall_y = float('inf'), cy

        if t_to_target <= t_wall:
            fy = cy + cdy * t_to_target
            segments.append(((int(cx), int(cy)), (int(target_x), int(fy))))
            return int(np.clip(fy, top_wall, bot_wall)), segments
        else:
            bx = cx + cdx * t_wall
            segments.append(((int(cx), int(cy)), (int(bx), int(wall_y))))
            cx, cy = bx, wall_y
            cdy = -cdy

    return int(cy), segments


def compute_defender_target(puck, paddle, target_filter):
    """
    Turn the puck/paddle detection into a servo target pixel on the defender line.
    Returns (target_px | None, predicted_y | None, clamped_y | None, segments).
    Module-level so it can be unit-tested without a camera.
    """
    if not (puck and paddle):
        return None, None, None, []
    px, py = puck
    dx, dy = paddle
    predicted_y, segments = predict_intercept_with_bounces(
        px, py, dx, dy, DEFENDER_X, TOP_WALL, BOT_WALL)
    if predicted_y is None:
        return None, None, None, segments
    clamped_y = int(np.clip(predicted_y, GOAL_TOP, GOAL_BOT))
    smooth_y  = int(round(target_filter.update(clamped_y)))
    return (DEFENDER_X, smooth_y), predicted_y, clamped_y, segments




# Main Robot opertaion
# -------------------------
class AirHockeyRobot:
    def __init__(self):
        self.motors = MotorController()
        time.sleep(0.5)                     # let the pre-position move settle before seeding
        self.servo  = VisualServoController(self.motors, enable_y=SERVO_FORWARD_AXIS)
        if not SERVO_FORWARD_AXIS:
            self.servo.sp_y = DEFENDER_Y_CM
        self.target_filter = EMASetpointFilter(TARGET_EMA_ALPHA)
        self.tracking = False

        # UI and Logging Flags
        self.show_overlay = True
        self.recording = False
        self.log_data = []
        self.log_start_time = 0.0

        cv2.startWindowThread()
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration({"size": (640, 480), "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.set_controls({'FrameRate': 80})
        self.cam.start()

        self.prev_time = time.time()
        self.fps = 0

    def detect_color_object(self, frame, lower_hsv, upper_hsv):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask  = cv2.erode(mask, None, iterations=2)
        mask  = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None

    def update_fps(self, dt):
        """Compute FPS from the already-measured dt; does NOT re-advance prev_time."""
        self.fps = 1.0 / dt if dt > 1e-6 else self.fps

    def save_log_to_csv(self):
        """Saves telemetry data to a CSV file along with the current PID gains."""
        if not self.log_data:
            print("[LOG] No data to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"servo_data_Kp{KP}_Ki{KI}_Kd{KD}.csv"

        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Add PID parameters as a header comment
            writer.writerow([f"# Kp={KP}", f"Ki={KI}", f"Kd={KD}"])
            # Add column headers
            writer.writerow(["Time_s", "EE_X_px", "EE_Y_px", "Target_X_px", "Target_Y_px", "Err_X_px", "Err_Y_px", "Err_Mag_px"])
            writer.writerows(self.log_data)

        print(f"[LOG] Saved {len(self.log_data)} telemetry records to {filename}")
        self.log_data.clear()

    def run(self):
        try:
            while True:
                frame = self.cam.capture_array()
                now = time.time()
                dt = now - self.prev_time
                self.prev_time = now

                puck = self.detect_color_object(frame, LOWER_PUCK,   UPPER_PUCK)
                paddle = self.detect_color_object(frame, LOWER_PADDLE, UPPER_PADDLE)
                ee = self.detect_color_object(frame, LOWER_EE,     UPPER_EE)

                # prediction -> servo target on the defender line
                target, predicted_y, clamped_y, segments = compute_defender_target(
                    puck, paddle, self.target_filter)
                if target is None:
                    self.target_filter.reset()   # reseed cleanly when puck/paddle return

                # bumpless resume after a tracking gap
                tracking_now = (target is not None) and (ee is not None)
                if tracking_now and not self.tracking:
                    self.servo.reseed(self.motors.read_xy())
                    if not SERVO_FORWARD_AXIS:
                        self.servo.sp_y = DEFENDER_Y_CM
                self.tracking = tracking_now

                # one control tick
                (sp_x, sp_y), err_vec, err_mag, status = self.servo.step(ee, target, dt)

                # Data Logging
                if self.recording:
                    t_log = now - self.log_start_time
                    ee_x, ee_y = ee if ee else (None, None)
                    tgt_x, tgt_y = target if target else (None, None)
                    ex, ey = err_vec if err_vec else (None, None)
                    
                    self.log_data.append([
                        round(t_log, 4), 
                        ee_x, ee_y, 
                        tgt_x, tgt_y, 
                        ex, ey, 
                        round(err_mag, 2) if err_mag is not None else None
                    ])

                # stream print: error, desired (target), real EE position
                _ee_str  = f"({ee[0]}, {ee[1]})px"         if ee     else "(--,--)px"
                _tgt_str = f"({target[0]}, {target[1]})px" if target else "(--,--)px"
                _err_str = (f"({err_vec[0]:+.0f}, {err_vec[1]:+.0f})px  mag={err_mag:.1f}px"
                            if err_vec is not None else "(--,--)px  mag=--")
                print(f"[{status:<10}]  EE={_ee_str:<18}  TARGET={_tgt_str:<18}  ERR={_err_str}")

                # overlay
                if self.show_overlay:
                    for seg in segments:
                        cv2.line(frame, seg[0], seg[1], (255, 50, 50), 2)

                    if puck:
                        cv2.circle(frame, puck, 15, (0, 165, 255), 3)
                    if paddle:
                        cv2.circle(frame, paddle, 20, (0, 255, 0), 3)
                        if puck:
                            cv2.line(frame, paddle, puck, (200, 200, 200), 1)

                    if predicted_y is not None:
                        cv2.circle(frame, (DEFENDER_X, predicted_y), 8, (0, 255, 255), 1)   # raw
                    if clamped_y is not None:
                        cv2.circle(frame, (DEFENDER_X, clamped_y), 10, (0, 220, 255), -1)   # clamped

                    # target crosshair + measured end-effector + error vector
                    if target is not None:
                        cv2.drawMarker(frame, target, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                    if ee is not None:
                        cv2.circle(frame, ee, 10, (255, 100, 0), -1)        # blue EE (BGR -> orange-ish dot)
                        cv2.circle(frame, ee, 12, (255, 200, 100), 2)
                        if target is not None:
                            cv2.arrowedLine(frame, ee, target, (0, 220, 255), 1, tipLength=0.05)

                    # arena + defender + goal
                    cv2.line(frame, (0, TOP_WALL), (640, TOP_WALL), (255, 255, 255), 1)
                    cv2.line(frame, (0, BOT_WALL), (640, BOT_WALL), (255, 255, 255), 1)
                    cv2.line(frame, (DEFENDER_X, TOP_WALL), (DEFENDER_X, BOT_WALL), (100, 255, 100), 2)
                    cv2.line(frame, (0, GOAL_TOP), (640, GOAL_TOP), (0, 200, 255), 1)
                    cv2.line(frame, (0, GOAL_BOT), (640, GOAL_BOT), (0, 200, 255), 1)
                    cv2.line(frame, (DEFENDER_X - 15, GOAL_TOP), (DEFENDER_X + 15, GOAL_TOP), (0, 200, 255), 3)
                    cv2.line(frame, (DEFENDER_X - 15, GOAL_BOT), (DEFENDER_X + 15, GOAL_BOT), (0, 200, 255), 3)

                    # HUD
                    colour = {"MOVING": (0, 220, 255), "CONVERGED": (0, 255, 0),
                              "FROZEN": (0, 165, 255), "EE LOST": (0, 0, 255),
                              "NO TARGET": (160, 160, 160)}.get(status, (255, 255, 255))
                    cv2.putText(frame, status, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
                    cv2.putText(frame, f"err={err_mag:.0f}px" if err_mag is not None else "err= --",
                                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                    cv2.putText(frame, f"sp=({sp_x:+.2f},{sp_y:+.2f})cm  "
                                       f"{'2-DOF' if SERVO_FORWARD_AXIS else '1-DOF'}",
                                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    if self.recording:
                        cv2.putText(frame, "* RECORDING *", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    self.update_fps(dt)
                    cv2.putText(frame, f"FPS: {int(self.fps)}", (560, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, "q quit  space freeze  o overlay  r record",
                                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

                cv2.imshow("Air Hockey", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.motors.move_to_xy(0,0)
                    time.sleep(5)
                    break
                elif key == ord(' '):
                    self.servo.frozen = not self.servo.frozen
                    print(f"[CTRL] {'FROZEN' if self.servo.frozen else 'RUNNING'}")
                elif key == ord('o'):
                    self.show_overlay = not self.show_overlay
                    print(f"[UI] Overlay {'ON' if self.show_overlay else 'OFF'}")
                elif key == ord('r'):
                    if not self.recording:
                        self.recording = True
                        self.log_start_time = time.time()
                        self.log_data.clear()
                        print("[LOG] Recording telemetry started...")
                    else:
                        self.recording = False
                        self.save_log_to_csv()

        finally:
            self.cam.stop()
            self.motors.shutdown()
            cv2.destroyAllWindows()
            # Ensure data saves even if the program terminates unexpectedly or user hits 'q'
            if self.log_data:
                self.save_log_to_csv()


if __name__ == "__main__":
    robot = AirHockeyRobot()
    robot.run()



