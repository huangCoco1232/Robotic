import sys
import os
import warnings
import math
import time

warnings.filterwarnings("ignore", "torch.meshgrid")
sys.path.insert(0, "/media/pi/robo/v5lite/YOLOv5-Lite")
import torch
import cv2
import numpy as np
from picamera2 import Picamera2
import picar_4wd as fc
from torchvision.ops import nms

ROBOT_SPEED = 25.92
MAX_POWER = 10
IMG_W = 640
IMG_H = 480
CENTER_THRESHOLD = 50

KP = 1.0
KI = 0.05
KD = 0.2

AVOID_DISTANCE = 0.3
STOP_THRESHOLD = 0.85 * IMG_H

# Load model
ckpt = torch.load("/media/pi/robo/v5lite/YOLOv5-Lite/v5lite-e.pt", map_location="cpu")
model = ckpt["model"].float().eval()
names = model.names
for m in model.modules():
    if isinstance(m, torch.nn.Upsample):
        m.recompute_scale_factor = None

def preprocess(img):
    im = cv2.resize(img, (320, 320))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))[None, ...]
    return torch.from_numpy(im)

def postprocess(preds, img_shape, names, conf_thres=0.4, iou_thres=0.5):
    boxes, scores, classes = [], [], []
    for det in preds:
        conf = float(det[4])
        if conf < conf_thres:
            continue
        x, y, w, h = det[:4]
        x1 = (x - w/2) * img_shape[1] / 320
        y1 = (y - h/2) * img_shape[0] / 320
        x2 = (x + w/2) * img_shape[1] / 320
        y2 = (y + h/2) * img_shape[0] / 320
        cls_i = int(det[5:].argmax())
        prob = float(det[5:][cls_i])
        score = conf * prob
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls_i)
    if not boxes:
        return []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, iou_thres).cpu().numpy()
    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cls_i = classes[i]
        results.append((int(x1), int(y1), int(x2), int(y2), scores[i], names[cls_i]))
    return results

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        if abs(self.integral) > 30:
            self.integral = 30 * (1 if self.integral > 0 else -1)
        self.integral += error * dt
        if self.prev_error * error < 0:
            self.integral *= 0.5
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def calculate_steering_error(target_point, robot_center=(320, 480)):
    return target_point[0] - robot_center[0]

def main():
    bev_size, scale = 500, 100
    bev = np.ones((bev_size, bev_size, 3), dtype=np.uint8)*255
    u0 = IMG_W/2
    robot_px = (bev_size//2, bev_size-20)
    tri = np.array([
        (robot_px[0], robot_px[1]-20),
        (robot_px[0]-10, robot_px[1]+10),
        (robot_px[0]+10, robot_px[1]+10),
    ], np.int32)

    pid = PIDController(KP, KI, KD)

    camera = Picamera2()
    camera.preview_configuration.main.size = (IMG_W, IMG_H)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()

    out_dir = '/home/pi/robotic/output'
    os.makedirs(out_dir, exist_ok=True)
    fc.start_speed_thread()

    target_found = False
    target_midpoint = None
    prev_time = time.time()
    prev_detection_time = time.time()
    had_detection = False
    single_avoided = False

    no_cls = {
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','parking meter','bench','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','skis','snowboard'
        ,'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'wine glass','fork','knife','spoon','banana','apple','sandwich',
        'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','remote',
        'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
        'book','clock','scissors','teddy bear','hair drier','toothbrush'
    }

    try:
        while True:
            frame = camera.capture_array()
            now = time.time()
            dt = now - prev_time
            prev_time = now

            bev[:] = 255
            cv2.fillPoly(bev, [tri], (0,0,255))

            inp = preprocess(frame)
            with torch.no_grad():
                preds = model(inp)[0].cpu().numpy()[0]
            boxes = postprocess(preds, frame.shape, names)

            relevant_objects = []
            for x1,y1,x2,y2,conf,cls in boxes:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"{cls}:{conf:.2f}", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
                if cls not in no_cls:
                    print(f'cls is:', cls)
                    cx, cy = (x1+x2)/2, y1
                    z = (650*0.135)/max(10, abs(y2-160))
                    x = (cx-u0)*z/650
                    px = int(robot_px[0] + x*scale)
                    py = int(robot_px[1] - z*scale)
                    cv2.circle(bev, (px,py),5,(0,128,0),-1)
                    cv2.putText(bev, f"{z:.2f}m",(px+5,py-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
                    relevant_objects.append((cx,cy,x,z,px,py))



            if len(relevant_objects) >= 2:
                single_avoided = False
                # print(f"current object is 2!!!!")
                relevant_objects.sort(key=lambda o: o[0])
                left, right = relevant_objects[0], relevant_objects[-1]
                mid_cx = (left[0]+right[0])/2
                mid_cy = (left[1]+right[1])/2
                mid_x = (left[2]+right[2])/2
                mid_z = (left[3]+right[3])/2
                mid_px = (left[4]+right[4])//2
                mid_py = (left[5]+right[5])//2

                cv2.line(frame, (int(left[0]),int(left[1])),
                         (int(right[0]),int(right[1])),(0,255,255),2)
                cv2.circle(frame,(int(mid_cx),int(mid_cy)),8,(0,0,255),-1)
                cv2.line(bev,(left[4],left[5]),(right[4],right[5]),(0,255,255),2)
                cv2.circle(bev,(mid_px,mid_py),7,(0,0,255),-1)
                cv2.line(bev, robot_px, (mid_px,mid_py),(255,0,0),2)

                target_midpoint = (mid_cx, mid_cy, mid_z)
                target_found = had_detection = True
                prev_detection_time = now

                cv2.putText(frame, f"Target: ({mid_x:.2f}m, {mid_z:.2f}m)",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)


            elif len(relevant_objects) == 1:
                obj = relevant_objects[0]
                cx, cy = obj[0], obj[1]
                z = obj[3]
                mirrored_cx = -cx if cx < IMG_W / 2 else 2 * IMG_W - cx
                mid_cx, mid_cy = (cx + mirrored_cx) / 2, cy
                side = "Left" if mid_cx < IMG_W / 2 else "Right"
                print(
                    f"[DEBUG] One tag: cx={cx:.1f}, z={z:.2f}m, mid_cx={mid_cx:.1f}, side={side}, single_avoided={single_avoided}")
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                end_x = int(np.clip(mirrored_cx, 0, IMG_W - 1))
                cv2.line(frame, (int(cx), int(cy)), (end_x, int(cy)), (0, 255, 255), 2)
                draw_x = int(np.clip(mid_cx, 0, IMG_W - 1))
                cv2.circle(frame, (draw_x, int(mid_cy)), 8, (0, 0, 255), -1)
                cv2.putText(frame, f"{side}, z={z:.2f}m", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if mid_cy > STOP_THRESHOLD:
                    print("[DEBUG] Decision: STOP (mid_cy > threshold)")
                    cv2.putText(frame, "STOP: target reached", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    fc.forward(10);
                    time.sleep(1);
                    fc.stop()
                elif z < AVOID_DISTANCE and not single_avoided:
                    print("[DEBUG] Decision: PIVOT (z < AVOID_DISTANCE)")
                    turn_power = 8
                    if side == "Left":
                        print("[DEBUG] Pivot RIGHT")
                        cv2.putText(frame, "PIVOT RIGHT", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        fc.left_front.set_power(turn_power)
                        fc.left_rear.set_power(turn_power)
                        fc.right_front.set_power(-turn_power)
                        fc.right_rear.set_power(-turn_power)
                    else:
                        print("[DEBUG] Pivot LEFT")
                        cv2.putText(frame, "PIVOT LEFT", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        fc.left_front.set_power(-turn_power)
                        fc.left_rear.set_power(-turn_power)
                        fc.right_front.set_power(turn_power)
                        fc.right_rear.set_power(turn_power)
                    single_avoided = True
                elif single_avoided:
                    print("[DEBUG] Decision: BYPASS & STRAIGHT")
                    cv2.putText(frame, "BYPASS & STRAIGHT", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    fc.forward(10)
                else:
                    err = calculate_steering_error((mid_cx, mid_cy))
                    cs = pid.compute(err, dt)
                    cs = np.clip(cs, -MAX_POWER, MAX_POWER)
                    base = 10
                    if abs(err) < CENTER_THRESHOLD:
                        lp = rp = base
                    else:
                        lp = int(base + cs)
                        rp = int(base - cs)
                    lp, rp = np.clip([lp, rp], -MAX_POWER, MAX_POWER)
                    print(f"[DEBUG] Decision: TRACK, err={err:.1f}, cs={cs:.1f}, L={lp}, R={rp}")
                    cv2.putText(frame, f"TRACK L:{lp} R:{rp}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    fc.left_front.set_power(lp)
                    fc.left_rear.set_power(lp)
                    fc.right_front.set_power(rp)
                    fc.right_rear.set_power(rp)
                cv2.putText(frame, "One tag (mirrored)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                continue


            elif had_detection and not single_avoided and (now - prev_detection_time) < 1.0:
                target_found = True
                cv2.putText(frame, "Continuing with last target",(10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
            else:
                target_found = False
                had_detection = False

            if single_avoided:
                # print("Bypassed obstacle")
                fc.forward(10)
                # time.sleep(0.5)
                # fc.stop()
                pid.reset()
                cv2.putText(frame, "Bypassed obstacle", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                continue

            elif target_found and target_midpoint:
                err = calculate_steering_error(target_midpoint)
                cs = pid.compute(err, dt)
                cs = max(min(cs,MAX_POWER),-MAX_POWER)
                if abs(err) < CENTER_THRESHOLD:
                    lp = rp = 10
                else:
                    lp = int(10 + cs)
                    rp = int(10 - cs)
                lp = max(min(MAX_POWER, lp), -MAX_POWER)
                rp = max(min(MAX_POWER, rp), -MAX_POWER)
                # print(f"left_power:", lp)
                # print(f"right_power:", rp)
                fc.left_front.set_power(lp)
                fc.left_rear.set_power(lp)
                fc.right_front.set_power(rp)
                fc.right_rear.set_power(rp)
                cv2.putText(frame, f"L:{lp} R:{rp}",(10,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                # print(f'Finsh control!!')
            else:
                # print(f'Have already finish the finding moving forward!!!')
                fc.forward(10)
                time.sleep(0.5)
                fc.stop()
                pid.reset()
                cv2.putText(frame, "No target found",(10,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            fps = 1/dt if dt>0 else 0
            cv2.putText(frame, f"FPS {fps:.2f}",(10,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.line(frame,(IMG_W//2,0),(IMG_W//2,IMG_H),(255,0,255),1)

            bev_resized = cv2.resize(bev, (round(frame.shape[1]*0.75), frame.shape[0]))
            combined = np.hstack([frame, bev_resized])
            cv2.imshow("Lane Following", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s'):
                fn = os.path.join(out_dir, f"lane_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(fn, combined)
            elif key == ord('q'):
                fc.stop()

    finally:
        fc.stop()
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
