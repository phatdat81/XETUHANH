import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue
import cv2
import numpy as np
import websockets
from PIL import Image

from lane_line_detection import *
from traffic_sign_detection import *

# Initialize traffic sign classifier
traffic_sign_model = cv2.dnn.readNetFromONNX(
    r"D:\AutoCar\hello_via_update\p2_traffic_sign_detection\traffic_sign_classifier_lenet_v3.onnx")

# Global queues
g_image_queue = Queue(maxsize=5)
traffic_sign_queue = Queue(maxsize=5)  # New queue for traffic sign messages

# Function to run sign classification model continuously
def process_traffic_sign_loop(image_queue, sign_queue):
    while True:
        if image_queue.empty():
            time.sleep(0.1)
            continue
        image = image_queue.get()
        # Prepare visualization image
        draw = image.copy()
        # Detect traffic signs
        detected_signs = detect_traffic_signs(image, traffic_sign_model, draw=draw)
        # Show the result to a window
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)
        # If a stop sign is detected, send a message to the main process
        for sign in detected_signs:
            if sign[0] == "stop":
                if not sign_queue.full():
                    sign_queue.put("stop")
            elif sign[0] == "right":  # Thêm điều kiện phát hiện biển báo rẽ phải
                if not sign_queue.full():
                    sign_queue.put("right")  # Gửi tín hiệu rẽ phải
async def process_image(websocket, path, sign_queue):
    stop_sign_detected = False  # Flag to track stop sign state
    right_turn_detected = False  # Flag to track right turn state
    is_turning_right = False
    stop_sign_counter = 0
    right_turn_counter = 0
    right_turn_delay_counter = 0  # Biến đếm trì hoãn sau khi nhận diện biển báo rẽ
    required_detections = 3  # Số lần phát hiện cần thiết để xác nhận biển báo
    while True:
        try:
            message = await websocket.recv()
        except websockets.ConnectionClosed:
            break

        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Prepare visualization image
        draw = image.copy()

        # Check for traffic sign messages
        while not sign_queue.empty():
            sign = sign_queue.get()
            if sign == "stop":
                stop_sign_counter += 1
                if stop_sign_counter >= required_detections:
                    stop_sign_detected = True
                    print("Confirmed stop sign detected. Stopping the car.")
            elif sign == "right":  # Phát hiện biển báo rẽ phải
                right_turn_counter += 1
                if right_turn_counter >= required_detections:
                    print("Right turn sign detected. Preparing to turn right.")
                    right_turn_detected = True  # Xác nhận sẽ rẽ phải
                    right_turn_delay_counter = 0  # Reset bộ đếm trì hoãn

        # Calculate control signals
        if stop_sign_detected:
            _, steering_angle = calculate_control_signal(image, draw=draw)  # Stop the car
            throttle = 0.0
        elif right_turn_detected:
            # Delay before actually turning
            if right_turn_delay_counter < 13:  # Giả sử cần 20 frame để đến vị trí rẽ
                throttle = 0.3  # Xe vẫn tiếp tục chạy thẳng
                steering_angle = 0  # Giữ thẳng tay lái
                right_turn_delay_counter += 1  # Tăng bộ đếm trì hoãn
                # print(f"Delaying turn... Frame: {right_turn_delay_counter}")
            else:
                # Khi đạt đủ số frame trì hoãn, thực hiện rẽ
                throttle = 0.3  # Tốc độ khi rẽ
                steering_angle = 30  # Góc cua khi rẽ phải
                print("Turning right now.")
                is_turning_right = True
        if is_turning_right:
              throttle = 0.3  # Xe vẫn tiếp tục chạy thẳng
              steering_angle = 0  # Giữ thẳng tay lái
                   
            
        else:
            throttle, steering_angle = calculate_control_signal(image, draw=draw)

        # Update image to g_image_queue - used to run sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Show the result to a window
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Send back throttle and steering angle
        response = json.dumps(
            {"throttle": throttle, "steering": steering_angle})
        await websocket.send(response)
        print(f"throttle: {throttle}, steering: {steering_angle}")

 

async def main():
    async def handler(websocket, path):
        await process_image(websocket, path, traffic_sign_queue)

    async with websockets.serve(handler, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue, traffic_sign_queue))
    p.start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        p.terminate()
        p.join()