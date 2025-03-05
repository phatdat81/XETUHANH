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
    r"D:\p2_traffic_sign_detection_v2\traffic_sign_classifier_lenet_v3.onnx")

# Global queues
g_image_queue = Queue(maxsize=5)
traffic_sign_queue = Queue(maxsize=5)  # New queue for traffic sign messages

# Function to run sign classification model continuously
 #Hàm chạy liên tục mô hình nhận diện biển báo
def process_traffic_sign_loop(image_queue, sign_queue):
    while True:
        if image_queue.empty():
            time.sleep(0.1)
            continue
        image = image_queue.get()
        # Chuẩn bị hình ảnh cho hiển thị
        draw = image.copy()
        # Nhận diện biển báo giao thông
        detected_signs = detect_traffic_signs(image, traffic_sign_model, draw=draw)
        # Hiển thị kết quả lên cửa sổ
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)
        # Nếu phát hiện biển báo dừng, gửi thông điệp đến tiến trình chính
        for sign in detected_signs:
            if sign[0] == "stop":
                if not sign_queue.full():
                    sign_queue.put("stop")
            elif sign[0] == "right":  # Điều kiện phát hiện biển báo rẽ phải
                if not sign_queue.full():
                    sign_queue.put("right")  # Gửi tín hiệu rẽ phải
            elif sign[0] == "left":  # Điều kiện phát hiện biển báo rẽ trái
                if not sign_queue.full():
                    sign_queue.put("left")  # Gửi tín hiệu rẽ trái

async def process_image(websocket, path, sign_queue):
    stop_sign_detected = False  # Cờ để theo dõi trạng thái biển báo dừng
    stop_sign_counter = 0

    right_turn_detected = False  # Cờ để theo dõi trạng thái rẽ phải
    is_turning_right = False     # Cờ để theo dõi xe đang trong quá trình rẽ
    right_turn_counter = 0
    right_turn_delay_counter = 0  # Biến đếm trì hoãn sau khi nhận diện biển báo rẽ

    left_turn_detected = False  # Cờ để theo dõi trạng thái rẽ trái
    is_turning_left = False     # Cờ để theo dõi xe đang trong quá trình rẽ
    left_turn_counter = 0
    left_turn_delay_counter = 0  # Biến đếm trì hoãn sau khi nhận diện biển báo rẽ

    turn_duration_counter = 0     # Biến đếm thời gian rẽ
    distance_to_sign = 100  # Biến giả định khoảng cách đến biển báo (giá trị ban đầu)
    required_detections = 3       # Số lần phát hiện cần thiết để xác nhận biển báo
    turn_duration_threshold = 60  # Giả sử cần 50 frames để hoàn thành rẽ
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
            elif sign == "left":  # Phát hiện biển báo rẽ phải
                left_turn_counter  += 1
                if left_turn_counter >= required_detections:
                    print("Right turn sign detected. Preparing to turn right.")
                    left_turn_detected = True  # Xác nhận sẽ rẽ phải
                    left_turn_delay_counter = 0  # Reset bộ đếm trì hoãn
        # Calculate control signals
        if stop_sign_detected:
            _, steering_angle = calculate_control_signal(image, draw=draw)  # Stop the car
            throttle = -10
        elif right_turn_detected and not is_turning_right:
            # Delay before actually turning

            # CUA PHẢI
            if right_turn_delay_counter < 100:  # Giả sử cần N frame để đến vị trí rẽ
                throttle = 0.2  # Xe vẫn tiếp tục chạy thẳngs
                #steering_angle = 0  # Giữ thẳng tay lái
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
                right_turn_delay_counter += 3.2  # Tăng bộ đếm trì hoãn
            else:          
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
                is_turning_right = True  # Bật cờ xe đang rẽ
                turn_duration_counter = 0  # Reset bộ đếm thời gian rẽ
                print("Turning right now.")
        elif is_turning_right:
            # Xe đang trong quá trình rẽ phải
            throttle = 0.3
            steering_angle = 30  # Giữ góc cua phải
            # Tăng bộ đếm thời gian rẽ
            turn_duration_counter += 1
            if turn_duration_counter >= turn_duration_threshold:
                # Khi đạt đủ thời gian rẽ, quay lại lái thẳng
                throttle = 0.3
                # steering_angle = 0  # Khôi phục góc lái về 0
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
                is_turning_right = False  # Đặt lại trạng thái rẽ
                right_turn_detected = False  # Reset trạng thái rẽ
                print("Completed right turn. Returning to straight driving.")



        #CUA TRÁI
        elif left_turn_detected and not is_turning_left:
            # Delay before actually turning
            if left_turn_delay_counter < 90:  # Giả sử cần N frame để đến vị trí rẽ
                throttle = 0.3  # Xe vẫn tiếp tục chạy thẳng
                #steering_angle = 0  # Giữ thẳng tay lái
                left_turn_delay_counter += 4.3   # Tăng bộ đếm trì hoãn
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
            else:            
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
                is_turning_left = True  # Bật cờ xe đang rẽ
                turn_duration_counter = 0  # Reset bộ đếm thời gian rẽ
                print("Turning right now.")
        elif is_turning_left:
            # Xe đang trong quá trình rẽ phải
            throttle = 0.3
            steering_angle = -30  # Giữ góc cua phải

            # Tăng bộ đếm thời gian rẽ
            turn_duration_counter += 1
            if turn_duration_counter >= turn_duration_threshold:
                # Khi đạt đủ thời gian rẽ, quay lại lái thẳng
                throttle = 0.3
                # steering_angle = 0  # Khôi phục góc lái về 0
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
                is_turning_left = False  # Đặt lại trạng thái rẽ
                left_turn_detected = False  # Reset trạng thái rẽ
                print("Completed right turn. Returning to straight driving.")
        else:
            throttle, steering_angle = calculate_control_signal(image, draw=draw)

        # Update image to g_image_queue - used to run sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Show the result to a window
        cv2.imshow("Result", draw)
        cv2.waitKey(1)
        print(throttle)
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
