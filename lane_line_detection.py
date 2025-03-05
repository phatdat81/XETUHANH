 
import cv2
import numpy as np

import cv2
import numpy as np

# Hàm phát hiện đường lane
def find_lane_lines(img):
    """
    Detecting road markings
    This function will take a color image, in BGR color system,
    Returns a filtered image of road markings
    """
    # Chuyển ảnh sang grayscale để xử lý dễ dàng hơn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Gaussian để giảm nhiễu
    # Kernel size (11,11) càng lớn thì ảnh càng mờ và giảm nhiễu tốt hơn
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)

    # Áp dụng Canny edge detection để phát hiện cạnh
    # thresh_low: ngưỡng dưới để lọc các cạnh yếu
    # thresh_high: ngưỡng trên để xác định các cạnh mạnh
    thresh_low = 150  
    thresh_high = 200
    img_canny = cv2.Canny(img_gauss, thresh_low, thresh_high)

    return img_canny

# Hàm chuyển đổi góc nhìn sang bird-view
def birdview_transform(img):
    """Apply bird-view transform to the image"""
    # Kiểm tra ảnh đầu vào hợp lệ
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid.")

    # Kích thước ảnh đầu ra
    IMAGE_H = 480
    IMAGE_W = 640
    
    # Định nghĩa các điểm nguồn và đích cho phép biến đổi perspective
    # src: 4 điểm trên ảnh gốc
    # dst: 4 điểm tương ứng trên ảnh bird-view
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    
    # Tính ma trận biến đổi và áp dụng warp perspective
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped_img

# Hàm tìm điểm trái phải của làn đường
def find_left_right_points(image, draw=None):
    """Find left and right points of lane"""
    # Lấy kích thước ảnh
    im_height, im_width = image.shape[:2]

    # Xác định vị trí dòng quan tâm (90% chiều cao ảnh từ trên xuống)
    interested_line_y = int(im_height * 0.9)
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (im_width, interested_line_y), (0, 0, 255), 2)
    interested_line = image[interested_line_y, :]

    # Khởi tạo các biến tìm điểm
    left_point = -1
    right_point = -1
    lane_width = 100  # Chiều rộng làn đường dự kiến
    center = im_width // 2

    # Quét từ giữa ra hai bên để tìm điểm đầu tiên có giá trị > 0
    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    # Dự đoán điểm còn lại nếu chỉ thấy một bên
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    # Vẽ các điểm tìm được lên ảnh
    if draw is not None:
        if left_point != -1:
            draw = cv2.circle(
                draw, (left_point, interested_line_y), 7, (255, 255, 0), -1)
        if right_point != -1:
            draw = cv2.circle(
                draw, (right_point, interested_line_y), 7, (0, 255, 0), -1)

    return left_point, right_point

# Hàm tính toán tín hiệu điều khiển
def calculate_control_signal(img, draw=None):
    """Calculate speed and steering angle"""
    # Xử lý ảnh và tìm điểm làn đường
    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)
    draw[:, :] = birdview_transform(draw)
    left_point, right_point = find_left_right_points(img_birdview, draw=draw)

    # Khởi tạo các giá trị điều khiển
    max_throttle = 0.5  # Tốc độ tối đa 50%
    throttle = max_throttle
    steering_angle = 0
    im_center = img.shape[1] // 2

    # Tính toán góc lái và tốc độ nếu phát hiện được làn đường
    if left_point != -1 and right_point != -1:
        # Tính điểm giữa làn đường và độ lệch so với tâm ảnh
        center_point = (right_point + left_point) // 2
        center_diff = im_center - center_point

        # Tính góc lái dựa trên độ lệch (hệ số 0.02 điều chỉnh độ nhạy)
        steering_angle = - float(center_diff * 0.02)

        # Giảm tốc trong khúc cua
        angle_threshold = 5.0  # Ngưỡng góc để bắt đầu giảm tốc
        if abs(steering_angle) > angle_threshold:
            throttle = max_throttle * (1 - (abs(steering_angle) / 20.0))

        # Đảm bảo tốc độ trong khoảng [0.1, max_throttle]
        throttle = np.clip(throttle, 0.1, max_throttle)

    return throttle, steering_angle