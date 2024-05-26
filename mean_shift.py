# Thuật toán Meanshift có thể được hiểu đơn giản như sau: Giả sử bạn có một tập hợp các điểm (có thể là một dạng phân phối pixel giống như biểu đồ), bạn được cung cấp một cửa sổ nhỏ (có thể là một hình tròn) và bạn phải di chuyển cửa sổ đó đến vùng có mật độ điểm ảnh tối đa (hoặc số điểm tối đa)

# Để sử dụng thuật toán meanshift trong OpenCV, trước tiên chúng ta cần thiết lập mục tiêu, tìm biểu đồ của nó để chúng ta có thể dự đoán lại mục tiêu trên mỗi khung hình để tính toán meanshift. Chúng ta cũng cần cung cấp vị trí ban đầu của cửa sổ. Đối với biểu đồ, chỉ có thang màu Hue được xem xét ở đây. Ngoài ra, để tránh các giá trị sai do ánh sáng yếu, các giá trị ánh sáng yếu sẽ bị loại bỏ bằng cách sử dụng hàm cv2.inRange()

import numpy as np
import cv2

cap = cv2.VideoCapture('video/input/highway.mp4')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 100, 30, 650, 70  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        if x + w + 70 >= frame.shape[1] or y + h + 70 >= frame.shape[0]:
            # Reset the tracking window to its initial position
            track_window = (c, r, w, h)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite('video/output/' + 'meanshift' + chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()