# Chúng ta cần điều chỉnh kích thước cửa sổ với kích thước và sự quay của mục tiêu. Giải pháp được gọi là CAMshift (Dịch chuyển liên tục thích ứng) do Gary Bradsky năm 1988.

# Đầu tiên là áp dụng meanshift. Sau khi meanshift hội tụ, nó sẽ cập nhật kích thước của cửa sổ và tính toán hướng của hình elip phù hợp nhất với nó. Một lần nữa, áp dụng meanshift với cửa sổ tìm kiếm được chia tỷ lệ mới và vị trí cửa sổ trước đó. Quá trình được tiếp tục cho đến khi đáp ứng được độ chính xác yêu cầu.

# Camshift gần giống như meanhift, chỉ khác là trả về một hình chữ nhật xoay và các tham số được sử dụng để chuyển làm cửa sổ tìm kiếm trong lần lặp tiếp theo.

import numpy as np
import cv2

cap = cv2.VideoCapture('video/input/slow.flv')

ret, frame = cap.read()

r, h, c, w = 200, 30, 300, 80
track_window = (c, r, w, h)

roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        else:
            cv2.imwrite('video/output/' + 'camshift' + chr(k) + ".jpg", img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()