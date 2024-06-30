import cv2
import numpy as np

input_image = cv2.imread('input.jpg')
mask_image = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

mask_image_resized = cv2.resize(mask_image, (input_image.shape[1], input_image.shape[0]))

_, binary_mask = cv2.threshold(mask_image_resized, 128, 255, cv2.THRESH_BINARY)

binary_mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

red_background = np.zeros_like(input_image)
red_background[:] = (0, 0, 255)  # BGR format for red

object_image = cv2.bitwise_and(input_image, binary_mask_3ch)

inverse_binary_mask = cv2.bitwise_not(binary_mask)
inverse_binary_mask_3ch = cv2.merge([inverse_binary_mask, inverse_binary_mask, inverse_binary_mask])

background_image = cv2.bitwise_and(red_background, inverse_binary_mask_3ch)

result_image = cv2.add(object_image, background_image)

x, y, w, h = cv2.boundingRect(binary_mask)

roi = result_image[y:y+h, x:x+w]

centered_image = np.zeros_like(input_image)
centered_image[:] = (0, 0, 255)  # BGR format for red

center_x = (centered_image.shape[1] - w) // 2
center_y = (centered_image.shape[0] - h) // 2

centered_image[center_y:center_y+h, center_x:center_x+w] = roi

cv2.imwrite('abc.jpg', centered_image)
