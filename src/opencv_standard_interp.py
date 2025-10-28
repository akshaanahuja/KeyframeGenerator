import cv2 

img_A = cv2.imread("/Users/akshaanahuja/KeyframeGenerator/KeyframeGenerator/data/raw/datasets/test_2k_540p/Disney_v4_0_000024_s2/frame1.png")
img_B = cv2.imread("/Users/akshaanahuja/KeyframeGenerator/KeyframeGenerator/data/raw/datasets/test_2k_540p/Disney_v4_0_000024_s2/frame3.png")
img_in_between_truth = cv2.imread("/Users/akshaanahuja/KeyframeGenerator/KeyframeGenerator/data/raw/datasets/test_2k_540p/Disney_v4_0_000024_s2/frame2.png")

# img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
# img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
# img_in_between_truth = cv2.cvtColor(img_in_between_truth, cv2.COLOR_BGR2RGB)

img_in_between_pred = cv2.addWeighted(img_A, 0.5, img_B, 0.5, 0)

cv2.imshow("img_A", img_A)
cv2.imshow("img_B", img_B)
cv2.imshow("img_in_between_truth", img_in_between_truth)
cv2.imshow("img_in_between_pred", img_in_between_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()
