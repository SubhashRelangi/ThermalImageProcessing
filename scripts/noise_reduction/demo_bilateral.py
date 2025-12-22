import cv2
from thermal.noise_reduction.bilateral import thermal_bilateral_filter
from thermal.noise_reduction.bilateral_scores import bilateral_effectiveness_score

img = cv2.imread(
    "data/images/SingleChannel.png",
    cv2.IMREAD_UNCHANGED
)

if img is None:
    raise FileNotFoundError("Input image not found")

out = thermal_bilateral_filter(img)
score = bilateral_effectiveness_score(img, out)

display = out.copy()
cv2.putText(
    display,
    f"Bilateral Score: {score:.1f}%",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255,),
    2
)

cv2.imwrite("outputs/noise_reduction/images/bilateral.png", display)
cv2.imshow("Bilateral Filter", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
