#python color_image.py --img anh/
from argparse import ArgumentParser
import numpy as np
import cv2

# Hàm tăng cường độ bão hòa
def enhance_color_saturation(image, alpha=1.3, beta=1.3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, alpha)
    v = cv2.multiply(v, beta)
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image

def main(image_path):
    # load model
    weights_path = "models/colorization_release_v2.caffemodel"
    config_path = "models/colorization_deploy_v2.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

    # add quantized_ab center, will be used for rebalancing
    pts = np.load("models/pts_in_hull.npy")
    pts = pts.transpose().reshape(2, 313, 1, 1).astype("float32")
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize, extract L channel, perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = resized[:, :, 0]
    L -= 50

    # run model
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # construct colored image from original L channel & predicted ab channel
    predicted_ab = cv2.resize(ab, (width, height))  # resize to original image size
    original_L = lab[:, :, 0]
    colorized = np.concatenate((original_L[:, :, np.newaxis], predicted_ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

    # Tăng cường độ bão hòa
    colorized_enhanced = enhance_color_saturation(colorized)

    # display image
    cv2.imshow("Anh goc", image)
    cv2.imshow("Anh mau", colorized)
    cv2.imshow("Anh mau sang hon", colorized_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img", dest="image_path", required=True)
    args = parser.parse_args()

    main(args.image_path)
