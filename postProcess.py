import cv2
import numpy as np
import os
import glob

# https://docs.opencv.org/2.4/modules/superres/doc/super_resolution.html

# https://github.com/soroushj/python-opencv-numpy-example/blob/master/unsharpmask.py
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# https://github.com/maitek/image_stacking/blob/master/auto_stack.py
def stackImagesECC(file_list):
    M = np.eye(3, 3, dtype=np.float32)
    number_of_iterations = 1000 # 5000
    termination_eps = 0.00001 # 1e-10
    C = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    first_image = None
    stacked_image = None

    for file in file_list:
        image = cv2.imread(file,1).astype(np.float32) / 255
        print(file)
        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            stacked_image = image
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY, C, None, 5)
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            stacked_image += image

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image

def stackImagesKeypointMatching(file_list):

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        print(file)
        image = cv2.imread(file,1)
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image

def shift(xs, n):
    if n >= 0:
        return np.r_[np.full(n, np.nan), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, np.nan)]

cap = cv2.VideoCapture(0)
#targetRes = [1280, 720-10] # resolution of screen
targetRes = [1920, 1080] # resolution of screen
originalRes = [384, 288] # resolution of camera
aspectRatio = originalRes[0]/originalRes[1]
fitRes = (int((targetRes[1]/originalRes[1])*originalRes[0]), targetRes[1])

stacks = 21
imageSeq = [0]*stacks # max number of frames to be stacked
first = 1

files = glob.glob(r'C:\Projects\SuperResolution\images\*')
for f in files:
    os.remove(f)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayResized = cv2.resize(gray, fitRes, interpolation = cv2.INTER_CUBIC)
    imageSeq[0] = grayResized

    for i in range (0,stacks):
        cv2.imwrite(r"C:\Projects\SuperResolution\images\image%04i.jpg" %i, imageSeq[i])
    image_folder = r"C:\Projects\SuperResolution\images"
    
    # outImage = stackImagesKeypointMatching(file_list)
    cv2.imwrite( r"C:\Projects\SuperResolution\inputPost1.jpg", frame )
    # cv2.imwrite( r"C:\Projects\SuperResolution\ouput1.jpg", outImage )
    # Display the resulting frame
    # cv2.imshow('frame', outImage)
    cv2.imshow('og', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        file_list = os.listdir(image_folder)
        file_list = [os.path.join(image_folder, x) for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]
        # outImage = stackImagesKeypointMatching(file_list)
        outImage = stackImagesECC(file_list)
        outImage = unsharp_mask(outImage, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0)
        cv2.imwrite( r"C:\Projects\SuperResolution\outputPost1.jpg", outImage )
        break
    # imageSeq[10] = outImage
    num = stacks
    while(num>0):
        imageSeq[num-1] = imageSeq[num - 2]
        num -= 1
        
    # shift(imageSeq, 1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
cv2.imshow('og', outImage)


