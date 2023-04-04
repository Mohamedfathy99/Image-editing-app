import streamlit as st  # for creating webapp
import cv2  # for image processing operation
from PIL import Image, ImageEnhance
import numpy as np  # to deal with arrays
import os

# haar cascade file:
# detectors/haarcascade_frontalface_default.xml
# detectors/haarcascade_eye.xml

# face_cascade = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('detectors/haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detectFaces(image):
    newImage = np.array(image.convert('RGB'))
    faces = face_cascade.detectMultiScale(newImage, 1.1, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return newImage, faces


def detectEyes(image):
    newImage = np.array(image.convert('RGB'))
    eyes = eye_cascade.detectMultiScale(newImage, 1.3, 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return newImage, eyes


def cartoonize_image(image):
    newImage = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(newImage, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def cannizeImage(image):
    newImage = np.array(image.convert('RGB'))
    img = cv2.GaussianBlur(newImage, (13, 13), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


def main():
    st.title('Image Editing App')
    st.text('Edit Your Images in a fast and simple way')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Acivity', activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            image = Image.open(image_file)
            st.text('Original Image')
            st.image(image, caption='Uploaded image', use_column_width=True)
            # img_array = np.array(image)

            enhance_type = st.sidebar.radio('Enhance type', ['Original', 'Gray-scale',
                                                             'Contrast', 'Brightness',
                                                             'Blurring', 'Sharpness'])

            if enhance_type == 'Gray-scale':
                img = np.array(image.convert('RGB'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray)
            elif enhance_type == "Contrast":
                rate = st.sidebar.slider("Contarst", 0.5, 6.0)
                enhancer = ImageEnhance.Contrast(image)
                enhancedImage = enhancer.enhance(rate)
                st.image(enhancedImage)

            elif enhance_type == "Brightness":
                rate = st.sidebar.slider("Brightness", 0.0, 8.0)
                enhancer = ImageEnhance.Brightness(image)
                enhancedImage = enhancer.enhance(rate)
                st.image(enhancedImage)

            elif enhance_type == "Blurring":
                rate = st.sidebar.slider("Blurring", 0.0, 7.0)
                blurredImage = cv2.GaussianBlur(np.array(image), (15, 15), rate)
                st.image(blurredImage)

            elif enhance_type == "Sharpness":
                rate = st.sidebar.slider("Sharpness", 0.0, 14.0)
                enhancer = ImageEnhance.Sharpness(image)
                enhancedImage = enhancer.enhance(rate)
                st.image(enhancedImage)

            elif enhance_type == "Original":
                st.image(image, width=600)

            else:
                st.image(image, width=600)

        tasks = ["Faces", "Eyes", "Cartoonize", "Cannize"]
        featureChoices = st.sidebar.selectbox("Find Features", tasks)
        if st.button("Process"):
            if featureChoices == "Faces":
                resultImage, resultFaces = detectFaces(image)
                st.image(resultImage)
                st.success("Found {} faces".format(len(resultFaces)))
            elif featureChoices == "Eyes":
                resultImage, resultEyes = detectEyes(image)
                st.image(resultImage)

            elif featureChoices == "Cartoonize":
                resultImage = cartoonize_image(image)
                st.image(resultImage)

            elif featureChoices == "Cannize":
                resultImage = cannizeImage(image)
                st.image(resultImage)



    elif choice == 'About':
        st.subheader('About the developer')
        st.markdown('Built with streamlit by [Mohamed Fathy](https://www.linkedin.com/in/mohamed-fathy-6a4b9b1a5/)')
        st.text("My name is Mohamed Fathy I'm Software engineer")


if __name__ == "__main__":
    main()