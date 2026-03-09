import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pytesseract
import cv2

# -------- TESSERACT PATH --------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------- CYBER THEME --------
st.markdown("""
<style>
body {background-color:#0a0a0a;color:#00ffff;}
h1 {text-align:center;color:#00ffff;text-shadow:0px 0px 20px #00ffff;}
.stFileUploader {border:2px dashed #00ffff;padding:20px;}
</style>
""", unsafe_allow_html=True)

st.title("CYBER AI HANDWRITING ANALYZER")

st.write("Upload Accustomed Handwriting and Unaccustomed Handwriting")

# -------- IMAGE UPLOAD --------
img_acc_file = st.file_uploader("Upload ACCUSTOMED HAND image", type=["jpg","png","jpeg"])
img_unacc_file = st.file_uploader("Upload UNACCUSTOMED HAND image", type=["jpg","png","jpeg"])

if img_acc_file and img_unacc_file:

    img_acc = Image.open(img_acc_file)
    img_unacc = Image.open(img_unacc_file)

    arr_acc = np.array(img_acc)
    arr_unacc = np.array(img_unacc)

    st.image(img_acc, caption="Accustomed Hand")
    st.image(img_unacc, caption="Unaccustomed Hand")

    # -------- FORENSIC COMPARISON --------
    h_acc = arr_acc.shape[0]
    h_unacc = arr_unacc.shape[0]

    regions = [
        (0.01,0.15),
        (0.18,0.32),
        (0.55,0.69),
        (0.85,0.99)
    ]

    fig, axes = plt.subplots(len(regions),2, figsize=(12,10))

    for i,(start,end) in enumerate(regions):

        crop_acc = arr_acc[int(start*h_acc):int(end*h_acc),:]
        crop_unacc = arr_unacc[int(start*h_unacc):int(end*h_unacc),:]

        axes[i,0].imshow(crop_acc)
        axes[i,0].set_title("ACCUSTOMED")
        axes[i,0].axis("off")

        axes[i,1].imshow(crop_unacc)
        axes[i,1].set_title("UNACCUSTOMED")
        axes[i,1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # -------- REAL LETTER DETECTION --------
    gray = cv2.cvtColor(arr_acc, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    letters = []

    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)

        if w>10 and h>10:

            crop = arr_acc[y:y+h,x:x+w]
            letters.append(crop)

            cv2.rectangle(arr_acc,(x,y),(x+w,y+h),(0,255,255),2)

    st.subheader("Detected Letters on Real Image")
    st.image(arr_acc)

    # -------- SHOW REAL CROPPED LETTERS --------
    st.subheader("Separated Real Letters")

    cols = st.columns(10)

    for i,letter in enumerate(letters):
        cols[i%10].image(letter)

    # -------- SIMPLE GROUPING BY SIZE --------
    groups = {}

    for letter in letters:
        h,w,_ = letter.shape
        key = (h//10,w//10)

        if key not in groups:
            groups[key] = []

        groups[key].append(letter)

    st.subheader("Grouped Similar Letters (Real Image Pieces)")

    for key in groups:
        cols = st.columns(10)
        for i,img in enumerate(groups[key]):
            cols[i%10].image(img)
