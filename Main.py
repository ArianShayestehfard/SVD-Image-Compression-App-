import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="SVD Compression")

def do_compress(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    out = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return np.clip(out, 0, 255).astype(np.uint8)

def calc_k(p, h, w):
    k = max(1, int((p / 100) * h * w / (h + w + 1)))
    return k

st.title("SVD Image Compression")

file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if file:
    file_bytes = file.getvalue()
    original_size = len(file_bytes)
    original_format = file.type.split('/')[-1].upper()

    img = Image.open(file).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    max_k = min(h, w, 100)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.empty()

    how = st.radio("Choose:", ["Set k value", "Set percentage"])

    if how == "Set k value":
        k = st.slider("k", 1, max_k)
    else:
        p = st.slider("Percentage", 1, 100)
        k = calc_k(p, h, w)

    if st.button("Compress"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Compressing...")
        channels = []
        for i in range(3):
            channels.append(do_compress(arr[:,:,i], k))
            progress_bar.progress((i + 1) / 3)

        comp = np.stack(channels, axis=2)
        progress_bar.progress(100)
        status_text.text("Done!")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(comp, caption=f"Compressed (k={k})", use_container_width=True)

        buf = BytesIO()
        comp_img = Image.fromarray(comp)
        comp_img.save(buf, format="JPEG", quality=10)

        compressed_size = len(buf.getvalue())

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Size", f"{original_size/1024:.1f} KB")
        col2.metric("Compressed Size", f"{compressed_size/1024:.1f} KB")
        col3.metric("Reduction", f"{(1 - compressed_size/original_size)*100:.1f}%")

        st.download_button("Download Compressed Image", buf.getvalue(), "compressed.jpg")

    st.divider()
