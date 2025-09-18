import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from skimage import color

MODEL_PATH = "optimized_deepfake_model.h5"

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

def optimized_fft_processing(img, low_cutoff=20):
    if len(img.shape) > 2 and img.shape[2] == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.squeeze()
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(fshift, dtype=np.float32)
    mask[crow-low_cutoff:crow+low_cutoff, ccol-low_cutoff:ccol+low_cutoff] = 1
    fshift *= mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    img_back = img_back / np.max(img_back)
    return img_back.astype(np.float32)

def preprocess_image(image, target_size=(64,64)):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    fft_img = optimized_fft_processing(img)
    fft_img = np.expand_dims(fft_img, axis=-1)
    fft_img = np.expand_dims(fft_img, axis=0)
    return fft_img

# ------------------------
# Custom CSS Styling
# ------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .real-result {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .fake-result {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .uncertain-result {
        background-color: #fff3cd;
        border: 2px solid #ffeeba;
        color: #856404;
    }
    .footer {
        text-align: center;
        color: gray;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
    .confidence-meter {
        background: linear-gradient(90deg, #4CAF50 0%, #FFEB3B 50%, #F44336 100%);
        height: 20px;
        border-radius: 10px;
        position: relative;
        margin: 20px 0;
    }
    .confidence-pointer {
        position: absolute;
        top: -5px;
        width: 2px;
        height: 30px;
        background-color: black;
    }
    .tech-details {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Deepfake Image Detector", layout="wide", page_icon="🕵️")

# Header Section
st.markdown('<h1 class="main-header">🕵️ Deepfake Image Detector</h1>', unsafe_allow_html=True)
# st.markdown("""
#     <div class="info-box">
#     This application uses a deep learning model with frequency domain analysis to detect deepfake images. 
#     Upload an image to analyze its authenticity using our advanced detection algorithms.
#     </div>
#     """, unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">Upload Image</div>', unsafe_allow_html=True)
    
    # File uploader with improved styling
    uploaded_file = st.file_uploader(
        "Drag and drop an image or click to browse", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add image details
        img_array = np.array(image)
        st.info(f"**Image Details:** {img_array.shape[1]}x{img_array.shape[0]} pixels, {uploaded_file.type} format")

with col2:
    st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Create a progress bar with status updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preprocessing image...")
        progress_bar.progress(25)
        
        status_text.text("Applying frequency analysis...")
        progress_bar.progress(50)
        
        status_text.text("Running deepfake detection model...")
        processed_img = preprocess_image(image)
        progress_bar.progress(75)
        
        status_text.text("Finalizing results...")
        prediction = model.predict(processed_img)[0][0]
        progress_bar.progress(100)
        
        # Clear the progress elements
        progress_bar.empty()
        status_text.empty()

        # Determine prediction and confidence
        if prediction < 0.80:
            predicted_class = "Real"
            result_style = "real-result"
            icon = "✅"
        elif prediction > 0.20:
            predicted_class = "Deepfake"
            result_style = "fake-result"
            icon = "⚠️"
        else:
            predicted_class = "Real (Uncertain)"
            result_style = "uncertain-result"
            icon = "🤔"

        confidence = prediction if predicted_class == "Deepfake" else 1 - prediction
        
        # Display results in a styled box
        st.markdown(f'<div class="result-box {result_style}"><h2>{icon} {predicted_class}</h2></div>', unsafe_allow_html=True)
        
        # Custom confidence meter
        st.markdown("**Confidence Level:**")
        confidence_percent = confidence * 100
        st.markdown(f"""
            <div class="confidence-meter">
                <div class="confidence-pointer" style="left: {confidence_percent}%;"></div>
            </div>
            <div style="text-align: center; font-weight: bold; margin-top: 10px;">
                {confidence_percent:.1f}%
            </div>
        """, unsafe_allow_html=True)
        
        # Additional technical details in an expander
        with st.expander("View Technical Details"):
            st.markdown("""
                <div class="tech-details">
                    <strong>Raw Prediction Value:</strong> {:.4f}<br>
                    <strong>Confidence Score:</strong> {:.2f}%<br>
                    <strong>Model:</strong> Custom CNN with Frequency Domain Analysis<br>
                    <strong>Input Size:</strong> 64x64 pixels (grayscale)<br>
                    <strong>Training Dataset:</strong> Celeb-DF, FaceForensics++
                </div>
            """.format(prediction, confidence_percent), unsafe_allow_html=True)
            
        # Explanation of the result
        if predicted_class == "Real":
            st.success("This image appears to be authentic. Our analysis didn't detect significant signs of AI manipulation.")
        elif predicted_class == "Deepfake":
            st.error("This image shows characteristics consistent with AI-generated or manipulated content. Please verify through additional sources.")
        else:
            st.warning("The analysis is inconclusive. This image may be of low quality or contain elements that make detection challenging.")

# Add project information in the sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center;">
            <h1>About This Project</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This deepfake detection system is designed as a final year project for Computer Science students.
    
    **Key Features:**
    - Frequency domain analysis using FFT
    - Deep learning model with CNN architecture
    - Real-time image processing
    - Confidence-based predictions
    
    **Technical Stack:**
    - TensorFlow/Keras for model inference
    - Streamlit for web interface
    - OpenCV for image processing
    - Scikit-image for frequency analysis
    
    **Developed By:** Nishant Sahu
    
    """)
    
    # Add a download button for sample images
    st.markdown("---")
    st.markdown("### Need Sample Images?")
    
    # Placeholder for sample images - in a real app, you would provide actual files
    st.info("Sample images would be available here in a complete implementation")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
    <p>For research and educational purposes only. Results may not be 100% accurate.</p>
    </div>
    """, unsafe_allow_html=True)