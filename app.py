import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import exposure
import io, base64, time
from streamlit.components.v1 import html
if 'achievements' not in st.session_state:
    st.session_state.achievements = {
        'first_upload': {'earned': False, 'name': '📸 First Upload!'},
        'selfie_master': {'earned': False, 'name': '🤳 Selfie Master'},
        'meme_genius': {'earned': False, 'name': '😂 Meme Genius'},
        'filter_king': {'earned': False, 'name': '👑 Filter King'}
    }

if 'filter_count' not in st.session_state:
    st.session_state.filter_count = 0
    
if 'tutorial_step' not in st.session_state:
    st.session_state.tutorial_step = 0
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    .css-1d391kg {
        background-color: #353535;
        color: #fff;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e2e2e, #1c1c1c);
        color: #fff;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .section-header {
        border-bottom: 2px solid #3498db;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def load_image(image_file):
    """Load an image from an uploaded file and convert it to a NumPy array."""
    img = Image.open(image_file)
    return np.array(img)

def plot_histogram(image, title="Histogram"):
    """Plot a histogram of the image pixel values."""
    fig, ax = plt.subplots()
    ax.hist(image.ravel(), bins=256, range=(0, 256), color='#3498db', edgecolor='black')
    ax.set_title(title)
    return fig

def display_fourier_transform(image_gray):
    """Compute and return the Fourier transform magnitude spectrum plot."""
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    fig, ax = plt.subplots()
    ax.imshow(magnitude_spectrum, cmap='inferno')
    ax.set_title("Fourier Transform Magnitude Spectrum")
    return fig

def apply_noise(image, noise_intensity):
    """Simulate noise on an image."""
    noisy = image + noise_intensity * np.random.randn(*image.shape) * 255
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def detect_sift_features(image_gray):
    """Detect SIFT keypoints and return an image with keypoints drawn."""
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)
        keypoints_img = cv2.drawKeypoints(image_gray, keypoints, None,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                        color=(0, 255, 0))
        return keypoints_img
    except:
        return image_gray

# --- Confetti Animation ---
def confetti():
    confetti_js = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
    <script>
    var duration = 3000;
    var end = Date.now() + duration;
    (function frame() {
        confetti({
            particleCount: 100,
            angle: 60,
            spread: 55,
            origin: { x: 0 },
            colors: ['#ff0000', '#00ff00', '#0000ff']
        });
        confetti({
            particleCount: 100,
            angle: 120,
            spread: 55,
            origin: { x: 1 },
            colors: ['#ff0000', '#00ff00', '#0000ff']
        });
        if (Date.now() < end) requestAnimationFrame(frame);
    }());
    </script>
    """
    html(confetti_js, height=0)

# --- Safe Balloon Function ---
def safe_balloon():
    """A safe wrapper for st.balloon() that checks if it exists first"""
    try:
        st.snow()  # Use st.snow() as an alternative
        st.success("🎈 Balloons! 🎈")
    except AttributeError:
        st.success("🎈 Balloons! 🎈")

# --- Update achievement function ---
def unlock_achievement(key):
    if key in st.session_state.achievements and not st.session_state.achievements[key]['earned']:
        st.session_state.achievements[key]['earned'] = True
        return True
    return False

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Image Processing & Pattern Analysis")
app_mode = st.sidebar.selectbox("Select a Page", [
    "Welcome", 
    "Photo Booth 🎮",
    "Meme Factory 😂",
    "Image Digitization",
    "Histogram & Metrics",
    "Filtering & Enhancements",
    "Edge Detection & Features",
    "Transforms & Frequency Domain",
    "Image Restoration",
    "Segmentation & Representation",
    "Shape Analysis"
])

# --- Page: Welcome ---
if app_mode == "Welcome":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://em-content.zobj.net/thumbs/120/apple/354/party-popper_1f389.png", width=150)
    with col2:
        st.title("Welcome to Image Playground! 🎪")
    
    st.markdown("### 🔥 Your Achievements")
    ach_cols = st.columns(4)
    for i, (key, ach) in enumerate(st.session_state.achievements.items()):
        with ach_cols[i]:
            if ach['earned']:
                st.success(f"{ach['name']} ✅")
            else:
                st.info("Locked 🔒")
    
    with st.expander("🚀 Quick Start Challenge!", expanded=True):
        st.markdown("""
        Complete these fun tasks to unlock achievements:
        1. Upload any image → Unlock 📸  
        2. Take a webcam selfie → Unlock 🤳  
        3. Create a meme → Unlock 😂  
        4. Apply 5 filters → Unlock 👑  
        """)
        
        # Ensure tutorial_step doesn't exceed 4
        progress_value = min(st.session_state.tutorial_step/4, 1.0)
        tutorial_progress = st.progress(progress_value)
        status_text = st.empty()
    
        if st.button("🎯 Start Tutorial"):
            # Increment tutorial step but cap it at 4
            if st.session_state.tutorial_step < 4:
                st.session_state.tutorial_step += 1
            st.rerun()
        
        if st.session_state.tutorial_step > 0:
            status_dict = {
                "1️⃣": "Upload an image in any section",
                "2️⃣": "Take a selfie in Photo Booth",
                "3️⃣": "Create a meme in Meme Factory",
                "4️⃣": "Apply 5 different filters"
            }
            
            # Show the appropriate task based on current step
            step_key = list(status_dict.keys())[min(st.session_state.tutorial_step - 1, 3)]
            current_task = status_dict[step_key]
            status_text.markdown(f"**Current Task:** {step_key} {current_task}")
            
            # Update progress bar safely
            tutorial_progress.progress(min(st.session_state.tutorial_step/4, 1.0))
    
    st.markdown("### Your Progress")
    progress = st.progress(0)
    num_achieved = sum(1 for a in st.session_state.achievements.values() if a['earned'])
    progress.progress(num_achieved / len(st.session_state.achievements))
    st.markdown('</div>', unsafe_allow_html=True)

# # --- Page: Photo Booth 🎮 ---
# elif app_mode == "Photo Booth 🎮":
#     st.markdown('<div class="main">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-header">📸 Crazy Photo Booth</h2>', unsafe_allow_html=True)
    
#     picture = st.camera_input("Take a selfie!", key="webcam")
#     if picture:
#         if unlock_achievement('selfie_master'):
#             confetti()
#             st.success("🤳 Achievement Unlocked: Selfie Master!")
        
#         st.markdown("### 🎭 Add Crazy Filters")
#         img = load_image(picture)
#         ar_type = st.selectbox("Choose AR Effect", 
#                              ["None", "Dog Ears", "Rainbow Vomit", "Alien Eyes"])
        
#         # Create a copy of the image for modifications
#         result_img = img.copy()
        
#         if ar_type != "None":
#             try:
#                 # Convert to grayscale for face detection
#                 if len(img.shape) == 3:
#                     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#                 else:
#                     gray = img
                
#                 # Try to use cascade classifier for face detection
#                 try:
#                     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#                     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#                 except:
#                     # If cascade classifier fails, add a fake "face" in the center
#                     h, w = img.shape[:2]
#                     faces = np.array([[w//4, h//4, w//2, h//2]])
                
#                 if len(faces) > 0:
#                     for (x, y, w, h) in faces:
#                         if ar_type == "Dog Ears":
#                             # Draw triangular dog ears
#                             cv2.fillConvexPoly(result_img, 
#                                             np.array([[x, y], [x - w//4, y - h//4], [x + w//4, y - h//4]]), 
#                                             (165, 42, 42))  # Brown color
#                             cv2.fillConvexPoly(result_img, 
#                                             np.array([[x+w, y], [x+w - w//4, y - h//4], [x+w + w//4, y - h//4]]), 
#                                             (165, 42, 42))  # Brown color
                            
#                         elif ar_type == "Rainbow Vomit":
#                             # Create rainbow vomit effect
#                             rainbow_height = h // 2
#                             rainbow_width = w
#                             rainbow_y_start = y + h // 2
#                             rainbow_x_start = x
                            
#                             # Check bounds
#                             if rainbow_y_start + rainbow_height <= result_img.shape[0] and rainbow_x_start + rainbow_width <= result_img.shape[1]:
#                                 for i in range(rainbow_height):
#                                     color_value = 255 * (i / rainbow_height)
#                                     for j in range(rainbow_width):
#                                         # Create a gradient rainbow effect
#                                         result_img[rainbow_y_start + i, rainbow_x_start + j] = [
#                                             int(color_value), 
#                                             int(255 - color_value), 
#                                             int(j * 255 / rainbow_width)
#                                         ]
                            
#                         elif ar_type == "Alien Eyes":
#                             # Draw alien eyes
#                             eye1_x = x + w // 3
#                             eye2_x = x + 2 * w // 3
#                             eye_y = y + h // 3
#                             eye_size = max(10, w // 10)
                            
#                             cv2.circle(result_img, (eye1_x, eye_y), eye_size, (0, 255, 0), -1)  # Green left eye
#                             cv2.circle(result_img, (eye2_x, eye_y), eye_size, (0, 255, 0), -1)  # Green right eye
#                             # Add black pupils
#                             cv2.circle(result_img, (eye1_x, eye_y), eye_size // 2, (0, 0, 0), -1)
#                             cv2.circle(result_img, (eye2_x, eye_y), eye_size // 2, (0, 0, 0), -1)
#                 else:
#                     st.warning("No faces detected. Try a different photo or adjust lighting for better detection.")
#             except Exception as e:
#                 st.error(f"Error applying filter: {e}")
#                 # Fall back to a simple color adjustment if face detection fails
#                 if ar_type == "Dog Ears":
#                     result_img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
#                 elif ar_type == "Rainbow Vomit":
#                     result_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#                 elif ar_type == "Alien Eyes":
#                     result_img = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
        
#         st.image(result_img, caption="Your AR Selfie!", use_column_width=True)
#         if st.button("Download Your Masterpiece"):
#             st.session_state.filter_count += 1
#             if st.session_state.filter_count >= 5:
#                 unlock_achievement('filter_king')
#     st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Photo Booth 🎮 ---
elif app_mode == "Photo Booth 🎮":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📸 Advanced Photo Studio</h2>', unsafe_allow_html=True)
    
    picture = st.camera_input("Capture an image", key="webcam")
    if picture:
        if unlock_achievement('photography_pro'):
            confetti()
            st.success("📷 Achievement Unlocked: Photography Pro!")
        
        st.markdown("### 🎨 Image Effects Gallery")
        img = load_image(picture)
        
        # More sophisticated filter options
        filter_type = st.selectbox("Select Effect", 
                             ["Original", "Film Noir", "Neo Cyberpunk", "Vintage", 
                              "Portrait Pro", "Glitch Art", "Vaporwave"])
        
        # Intensity slider for adjustable effects
        intensity = st.slider("Effect Intensity", 0.1, 1.0, 0.7, 0.1)
        
        # Advanced parameters for specific filters
        col1, col2 = st.columns(2)
        with col1:
            if filter_type in ["Neo Cyberpunk", "Glitch Art"]:
                hue_shift = st.slider("Color Shift", 0, 180, 30)
            elif filter_type == "Film Noir":
                contrast = st.slider("Contrast", 0.5, 2.0, 1.2, 0.1)
            elif filter_type == "Portrait Pro":
                blur_amount = st.slider("Skin Smoothing", 1, 15, 5)
        
        with col2:
            if filter_type in ["Vintage", "Vaporwave"]:
                saturation = st.slider("Saturation", 0.0, 2.0, 0.8, 0.1)
            elif filter_type == "Glitch Art":
                glitch_strength = st.slider("Glitch Strength", 1, 20, 8)
            elif filter_type == "Portrait Pro":
                brightness = st.slider("Brightness", 0.8, 1.5, 1.1, 0.05)
        
        # Create a copy of the image for modifications
        result_img = img.copy()
        
        if filter_type != "Original":
            try:
                # Convert to grayscale for face detection (when needed)
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
                
                # Attempt face detection for portrait filters
                faces = []
                if filter_type == "Portrait Pro":
                    try:
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    except Exception as e:
                        st.warning(f"Face detection limited: {str(e)}")
                
                # Apply selected filter
                if filter_type == "Film Noir":
                    # High contrast black and white with film grain
                    result_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    result_img = cv2.convertScaleAbs(result_img, alpha=contrast, beta=0)
                    
                    # Add film grain
                    grain = np.random.normal(0, 15 * intensity, result_img.shape).astype(np.uint8)
                    result_img = cv2.add(result_img, grain)
                    
                    # Convert back to RGB for consistency
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
                    
                    # Add vignette effect
                    rows, cols = result_img.shape[:2]
                    kernel_x = cv2.getGaussianKernel(cols, cols/3)
                    kernel_y = cv2.getGaussianKernel(rows, rows/3)
                    kernel = kernel_y * kernel_x.T
                    mask = 255 * kernel / np.linalg.norm(kernel)
                    for i in range(3):
                        result_img[:, :, i] = result_img[:, :, i] * mask * intensity
                
                elif filter_type == "Neo Cyberpunk":
                    # Split the image into channels
                    b, g, r = cv2.split(result_img)
                    
                    # Shift hue by adjusting the color channels
                    r_shift = int(hue_shift * intensity) % 180
                    b = np.roll(b, r_shift, axis=0)
                    
                    # Enhance the blues and reduce reds for cyberpunk feel
                    b = np.clip(b * (1.5 * intensity), 0, 255).astype(np.uint8)
                    r = np.clip(r * (0.8), 0, 255).astype(np.uint8)
                    
                    # Add some color artifacts
                    noise = np.random.normal(0, 15 * intensity, g.shape).astype(np.uint8)
                    g = cv2.add(g, noise)
                    
                    # Merge channels back
                    result_img = cv2.merge([b, g, r])
                    
                    # Add a subtle blue/purple tint
                    tint = np.zeros_like(result_img, dtype=np.uint8)
                    tint[:] = (50, 0, 30)  # BGR format (blue/purple tint)
                    result_img = cv2.addWeighted(result_img, 1, tint, 0.2 * intensity, 0)
                
                elif filter_type == "Vintage":
                    # Convert to HSV
                    hsv = cv2.cvtColor(result_img, cv2.COLOR_RGB2HSV)
                    h, s, v = cv2.split(hsv)
                    
                    # Adjust saturation
                    s = np.clip(s * saturation, 0, 255).astype(np.uint8)
                    
                    # Add sepia tone
                    h = np.clip(h * 0.5 + 20, 0, 179).astype(np.uint8)
                    
                    # Reduce brightness slightly for aged look
                    v = np.clip(v * 0.9, 0, 255).astype(np.uint8)
                    
                    # Merge channels
                    hsv = cv2.merge([h, s, v])
                    result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    
                    # Add subtle vignette
                    rows, cols = result_img.shape[:2]
                    kernel_x = cv2.getGaussianKernel(cols, cols/2)
                    kernel_y = cv2.getGaussianKernel(rows, rows/2)
                    kernel = kernel_y * kernel_x.T
                    mask = 255 * kernel / np.linalg.norm(kernel)
                    for i in range(3):
                        result_img[:, :, i] = result_img[:, :, i] * mask * intensity
                    
                    # Add some scratches
                    num_scratches = int(20 * intensity)
                    for _ in range(num_scratches):
                        x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
                        x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
                        thickness = np.random.randint(1, 3)
                        cv2.line(result_img, (x1, y1), (x2, y2), (200, 200, 200), thickness)
                
                elif filter_type == "Portrait Pro":
                    # Apply skin smoothing and beautification
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            # Extract face region
                            face_roi = result_img[y:y+h, x:x+w]
                            
                            # Apply bilateral filter for skin smoothing while preserving edges
                            face_roi = cv2.bilateralFilter(face_roi, blur_amount, 75, 75)
                            
                            # Enhance eyes
                            eye_y = y + int(h * 0.35)
                            eye_height = int(h * 0.15)
                            eye_width = int(w * 0.2)
                            
                            # Left eye region
                            left_eye_x = x + int(w * 0.25)
                            left_eye_roi = result_img[eye_y:eye_y+eye_height, 
                                                 left_eye_x:left_eye_x+eye_width]
                            # Enhance contrast for eyes
                            if left_eye_roi.size > 0:
                                left_eye_roi = cv2.convertScaleAbs(left_eye_roi, alpha=1.1, beta=5)
                            
                            # Right eye region
                            right_eye_x = x + int(w * 0.55)
                            right_eye_roi = result_img[eye_y:eye_y+eye_height, 
                                                  right_eye_x:right_eye_x+eye_width]
                            # Enhance contrast for eyes
                            if right_eye_roi.size > 0:
                                right_eye_roi = cv2.convertScaleAbs(right_eye_roi, alpha=1.1, beta=5)
                            
                            # Apply brightness adjustment to face
                            face_roi = cv2.convertScaleAbs(face_roi, alpha=brightness, beta=10)
                            
                            # Put modified face back
                            result_img[y:y+h, x:x+w] = face_roi
                    else:
                        # Fall back to overall image enhancement if no faces detected
                        result_img = cv2.bilateralFilter(result_img, blur_amount, 50, 50)
                        result_img = cv2.convertScaleAbs(result_img, alpha=brightness, beta=10)
                
                elif filter_type == "Glitch Art":
                    # Create digital glitch effect
                    rows, cols = result_img.shape[:2]
                    
                    # Channel shifting
                    b, g, r = cv2.split(result_img)
                    
                    # Horizontal channel shift
                    shift_amount = int(glitch_strength * intensity)
                    if cols > shift_amount > 0:
                        r = np.roll(r, shift_amount, axis=1)
                        b = np.roll(b, -shift_amount, axis=1)
                    
                    # Random blocks of shifted pixels
                    num_glitches = int(10 * intensity)
                    for _ in range(num_glitches):
                        block_height = np.random.randint(10, max(11, int(rows/10)))
                        block_y = np.random.randint(0, rows - block_height)
                        block_width = np.random.randint(int(cols/5), int(cols/2))
                        block_x = np.random.randint(0, cols - block_width)
                        
                        # Shift a color channel in the block
                        channel = np.random.choice([0, 1, 2])
                        shift = np.random.randint(5, glitch_strength)
                        
                        if channel == 0:  # B channel
                            b_block = b[block_y:block_y+block_height, block_x:block_x+block_width]
                            b[block_y:block_y+block_height, block_x:block_x+block_width] = np.roll(b_block, shift, axis=1)
                        elif channel == 1:  # G channel
                            g_block = g[block_y:block_y+block_height, block_x:block_x+block_width]
                            g[block_y:block_y+block_height, block_x:block_x+block_width] = np.roll(g_block, shift, axis=1)
                        else:  # R channel
                            r_block = r[block_y:block_y+block_height, block_x:block_x+block_width]
                            r[block_y:block_y+block_height, block_x:block_x+block_width] = np.roll(r_block, shift, axis=1)
                    
                    # Merge channels
                    result_img = cv2.merge([b, g, r])
                    
                    # Add some digital noise
                    noise = np.random.normal(0, 20 * intensity, result_img.shape).astype(np.uint8)
                    result_img = cv2.add(result_img, noise)
                
                elif filter_type == "Vaporwave":
                    # Convert to HSV
                    hsv = cv2.cvtColor(result_img, cv2.COLOR_RGB2HSV)
                    h, s, v = cv2.split(hsv)
                    
                    # Shift hue towards pink/purple
                    h = np.mod(h + 120, 180).astype(np.uint8)
                    
                    # Increase saturation
                    s = np.clip(s * saturation * 1.5, 0, 255).astype(np.uint8)
                    
                    # Merge channels
                    hsv = cv2.merge([h, s, v])
                    result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    
                    # Add duotone effect (purple/cyan gradient)
                    cyan = np.zeros_like(result_img)
                    cyan[:] = (255, 255, 0)  # BGR format (cyan)
                    
                    magenta = np.zeros_like(result_img)
                    magenta[:] = (255, 0, 255)  # BGR format (magenta)
                    
                    # Create gradient mask
                    rows, cols = result_img.shape[:2]
                    gradient_mask = np.zeros((rows, cols), dtype=np.float32)
                    for i in range(rows):
                        gradient_mask[i, :] = i / rows
                    
                    # Apply gradient mask
                    cyan_part = cyan * (1 - gradient_mask)[:, :, np.newaxis]
                    magenta_part = magenta * gradient_mask[:, :, np.newaxis]
                    duotone = (cyan_part + magenta_part).astype(np.uint8)
                    
                    # Blend with original
                    result_img = cv2.addWeighted(result_img, 0.7, duotone, 0.3 * intensity, 0)
                    
                    # Add some scanlines
                    scanline_gap = max(2, int(10 * (1 - intensity)))
                    for y in range(0, rows, scanline_gap):
                        if y < rows:
                            result_img[y:y+1, :] = result_img[y:y+1, :] * 0.7
            except Exception as e:
                st.error(f"Error applying filter: {e}")
                st.write("Falling back to basic filter...")
                # Fall back to basic color adjustments if advanced processing fails
                if filter_type == "Film Noir":
                    result_img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                elif filter_type == "Neo Cyberpunk":
                    result_img = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
                elif filter_type == "Vintage":
                    result_img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
                elif filter_type == "Glitch Art":
                    result_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                elif filter_type == "Vaporwave":
                    result_img = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
        
        # Display the result with metadata
        st.image(result_img, caption=f"Advanced Photo: {filter_type}", use_column_width=True)
        
        # Additional options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Image"):
                st.session_state.filter_count += 1
                if st.session_state.filter_count >= 5:
                    unlock_achievement('visual_artist')
                    confetti()
                    st.success("🎨 Achievement Unlocked: Visual Artist!")
        
        with col2:
            if st.button("Compare Original"):
                st.image(img, caption="Original Photo", use_column_width=True)
    
    st.markdown("---")
    st.markdown("### 📱 Pro Tips")
    st.markdown("- For best results, position yourself in good lighting")
    st.markdown("- Try different effect intensities for subtle to dramatic transformations")
    st.markdown("- Portrait Pro works best with clear face visibility")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Meme Factory 😂 ---
elif app_mode == "Meme Factory 😂":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">😂 Meme Generator</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your meme template", type=["png", "jpg", "jpeg"], key="meme")
    if uploaded_file:
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        img = Image.open(uploaded_file)
        st.image(img, caption="Your Meme Canvas", use_column_width=True)
        
        top_text = st.text_input("Top Text", "When I see")
        bottom_text = st.text_input("Bottom Text", "a Streamlit app")
        
        if st.button("Generate Meme!"):
            # Create a copy of the image to avoid modifying the original
            meme_img = img.copy()
            draw = ImageDraw.Draw(meme_img)
            try:
                # Use default font instead of trying to load Impact
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Calculate font size based on image width
            font_size = max(14, int(meme_img.width / 15))
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                # Fall back to default font if Arial is not available
                pass
            
            # Draw top text
            bbox = draw.textbbox((0, 0), top_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((meme_img.width - text_width) // 2, 10)
            # Add text shadow for better visibility
            draw.text((position[0]-2, position[1]-2), top_text, font=font, fill="black")
            draw.text((position[0]+2, position[1]-2), top_text, font=font, fill="black")
            draw.text((position[0]-2, position[1]+2), top_text, font=font, fill="black")
            draw.text((position[0]+2, position[1]+2), top_text, font=font, fill="black")
            # Draw main text
            draw.text(position, top_text, font=font, fill="white")
            
            # Draw bottom text
            bbox = draw.textbbox((0, 0), bottom_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((meme_img.width - text_width) // 2, meme_img.height - text_height - 10)
            # Add text shadow for better visibility
            draw.text((position[0]-2, position[1]-2), bottom_text, font=font, fill="black")
            draw.text((position[0]+2, position[1]-2), bottom_text, font=font, fill="black")
            draw.text((position[0]-2, position[1]+2), bottom_text, font=font, fill="black")
            draw.text((position[0]+2, position[1]+2), bottom_text, font=font, fill="black")
            # Draw main text
            draw.text(position, bottom_text, font=font, fill="white")
            
            st.image(meme_img, caption="Your Fresh Meme", use_column_width=True)
            if unlock_achievement('meme_genius'):
                st.success("😂 Achievement Unlocked: Meme Genius!")
                safe_balloon()
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Image Digitization ---
elif app_mode == "Image Digitization":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📷 Interactive Pixel Explorer</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="dig")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        st.markdown("### 🔍 Pixel Inspector")
        
        # Display the image without the canvas
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Create a simpler pixel inspector without st_canvas
        st.markdown("### 📊 Image Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Image Shape: {image.shape}")
            st.write(f"Image Type: {image.dtype}")
            
            if image.ndim == 3:
                avg_color = np.mean(image, axis=(0,1))
                st.write(f"Average RGB: [{int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])}]")
                st.color_picker("Average Color", '#%02x%02x%02x' % 
                               (int(avg_color[0]), int(avg_color[1]), int(avg_color[2])))
        
        with col2:
            # Let user crop with sliders
            if image.ndim == 3:
                height, width, _ = image.shape
            else:
                height, width = image.shape
                
            st.write("Crop Region")
            x1 = st.slider("X Start", 0, width-10, 0)
            x2 = st.slider("X End", x1+1, width, min(x1+100, width))
            y1 = st.slider("Y Start", 0, height-10, 0)
            y2 = st.slider("Y End", y1+1, height, min(y1+100, height))
            
            cropped = image[y1:y2, x1:x2]
            st.image(cropped, caption="Cropped Region", use_column_width=True)
        
        if st.checkbox("🚦 Apply Neon Glow Effect"):
            if image.ndim == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv[...,1] = 255  # Max saturation
                neon_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                st.image(neon_img, caption="Neon Version", use_column_width=True)
            else:
                st.warning("Need a color image for neon effect")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Histogram & Metrics ---
elif app_mode == "Histogram & Metrics":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 Live Histogram Playground</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="hist2")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        st.markdown("### 🎚️ Live Histogram Manipulation")
        col1, col2 = st.columns(2)
        with col1:
            gamma = st.slider("Gamma Correction", 0.1, 3.0, 1.0)
            adjusted = exposure.adjust_gamma(image_gray, gamma)
        with col2:
            equalize = st.checkbox("Enable Histogram Equalization")
            if equalize:
                adjusted = exposure.equalize_hist(adjusted)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_gray, caption="Original", use_column_width=True)
            st.pyplot(plot_histogram(image_gray))
        with col2:
            st.image(adjusted, caption="Adjusted", use_column_width=True)
            st.pyplot(plot_histogram(adjusted))
        
        hist = np.histogram(image_gray.ravel(), bins=256)[0]
        hist = hist / hist.sum()
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        st.markdown(f"""
        ### 🧮 Entropy Gauge
        <div style="background: #ddd; width: 100%; height: 30px; border-radius: 15px">
            <div style="background: linear-gradient(90deg, red, yellow, green); 
                width: {entropy/8*100}%; height: 100%; border-radius: 15px; 
                text-align: center; color: black">
                {entropy:.2f} bits/pixel
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Filtering & Enhancements ---
elif app_mode == "Filtering & Enhancements":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🎮 Filter Arcade</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="filt")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        filter_choice = st.radio("Select Filter Mode:", [
            "🤖 Cyborg Vision", 
            "🍄 Mushroom Kingdom", 
            "🕶️ Noir", 
            "🌈 Rainbow Boost"
        ], horizontal=True)
        
        try:
            if filter_choice == "🤖 Cyborg Vision":
                processed = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            elif filter_choice == "🍄 Mushroom Kingdom":
                processed = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                processed[...,0] = (processed[...,0] + 90) % 180
                processed = cv2.cvtColor(processed, cv2.COLOR_HSV2RGB)
            elif filter_choice == "🕶️ Noir":
                if image.ndim == 3:
                    processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                else:
                    processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif filter_choice == "🌈 Rainbow Boost":
                if image.ndim == 3:
                    try:
                        processed = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                        processed[:,:,1] = np.clip(processed[:,:,1]*1.5, 0, 255)
                        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
                    except:
                        # Fallback if LAB conversion fails
                        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                        hsv[...,1] = np.clip(hsv[...,1]*1.5, 0, 255)
                        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:
                    # Apply a colormap for grayscale images
                    processed = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)
        except Exception as e:
            st.error(f"Error applying filter: {e}")
            processed = image  # Fallback to original image
        
        st.markdown("### 👆 Before/After Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(processed, caption="Filtered", use_column_width=True)
        
        # Increment filter count in session state
        st.session_state.filter_count += 1
        
        if st.session_state.filter_count >= 5:
            if unlock_achievement('filter_king'):
                st.success("👑 Achievement Unlocked: Filter King!")
                confetti()
        
        if st.button("📸 Save as Polaroid"):
            try:
                polaroid = Image.fromarray(processed).convert("RGB")
                polaroid = polaroid.resize((600, 600))
                frame = Image.new("RGB", (650, 750), "white")
                frame.paste(polaroid, (25, 25))
                draw = ImageDraw.Draw(frame)
                draw.text((50, 640), "Image Playground", fill="black")
                st.image(frame, caption="Your Polaroid")
            except Exception as e:
                st.error(f"Error creating polaroid: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Edge Detection & Features ---
elif app_mode == "Edge Detection & Features":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🕵️♂️ Feature Detective</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="edge2")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        st.markdown("### 🎚️ Live Edge Tuner")
        threshold1 = st.slider("Edge Sensitivity", 0, 255, 100, key="edge_low")
        threshold2 = st.slider("Edge Strength", 0, 255, 200, key="edge_high")
        edges = cv2.Canny(image_gray, threshold1, threshold2)
        
        glow_edges = np.zeros_like(image)
        if image.ndim == 3:
            glow_edges = np.zeros_like(image)
            glow_edges[edges > 0] = [0, 255, 255]
            blended = cv2.addWeighted(image, 0.7, glow_edges, 0.3, 0)
        else:
            glow_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            blended = glow_edges
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(edges, caption="Pure Edges", use_column_width=True)
        with col2:
            st.image(blended, caption="Glowing Overlay", use_column_width=True)
        
        st.markdown("### 🔍 SIFT Feature Detection")
        try:
            keypoints_img = detect_sift_features(image_gray)
            st.image(keypoints_img, caption="Detected Features", use_column_width=True)
        except Exception as e:
                st.error(f"Error detecting features: {e}")
                # Fallback to showing the grayscale image
                st.image(image_gray, caption="Feature detection failed", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Transforms & Frequency Domain ---
elif app_mode == "Transforms & Frequency Domain":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔮 Frequency Domain Explorer</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="freq")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        try:
            st.pyplot(display_fourier_transform(image_gray))
            
            st.markdown("### 🎛️ Frequency Filter Simulator")
            filter_radius = st.slider("Low Pass Filter Radius", 1, 100, 30)
            
            # Create a low pass filter mask
            rows, cols = image_gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.uint8)
            cv2.circle(mask, (ccol, crow), filter_radius, 1, -1)
            
            # Apply FFT and filter
            f = np.fft.fft2(image_gray)
            fshift = np.fft.fftshift(f)
            f_filtered = fshift * mask
            f_filtered_shift = np.fft.ifftshift(f_filtered)
            filtered_img = np.fft.ifft2(f_filtered_shift)
            filtered_img = np.abs(filtered_img).clip(0, 255).astype(np.uint8)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_gray, caption="Original", use_column_width=True)
            with col2:
                st.image(filtered_img, caption="Low Pass Filtered", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image in frequency domain: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Image Restoration ---
elif app_mode == "Image Restoration":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔄 Image Restoration Lab</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="restore")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        st.markdown("### 🧪 Corruption Simulator")
        
        noise_type = st.selectbox("Apply Corruption", [
            "None", 
            "Salt & Pepper Noise", 
            "Motion Blur", 
            "Compression Artifacts"
        ])
        
        corrupted = image.copy()
        
        try:
            if noise_type == "Salt & Pepper Noise":
                # Apply salt and pepper noise
                noise_intensity = st.slider("Noise Intensity", 0.0, 1.0, 0.05)
                corrupted = apply_noise(image, noise_intensity)
            elif noise_type == "Motion Blur":
                # Apply motion blur
                kernel_size = st.slider("Blur Intensity", 3, 31, 15, step=2)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, :] = 1.0 / kernel_size
                corrupted = cv2.filter2D(image, -1, kernel)
            elif noise_type == "Compression Artifacts":
                # Simulate JPEG compression artifacts
                quality = st.slider("Compression Level", 1, 100, 20)
                # Convert to PIL image for compression simulation
                pil_img = Image.fromarray(image)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                jpeg_img = Image.open(buffer)
                corrupted = np.array(jpeg_img)
        except Exception as e:
            st.error(f"Error applying corruption: {e}")
        
        st.markdown("### 🚑 Restoration Techniques")
        restoration = st.selectbox("Restoration Method", [
            "None", 
            "Median Filter", 
            "Bilateral Filter", 
            "Contrast Enhancement"
        ])
        
        restored = corrupted.copy()
        
        try:
            if restoration == "Median Filter":
                kernel_size = st.slider("Filter Size", 1, 11, 3, step=2)
                if image.ndim == 3:
                    # Apply median filter to each channel
                    for i in range(3):
                        restored[:,:,i] = cv2.medianBlur(corrupted[:,:,i], kernel_size)
                else:
                    restored = cv2.medianBlur(corrupted, kernel_size)
            elif restoration == "Bilateral Filter":
                d = st.slider("Diameter", 1, 15, 5)
                sigma_color = st.slider("Color Sigma", 1, 150, 75)
                sigma_space = st.slider("Space Sigma", 1, 150, 75)
                restored = cv2.bilateralFilter(corrupted, d, sigma_color, sigma_space)
            elif restoration == "Contrast Enhancement":
                # Apply CLAHE for contrast enhancement
                if image.ndim == 3:
                    # Convert to LAB color space for CLAHE
                    lab = cv2.cvtColor(corrupted, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl,a,b))
                    restored = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    restored = clahe.apply(corrupted)
        except Exception as e:
            st.error(f"Error applying restoration: {e}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(corrupted, caption="Corrupted", use_column_width=True)
        with col3:
            st.image(restored, caption="Restored", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Segmentation & Representation ---
elif app_mode == "Segmentation & Representation":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🧩 Object Segmentation</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="seg")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        st.markdown("### 🎯 Interactive Thresholding")
        threshold_value = st.slider("Threshold Value", 0, 255, 127)
        _, thresholded = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_gray, caption="Original Grayscale", use_column_width=True)
        with col2:
            st.image(thresholded, caption="Thresholded", use_column_width=True)
        
        st.markdown("### 🧠 Advanced Segmentation")
        segmentation_method = st.selectbox("Segmentation Method", [
            "Otsu's Method", 
            "K-means Clustering", 
            "Watershed Algorithm"
        ])
        
        try:
            if segmentation_method == "Otsu's Method":
                # Apply Otsu's thresholding
                blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
                _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Colorize the segmentation result
                if image.ndim == 3:
                    colored_segmentation = np.zeros_like(image)
                    colored_segmentation[otsu == 255] = [0, 255, 0]  # Green for foreground
                    colored_segmentation[otsu == 0] = [255, 0, 0]    # Red for background
                else:
                    colored_segmentation = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
                
                st.image(colored_segmentation, caption="Otsu's Method Segmentation", use_column_width=True)
                
            elif segmentation_method == "K-means Clustering":
                # Reshape the image for K-means
                if image.ndim == 3:
                    vectorized = image.reshape((-1, 3))
                else:
                    vectorized = image_gray.reshape((-1, 1))
                    
                vectorized = np.float32(vectorized)
                
                # Define criteria and apply K-means
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = st.slider("Number of Clusters", 2, 8, 3)
                _, labels, centers = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Convert back to uint8
                centers = np.uint8(centers)
                segmented_image = centers[labels.flatten()]
                
                # Reshape back to the original image shape
                if image.ndim == 3:
                    segmented_image = segmented_image.reshape((image.shape))
                else:
                    segmented_image = segmented_image.reshape((image_gray.shape))
                    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)
                
                st.image(segmented_image, caption=f"K-means Segmentation (k={k})", use_column_width=True)
                
            elif segmentation_method == "Watershed Algorithm":
                # Apply Watershed algorithm
                # Use Otsu's thresholding first
                blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Noise removal
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                
                # Sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=3)
                
                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
                
                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                
                # Marker labelling
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                
                # Apply watershed
                if image.ndim == 3:
                    markers = cv2.watershed(image, markers)
                    # Create a colored visualization
                    watershed_vis = image.copy()
                    watershed_vis[markers == -1] = [255, 0, 0]  # Red for boundary
                else:
                    # Convert to color for watershed
                    color_img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
                    markers = cv2.watershed(color_img, markers)
                    watershed_vis = color_img.copy()
                    watershed_vis[markers == -1] = [255, 0, 0]  # Red for boundary
                
                st.image(watershed_vis, caption="Watershed Segmentation", use_column_width=True)
        except Exception as e:
            st.error(f"Error applying segmentation: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Page: Shape Analysis ---
elif app_mode == "Shape Analysis":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔍 Shape Analysis Lab</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="shape")
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if unlock_achievement('first_upload'):
            st.success("📸 Achievement Unlocked: First Upload!")
            
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        st.markdown("### 🔢 Shape Detection")
        # Preprocess the image for shape detection
        _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create a copy of the original for drawing on
        if image.ndim == 3:
            shape_image = image.copy()
        else:
            shape_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        
        min_area = st.slider("Minimum Contour Area", 10, 10000, 500)
        
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            cv2.drawContours(shape_image, filtered_contours, -1, (0, 255, 0), 2)
            shape_info = []
            for i, cnt in enumerate(filtered_contours):
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(shape_image, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(shape_image, str(i+1), (cx-10, cy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    epsilon = 0.04 * perimeter
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    shape_name = "Unknown"
                    if len(approx) == 3:
                        shape_name = "Triangle"
                    elif len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        if 0.95 <= aspect_ratio <= 1.05:
                            shape_name = "Square"
                        else:
                            shape_name = "Rectangle"
                    elif len(approx) == 5:
                        shape_name = "Pentagon"
                    elif len(approx) == 6:
                        shape_name = "Hexagon"
                    elif len(approx) > 6:
                        # Check if it's a circle
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.8:
                            shape_name = "Circle"
                        else:
                            shape_name = f"Polygon ({len(approx)} sides)"
                    
                    # Add to shape info
                    shape_info.append({
                        "id": i+1,
                        "shape": shape_name,
                        "area": int(area),
                        "perimeter": int(perimeter),
                        "vertices": len(approx)
                    })
            
            st.image(shape_image, caption="Detected Shapes", use_column_width=True)
            
            if shape_info:
                st.markdown("### 📋 Shape Analysis Results")
                for shape in shape_info:
                    st.markdown(f"""
                    **Shape {shape['id']}**: {shape['shape']}  
                    Area: {shape['area']} px², Perimeter: {shape['perimeter']} px, Vertices: {shape['vertices']}
                    """)
            else:
                st.info("No shapes detected with the current settings. Try adjusting the minimum area.")
                
        except Exception as e:
            st.error(f"Error analyzing shapes: {e}")
    st.markdown('</div>', unsafe_allow_html=True)