from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
import numpy as np
import json
import uuid
import os
import base64
import cv2
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

label = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

with open("tamil_translations.json", 'r', encoding='utf-8') as file:
    tamil = json.load(file)

# Supported plants the model can identify
SUPPORTED_PLANTS = [
    {'en': 'Apple', 'ta': 'ஆப்பிள்', 'icon': '🍎'},
    {'en': 'Blueberry', 'ta': 'நீலப்பழம்', 'icon': '🫐'},
    {'en': 'Cherry', 'ta': 'செர்ரி', 'icon': '🍒'},
    {'en': 'Corn', 'ta': 'மக்காச்சோளம்', 'icon': '🌽'},
    {'en': 'Grape', 'ta': 'திராட்சை', 'icon': '🍇'},
    {'en': 'Orange', 'ta': 'ஆரஞ்சு', 'icon': '🍊'},
    {'en': 'Peach', 'ta': 'பீச்', 'icon': '🍑'},
    {'en': 'Pepper', 'ta': 'குடை மிளகாய்', 'icon': '🌶️'},
    {'en': 'Potato', 'ta': 'உருளைக்கிழங்கு', 'icon': '🥔'},
    {'en': 'Raspberry', 'ta': 'ராஸ்ப்பெர்ரி', 'icon': '🫐'},
    {'en': 'Soybean', 'ta': 'சோயா', 'icon': '🫘'},
    {'en': 'Squash', 'ta': 'சுரைக்காய்', 'icon': '🎃'},
    {'en': 'Strawberry', 'ta': 'ஸ்ட்ராபெரி', 'icon': '🍓'},
    {'en': 'Tomato', 'ta': 'தக்காளி', 'icon': '🍅'},
]

CONFIDENCE_THRESHOLD = 40  # Below this = not recognized


def format_label(raw_label):
    """Format raw label to human-readable."""
    parts = raw_label.split('___')
    if len(parts) == 2:
        plant = parts[0].replace('_', ' ').strip()
        disease = parts[1].replace('_', ' ').strip()
        return f"{plant} — {disease}"
    return raw_label.replace('_', ' ').strip()


def get_plant_name(raw_label):
    parts = raw_label.split('___')
    return parts[0].replace('_', ' ').strip()


def get_disease_name(raw_label):
    parts = raw_label.split('___')
    if len(parts) == 2:
        return parts[1].replace('_', ' ').strip()
    return 'Unknown'


def get_tamil_data(raw_label, plant_name, disease_name):
    """Get Tamil translations for plant name, disease name, cause, and cure."""
    tamil_plant = tamil['plants'].get(plant_name, plant_name)
    tamil_disease = tamil['diseases'].get(disease_name, disease_name)
    tamil_cause = tamil['cause'].get(raw_label, '')
    tamil_cure = tamil['cure'].get(raw_label, '')
    return {
        'plant': tamil_plant,
        'disease': tamil_disease,
        'cause': tamil_cause,
        'cure': tamil_cure,
    }


def is_leaf_image(image_path):
    """
    Check if the image likely contains a plant leaf.
    Uses HSV color space to detect green/plant-like content.
    Returns True if image looks like it contains plant material.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]

    # Detect green-ish plant pixels (broad range including yellow-green, dark green)
    lower_plant = np.array([15, 20, 20])
    upper_plant = np.array([95, 255, 255])
    plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)

    # Also detect brown/tan (dried leaves, diseased tissue)
    lower_brown = np.array([5, 20, 20])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    plant_mask = cv2.bitwise_or(plant_mask, brown_mask)
    plant_pixels = cv2.countNonZero(plant_mask)
    plant_ratio = plant_pixels / total_pixels

    # At least 5% of image should look like plant material
    return plant_ratio >= 0.05


def analyze_disease_severity(image_path):
    """
    Analyze the actual disease-affected area of the leaf using image processing.
    Uses HSV color segmentation to detect healthy green tissue vs.
    diseased tissue (brown, yellow, black spots, lesions).
    Returns the percentage of leaf area that is affected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Isolate the leaf from background
    # Detect leaf pixels (non-white, non-very-dark background)
    lower_leaf = np.array([0, 20, 30])
    upper_leaf = np.array([180, 255, 245])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Remove very bright white/grey background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    not_bg = cv2.inRange(gray, 15, 240)
    leaf_mask = cv2.bitwise_and(leaf_mask, not_bg)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    if total_leaf_pixels < 100:
        return 0.0

    # Step 2: Detect healthy green areas
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_on_leaf = cv2.bitwise_and(green_mask, leaf_mask)
    healthy_pixels = cv2.countNonZero(green_on_leaf)

    # Step 3: Detect diseased areas (brown, yellow, black spots, lesions)
    # Brown/tan regions
    lower_brown = np.array([8, 30, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spots/lesions (very dark on leaf)
    lower_dark = np.array([0, 0, 15])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Yellow/chlorosis
    lower_yellow = np.array([18, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White/powdery mildew areas
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine all diseased masks
    diseased_mask = cv2.bitwise_or(brown_mask, dark_mask)
    diseased_mask = cv2.bitwise_or(diseased_mask, yellow_mask)
    diseased_mask = cv2.bitwise_or(diseased_mask, white_mask)

    # Only count diseased pixels that are on the leaf
    diseased_on_leaf = cv2.bitwise_and(diseased_mask, leaf_mask)
    diseased_pixels = cv2.countNonZero(diseased_on_leaf)

    # Calculate percentage
    affected_pct = (diseased_pixels / total_leaf_pixels) * 100
    affected_pct = min(affected_pct, 100.0)

    return round(affected_pct, 1)


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image):
    """Run prediction and return disease info + confidence data."""
    # Step 1: Check if image looks like a plant leaf
    leaf_check = is_leaf_image(image)

    if not leaf_check:
        return {
            'prediction': None,
            'confidence': 0,
            'plant_name': '',
            'disease_name': '',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': '', 'disease': '', 'cause': '', 'cure': ''},
            'is_valid': False,
            'error_message': 'This image does not appear to contain a plant leaf. Please upload a clear photo of a leaf.',
            'error_tamil': 'இந்த படத்தில இலை இருக்க மாதிரி தெரியல. தயவுசெய்து ஒரு இலையோட படத்தை போடுங்க.',
        }

    try:
        img = extract_features(image)
        prediction = model.predict(img)
        probabilities = prediction[0]
    except Exception as e:
        return {
            'prediction': None,
            'confidence': 0,
            'plant_name': '',
            'disease_name': '',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': '', 'disease': '', 'cause': '', 'cure': ''},
            'is_valid': False,
            'error_message': 'Uploaded image file is corrupted or unsupported. Please use JPG/PNG.',
            'error_tamil': 'பதிவேற்றப்பட்ட படம் சேதமடைந்துள்ளது அல்லது ஆதரிக்கப்படவில்லை. தயவுசெய்து JPG/PNG படத்தை போடுங்க.',
        }


    top_index = probabilities.argmax()
    top_confidence = float(probabilities[top_index]) * 100
    prediction_label = plant_disease[top_index]
    raw_label = label[top_index]

    plant_name = get_plant_name(raw_label)
    disease_name = get_disease_name(raw_label)
    is_healthy = 'healthy' in raw_label.lower()

    # Step 2: Determine if prediction is valid
    is_valid = True
    error_message = ''
    error_tamil = ''

    if raw_label == 'Background_without_leaves':
        is_valid = False
        error_message = 'No plant leaf detected in this image. Please upload a clear photo of a leaf.'
        error_tamil = 'இந்த படத்தில இலை கண்டுபிடிக்க முடியல. தயவுசெய்து ஒரு இலையோட படத்தை போடுங்க.'
    elif top_confidence < CONFIDENCE_THRESHOLD:
        is_valid = False
        error_message = f'Could not identify this plant with enough certainty (confidence: {round(top_confidence, 1)}%). The image may not be a supported plant.'
        error_tamil = f'இந்த செடியை நம்பகமா கண்டுபிடிக்க முடியல (நம்பகத்தன்மை: {round(top_confidence, 1)}%). இது நாங்க ஆராய்ச்சி செய்யற செடியா இருக்காது.'

    # Tamil translations
    tamil_data = get_tamil_data(raw_label, plant_name, disease_name)

    # Real disease severity from image analysis
    severity = 0.0
    if is_valid:
        severity = analyze_disease_severity(image)
        if is_healthy:
            severity = 0.0

    return {
        'prediction': prediction_label,
        'confidence': round(top_confidence, 2),
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy,
        'severity': severity,
        'tamil': tamil_data,
        'is_valid': is_valid,
        'error_message': error_message,
        'error_tamil': error_tamil,
    }


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        filepath = f'{temp_name}_{image.filename}'
        image.save(filepath)

        result = model_predict(f'./{filepath}')

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{filepath}',
            prediction=result['prediction'],
            confidence=result['confidence'],
            plant_name=result['plant_name'],
            disease_name=result['disease_name'],
            is_healthy=result['is_healthy'],
            severity=result['severity'],
            tamil=result['tamil'],
            is_valid=result['is_valid'],
            error_message=result['error_message'],
            error_tamil=result['error_tamil'],
            supported_plants=SUPPORTED_PLANTS,
        )
    else:
        return redirect('/')


@app.route('/upload-camera/', methods=['POST'])
def upload_camera():
    """Handle camera capture — receives base64 image data."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    filepath = f"uploadimages/camera_{uuid.uuid4().hex}.jpg"

    with open(filepath, 'wb') as f:
        f.write(img_bytes)

    result = model_predict(f'./{filepath}')

    return jsonify({
        'success': True,
        'imagepath': f'/{filepath}',
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'plant_name': result['plant_name'],
        'disease_name': result['disease_name'],
        'is_healthy': result['is_healthy'],
        'severity': result['severity'],
        'tamil': result['tamil'],
        'is_valid': result['is_valid'],
        'error_message': result['error_message'],
        'error_tamil': result['error_tamil'],
        'supported_plants': SUPPORTED_PLANTS,
    })


if __name__ == "__main__":
    app.run(debug=True)
