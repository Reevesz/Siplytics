import cv2
import numpy as np
import os
from collections import Counter
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# REFERENCE COLORS
# ============================================================

REFERENCES = {
    "French Mole": {"ml": 0, "rgb": (69, 53, 46)},
    "Smokey Topaz": {"ml": 5, "rgb": (104, 64, 46)},
    "Iron Brew": {"ml": 8, "rgb": (148, 99, 64)},
    "Whippets Paw": {"ml": 10, "rgb": (156, 92, 51)},
    "Towie Tan": {"ml": 17.5, "rgb": (200, 124, 69)},
    "Action Man": {"ml": 23, "rgb": (203, 143, 98)},
    "Caramelised Parsnip": {"ml": 35, "rgb": (225, 190, 122)},
    "Dorset Sand": {"ml": 60, "rgb": (231, 197, 147)},
    "Skimmed Alive": {"ml": 85, "rgb": (233, 212, 177)}
}


# ============================================================
# RGB -> LAB
# ============================================================

def rgb_to_lab(rgb):

    pixel = np.uint8([[rgb]])

    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)

    return lab[0, 0].astype(float)


REFERENCE_LAB = {
    name: rgb_to_lab(data["rgb"])
    for name, data in REFERENCES.items()
}


# ============================================================
# GET DOMINANT COLORS
# ============================================================

def get_dominant_colors(image, number_of_colors=8):

    img = cv2.resize(image, (800, 600))

    h, w = img.shape[:2]

    roi = img[
        int(h * 0.25):int(h * 0.75),
        int(w * 0.25):int(w * 0.75)
    ]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(hsv)

    mask = (
        (V < 240) &
        (V > 25) &
        (S > 30)
    )

    pixels = roi[mask]

    if len(pixels) < 100:
        raise ValueError("Not enough tea pixels detected.")

    # BGR to RGB
    pixels = pixels[:, ::-1]

    # Quantization
    pixels = (pixels // 8) * 8

    colors = [tuple(pixel) for pixel in pixels]

    counts = Counter(colors)

    total = len(colors)

    dominant = []

    for color, count in counts.most_common(number_of_colors):

        percentage = (count / total) * 100

        dominant.append({
            "rgb": color,
            "percentage": percentage
        })

    return dominant


# ============================================================
# TEA COLOR ANALYZER
# ============================================================

def analyze_tea(image):

    dominant = get_dominant_colors(image, number_of_colors=10)

    scores = {
        name: 0.0
        for name in REFERENCES
    }

    for item in dominant:

        rgb = item["rgb"]
        percentage = item["percentage"]

        lab = rgb_to_lab(rgb)

        for name, reference_lab in REFERENCE_LAB.items():

            distance = np.linalg.norm(
                lab - reference_lab
            )

            similarity = np.exp(-distance / 25)

            scores[name] += similarity * percentage

    best = max(scores, key=scores.get)

    result = REFERENCES[best]

    total_score = sum(scores.values())

    confidence = (
        scores[best] / total_score
    ) * 100

    return {
        "reference": best,
        "milk_ml": result["ml"],
        "confidence": round(confidence, 1),
        "dominant_colors": dominant
    }


# ============================================================
# BUBBLE DETECTOR
# ============================================================

def detect_bubbles(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=8,
        param1=50,
        param2=15,
        minRadius=2,
        maxRadius=25
    )

    output = image.copy()

    count = 0

    if circles is not None:

        circles = np.uint16(np.around(circles))

        count = len(circles[0])

        for x, y, r in circles[0]:

            cv2.circle(
                output,
                (x, y),
                r,
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (x, y),
                2,
                (0, 0, 255),
                3
            )

    return count, output


# ============================================================
# FROTH RATING
# ============================================================

def get_froth_rating(bubble_count):

    if bubble_count < 10:
        return "Low Froth"

    elif bubble_count < 30:
        return "Moderate Froth"

    elif bubble_count < 60:
        return "Good Froth"

    else:
        return "Very Frothy"


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No image uploaded.")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        image = cv2.imread(filepath)

        if image is None:
            return render_template("index.html", error="Could not read image.")

        tea_result = analyze_tea(image)
        bubble_count, processed_image = detect_bubbles(image)
        froth_rating = get_froth_rating(bubble_count)

        verdict = get_overall_message(
            tea_result,
            bubble_count,
            froth_rating
        )

        processed_filename = "processed_" + filename
        processed_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            processed_filename
        )

        cv2.imwrite(processed_path, processed_image)

        result = {
            "tea": tea_result,
            "bubble_count": bubble_count,
            "froth_rating": froth_rating,
            "verdict": verdict
        }

        return render_template(
            "index.html",
            result=result,
            image_url="/static/uploads/" + processed_filename
        )

    return render_template("index.html")

def get_overall_message(tea_result, bubble_count, froth_rating):
    milk_ml = tea_result["milk_ml"]
    confidence = tea_result["confidence"]

    if milk_ml < 20:
        return {
            "title": "THIS TEA IS ENGLISH CERTIFIED",
            "message": "Strong, dark, and dangerously serious. Even the Queen would approve.",
            "class": "excellent"
        }

    elif milk_ml < 40:
        return {
            "title": "TEA HAS ACHIEVED INNER PEACE",
            "message": "A respectable cup. Not too weak, not too aggressive. Scientifically drinkable.",
            "class": "good"
        }

    elif milk_ml < 70:
        return {
            "title": "THIS TEA IS HAVING AN IDENTITY CRISIS",
            "message": "Somewhere between tea and milk. It needs to make a decision.",
            "class": "average"
        }

    else:
        return {
            "title": "THIS TEA IS SHIT. THROW IT OUT.",
            "message": "The milk has won. The tea has officially surrendered.",
            "class": "bad"
        }

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )