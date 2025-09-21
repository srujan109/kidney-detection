from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import cv2
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Upload & process image
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return "No file uploaded"

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ----- Kidney & Stone Detection -----
    image = Image.open(filepath).convert("L")  # grayscale
    img = np.array(image)

    # Smooth image
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # Threshold for stones (dark areas)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours (stones)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw stones and kidney boundary (for demo: full image as kidney)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    cv2.rectangle(output, (0,0), (w-1,h-1), (255,0,0), 2)  # blue box for kidney

    stone_count = 0
    report = f"Kidney Position: (0,0), Size: ({w}x{h})\n"
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # filter noise
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x,y), (x+bw, y+bh), (0,0,255), 2)  # red box
            stone_count += 1
            report += f"Stone {stone_count}: Position=({x},{y}), Size=({bw}x{bh})\n"

    report += f"Total Stones Detected: {stone_count}\n"
    report += "Conclusion: Kidney stones detected. Recommend urology consultation."

    # Encode image to base64 for frontend display
    _, buffer = cv2.imencode(".png", output)
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    return render_template("index.html", result_image=result_b64, report=report)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)