from flask import Flask, render_template_string, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)


def apply_filter(img, filter_name):
    if filter_name == 'grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_name == 'blur':
        return cv2.GaussianBlur(img, (15, 15), 0)
    elif filter_name == 'edges':
        return cv2.Canny(img, 100, 200)
    elif filter_name == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(img, kernel)
    elif filter_name == 'black_and_white':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return bw
    else:
        return img


html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Filter Studio</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            width: 100%;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 2.2em;
            font-weight: 300;
            letter-spacing: -1px;
        }

        .upload-section {
            margin-bottom: 30px;
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
            cursor: pointer;
            margin-bottom: 20px;
        }

        .file-input {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }

        .file-input-button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            border: none;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            min-width: 200px;
        }

        .file-input-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .file-name {
            margin-top: 10px;
            color: #666;
            font-size: 14px;
            font-style: italic;
        }

        .filter-section {
            margin-bottom: 30px;
        }

        .filter-label {
            display: block;
            color: #555;
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 15px;
        }

        .filter-select {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            background: white;
            color: #333;
            cursor: pointer;
            transition: all 0.3s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 12px center;
            background-repeat: no-repeat;
            background-size: 16px;
        }

        .filter-select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .submit-button {
            background: linear-gradient(45deg, #11998e, #38ef7d);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
            min-width: 200px;
        }

        .submit-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
        }

        .submit-button:active {
            transform: translateY(0);
        }

        .submit-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .filter-preview {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        .filter-option {
            padding: 8px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 12px;
            color: #666;
            text-align: center;
            transition: all 0.2s ease;
        }

        .filter-option:hover {
            background: #e9ecef;
            color: #333;
        }

        .loading {
            display: none;
            margin-top: 20px;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 600px) {
            .container {
                padding: 30px 20px;
                margin: 10px;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            .file-input-button,
            .submit-button {
                min-width: 150px;
                padding: 12px 25px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ Image Filter Studio</h1>
        
        <form method="POST" enctype="multipart/form-data" action="/upload" id="uploadForm">
            <div class="upload-section">
                <div class="file-input-wrapper">
                    <input type="file" name="image" class="file-input" accept="image/*" required id="imageInput">
                    <div class="file-input-button">
                        📸 Choose Image
                    </div>
                </div>
                <div class="file-name" id="fileName"></div>
            </div>

            <div class="filter-section">
                <label class="filter-label">🎨 Choose Your Filter:</label>
                <select name="filter" class="filter-select" id="filterSelect">
                    <option value="grayscale">🔳 Grayscale - Classic monochrome</option>
                    <option value="black_and_white">⚫ Black & White - High contrast</option>
                    <option value="blur">🌫️ Blur - Soft focus effect</option>
                    <option value="edges">⚡ Edge Detection - Artistic outline</option>
                    <option value="sepia">🍂 Sepia - Vintage warmth</option>
                </select>
            </div>

            <button type="submit" class="submit-button" id="submitBtn">
                🚀 Apply Filter
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #666;">Processing your image...</p>
            </div>
        </form>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');

        // Show selected file name
        imageInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                fileName.textContent = `Selected: ${e.target.files[0].name}`;
                fileName.style.color = '#667eea';
            } else {
                fileName.textContent = '';
            }
        });

        // Handle form submission with loading state
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('imageInput');
            if (!fileInput.files.length) {
                e.preventDefault();
                alert('Please select an image first!');
                return;
            }

            // Show loading state
            submitBtn.style.display = 'none';
            loading.style.display = 'block';
        });

        // Add hover effect to file input
        const fileInputWrapper = document.querySelector('.file-input-wrapper');
        fileInputWrapper.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.transform = 'scale(1.02)';
        });

        fileInputWrapper.addEventListener('dragleave', function(e) {
            this.style.transform = 'scale(1)';
        });

        fileInputWrapper.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.transform = 'scale(1)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                imageInput.files = files;
                fileName.textContent = `Selected: ${files[0].name}`;
                fileName.style.color = '#667eea';
            }
        });
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_template)

@app.route("/upload", methods=['POST'])
def upload_image():
    file = request.files.get("image")
    filter_type = request.form.get("filter")

    if file:
        nping = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nping, cv2.IMREAD_COLOR)

        output = apply_filter(img, filter_type)
        output_path = "static/output.jpg"

        if len(output.shape) == 2:
            cv2.imwrite(output_path, output)
        else:
            cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        return send_file(output_path, mimetype='image/jpeg')

    return "No image uploaded or filter not selected", 400

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
