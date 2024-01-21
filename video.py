from flask import Flask, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'Content-Type' not in request.headers or not request.headers['Content-Type'].startswith('video/'):
        return 'Invalid content type', 400

    data = request.get_data()

    if not data:
        return 'No data received', 400
    
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'video.mp4'

    # Generate a filename or define your own logic here

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as file:
        file.write(data)
    
    
        
    

    return 'File uploaded', 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
