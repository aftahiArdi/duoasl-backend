from flask import Flask, request, send_from_directory
import zipfile
import os
import shutil

app = Flask(__name__)

# Directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file and file.filename.endswith('.zip'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Unzipping the file directly into the UPLOAD_FOLDER
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)
            
        # for item in os.listdir(UPLOAD_FOLDER):
        #     if not item.endswith('.mp4'):
        #         os.remove(os.path.join(UPLOAD_FOLDER, item))
        
        for item in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and all its contents
            elif not item.endswith('.mp4'):
                os.remove(item_path)  # Remove the file if it's not .mp4


        return 'File uploaded and unzipped successfully', 200
    else:
        return 'Invalid file format', 400

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
