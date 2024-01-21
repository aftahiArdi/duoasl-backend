from flask import Flask, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload/<id>', methods=['POST'])
def upload_file(id):
    if 'Content-Type' not in request.headers or not request.headers['Content-Type'].startswith('video/'):
        return 'Invalid content type', 400

    data = request.get_data()

    if not data:
        return 'No data received', 400
    
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'uploaded_video_{current_time}.mp4'

    # Generate a filename or define your own logic here

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as file:
        file.write(data)
        

        
    asl_translation = {
        "1": "hello",
        "2":"thanks",
        "3":"I love you",
    }

    
    # Load the notebook
    with open('test.ipynb') as f:
        nb = nbformat.read(f, as_version=4)

    # Run the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb)

    # Extract outputs from each code cell
    output_data = []
    # for cell in nb.cells:
    #     if cell.cell_type == 'code':
    #         for output in cell.outputs:
    #             if output.output_type == 'stream':
    #                 output_data.append(output.text)
    #             elif output.output_type == 'execute_result':
    #                 output_data.append(output.data.get('text/plain', ''))
    output_data.append(nb.cells[2].outputs[0].text)                    

    # Join all outputs into a single string
    full_output = "\n".join(output_data)
    
    
    return 'File uploaded', 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
