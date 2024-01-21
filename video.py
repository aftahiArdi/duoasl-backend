from flask import Flask, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import numpy as np




import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


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
    filename = f'video.mp4'

    # Generate a filename or define your own logic here

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as file:
        file.write(data)
        

        
    asl_translation = {
        "1": "hello",
        "2":"thanks",
        "3":"I love you",
    }

    with open('ml_model/use_model.ipynb') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    #print("nb!!!!!!", nb)
    # # Run the notebook
    ep = ExecutePreprocessor(timeout=600000)
    #print(ep)
    try:
        ep.preprocess(nb, {'metadata': {'path': 'ml_model/'}})
    except Exception as e:
        print("hello", e)

    # # Extract outputs from each code cell
    output_data = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell.outputs:
                if output.output_type == 'stream':
                    output_data.append(output.text)
                elif output.output_type == 'execute_result':
                    output_data.append(output.data.get('text/plain', ''))

    print(output_data)
                    
    # # Join all outputs into a single string

    array = np.load("ml_model/outputs/test.npy").tolist()
    print(type(array))
    # print # of thanks and hello 
    count_hello = array.count("hello")
    count_thanks = array.count("thanks")
    count_love = array.count("iloveyou")

    max_count = max(count_hello, count_thanks, count_love)
    print(count_hello, count_thanks, count_love)
    if max_count == count_hello and id == "1":
        return {"success": True}, 200
    elif max_count == count_thanks and id == "2":
        return {"success": True}, 200
    elif max_count == count_love and id == "3":
        return {"success": True}, 200

    
    return {"success": False}, 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
