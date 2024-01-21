from flask import Flask
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor




app = Flask(__name__)

@app.route('/run_notebook')
def run_notebook():
    
    
    
    # Load the notebook
    with open('test.ipynb') as f:
        nb = nbformat.read(f, as_version=4)

    # Run the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb)

    # Extract outputs from each code cell
    output_data = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell.outputs:
                if output.output_type == 'stream':
                    output_data.append(output.text)
                elif output.output_type == 'execute_result':
                    output_data.append(output.data.get('text/plain', ''))
                    

    # Join all outputs into a single string
    full_output = "\n".join(output_data)

    # Return the output as plain text
    return f"<pre>{full_output}</pre>"

if __name__ == '__main__':
    app.run(debug=True)
