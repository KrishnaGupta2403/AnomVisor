from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from model.ae_transformer import AETransformer
from model.utils import preprocess_data
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up folder for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mat'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained model (make path configurable)
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'ae_transformer_model.pth')  # Absolute path to model file
model = None
try:
    # Update input_dim to 40 to match PCA components in preprocess_data
    input_dim = 40
    latent_dim = 32
    model = AETransformer(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(f"AETransformer model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading AETransformer model from {MODEL_PATH}: {e}")

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check for both files in the request
    if 'hsi_file' not in request.files or 'gt_file' not in request.files:
        return jsonify({'error': 'Both hsi_file and gt_file must be provided'}), 400

    hsi_file = request.files['hsi_file']
    gt_file = request.files['gt_file']

    if hsi_file.filename == '' or gt_file.filename == '':
        return jsonify({'error': 'No selected file(s)'}), 400

    dataset_name = request.form.get('dataset_name')
    if not dataset_name:
        return jsonify({'error': 'No dataset_name provided'}), 400

    if allowed_file(hsi_file.filename) and allowed_file(gt_file.filename):
        hsi_filename = secure_filename(hsi_file.filename)
        gt_filename = secure_filename(gt_file.filename)

        hsi_path = os.path.join(app.config['UPLOAD_FOLDER'], hsi_filename)
        gt_path = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)

        try:
            hsi_file.save(hsi_path)
            gt_file.save(gt_path)
            print(f"HSI file saved at {hsi_path}")
            print(f"GT file saved at {gt_path}")

            # Import the new pipeline utility function
            from utils import run_pipeline_with_files

            # Run the pipeline with uploaded files
            results = run_pipeline_with_files(hsi_path, gt_path, dataset_name)

            return jsonify({'message': 'Files uploaded and processed successfully', 'results': results})
        except Exception as e:
            print(f"Error during file processing: {e}")
            return jsonify({'error': f'File processing failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'input_data' not in data:
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        # Convert input data to torch tensor
        input_data = data['input_data']
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Reshape input_tensor to 2D (num_samples, input_dim)
        if input_tensor.ndim == 3:
            h, w, c = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, c)
        elif input_tensor.ndim == 2:
            # Already 2D, no change
            pass
        else:
            return jsonify({'error': 'Input data has invalid shape'}), 400

        print(f"Input tensor shape for prediction: {input_tensor.shape}")

        x_recon, z_trans = model.predict(input_tensor)  # Get reconstructed output and latent
        # Compute reconstruction error (MSE) per sample
        recon_error = torch.mean((input_tensor - x_recon) ** 2, dim=1)
        prediction_list = recon_error.tolist()

        print(f"Reconstruction error shape: {recon_error.shape}")
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debugging output
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'prediction': prediction_list})


from model.utils import load_test_data, evaluate_ae_model

@app.route('/metrics', methods=['GET'])
def metrics():
    dataset_name = 'pavia'  # or get from config or request args
    test_file_path = 'backend/data/pavia.mat'  # Update with actual test dataset path

    try:
        test_data, test_labels = load_test_data(test_file_path, dataset_name)
        accuracy, confusion_mat = evaluate_ae_model(model, test_data, test_labels)
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': confusion_mat
        }
    except Exception as e:
        return jsonify({'error': f'Failed to compute metrics: {str(e)}'}), 500

    return jsonify(metrics)


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Backend is connected'})

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Backend is running'})

if __name__ == '__main__':
    app.run(debug=True)
