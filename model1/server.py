from flask import Flask, request, jsonify
import json
import os
from model import owlv2_results

app = Flask(__name__)

# Read the output file path from environment variable
RESULT_FILE_PATH = os.getenv("RESULT_FILE_PATH", "/data/results.json")

result_dir = os.path.dirname(RESULT_FILE_PATH)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "image_url" not in data or "query" not in data:
        return jsonify({"error": "Missing image_url or query in request"}), 400

    image_url = data["image_url"]
    query = data["query"]
    #print(image_url, query)
    
    results = owlv2_results(image_url, query)
    #print(results)
    # Simulate model processing (replace this with your owl v2 logic)
    result = {
        "image_url": image_url,
        "query": query,
        "prediction": results.prediction,
        "confidence": results.confidence
    }

    # Write the result to the external volume as a JSON file
    try:
        if not os.path.exists(RESULT_FILE_PATH):
            with open(RESULT_FILE_PATH, 'w') as f:
                json.dump(result, f)
        else:
            # If the file exists, append the result to it (if you need this functionality)
            with open(RESULT_FILE_PATH, 'a') as f:
                f.write(json.dumps(result) + "\n")
                
        return jsonify({"status": "success", "result": result}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
