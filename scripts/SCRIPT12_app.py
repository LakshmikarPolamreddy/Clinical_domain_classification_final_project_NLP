from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import time

app = Flask(__name__)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('tokenizer', cdlocal_files_only=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=40)
model_weights_path = 'weights/best_model.pt'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

label_encoder = LabelEncoder()

# Define batch size
batch_size = 8
class_name_dict = {
    0: 'Allergy / Immunology',
    1: 'Autopsy',
    2: 'Bariatrics',
    3: 'Cardiovascular / Pulmonary',
    4: 'Chiropractic',
    5: 'Consult - History and Phy.',
    6: 'Cosmetic / Plastic Surgery',
    7: 'Dentistry',
    8: 'Dermatology',
    9: 'Diets and Nutritions',
    10: 'Discharge Summary',
    12: 'Emergency Room Reports',
    13: 'Endocrinology',
    11: 'ENT - Otolaryngology',
    14: 'Gastroenterology',
    15: 'General Medicine',
    16: 'Hematology - Oncology',
    17: 'Hospice - Palliative Care',
    18: 'IME-QME-Work Comp etc.',
    19: 'Lab Medicine - Pathology',
    20: 'Letters',
    21: 'Nephrology',
    22: 'Neurology',
    23: 'Neurosurgery',
    24: 'Obstetrics / Gynecology',
    25: 'Office Notes',
    26: 'Ophthalmology',
    27: 'Orthopedic',
    28: 'Pain Management',
    29: 'Pediatrics - Neonatal',
    30: 'Physical Medicine - Rehab',
    31: 'Podiatry',
    32: 'Psychiatry / Psychology',
    33: 'Radiology',
    34: 'Rheumatology',
    36: 'Sleep Medicine',
    35: 'SOAP / Chart / Progress Notes',
    37: 'Speech - Language',
    38: 'Surgery',
    39: 'Urology'
}

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    texts = [item['text'] for item in data]

    encodings = tokenizer.batch_encode_plus(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    _, predicted_labels = torch.max(outputs.logits, 1)
    predicted_labels = predicted_labels.tolist()

    # Convert int64 labels to class names
    predicted_class_names = [class_name_dict[label] for label in predicted_labels]

    results = [{'text': text, 'predicted_label': predicted_label} for text, predicted_label in zip(texts, predicted_class_names)]
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)