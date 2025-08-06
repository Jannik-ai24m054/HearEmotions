import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from IPython.display import Audio, display
import ipywidgets as widgets
from moviepy.editor import VideoFileClip
import requests 

print("Welcome to HearMyPet - Multi-class Sound Classification (Baby + Animal)")

def extract_mfcc_with_deltas(file_path, n_mfcc=40, include_delta=True, include_delta_delta=True):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = [np.mean(mfcc.T, axis=0)]
    if include_delta:
        delta = librosa.feature.delta(mfcc)
        features.append(np.mean(delta.T, axis=0))
    if include_delta_delta:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(np.mean(delta2.T, axis=0))
    return np.concatenate(features), mfcc, y, sr

# Dataset paths (adjust if needed)
animal_dataset = "/kaggle/input/dog-voice-emotion-dataset"
baby_dataset = "/kaggle/input/baby-crying-sounds-dataset/Baby Crying Sounds"

animal_files, animal_labels = [], []
baby_files, baby_labels = [], []

# Load animal data + prefix labels with 'animal_'
for root, dirs, files in os.walk(animal_dataset):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower().replace(" ", "_")
            animal_files.append(path)
            animal_labels.append(f"animal_{label}")

# Load baby data + prefix labels with 'baby_'
for root, dirs, files in os.walk(baby_dataset):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower().replace(" ", "_")
            baby_files.append(path)
            baby_labels.append(f"baby_{label}")

print(f"Found {len(animal_files)} animal audios, {len(baby_files)} baby audios.")

# Combine files and labels
all_files = animal_files + baby_files
all_labels = animal_labels + baby_labels

combine_classes = {
    "baby_discomfort": "baby_unwell",
    "baby_belly_pain": "baby_unwell",
}
exclude_classes = [
    "baby_cold_hot",
    "baby_tired",
    "baby_burping",
]

filtered_files = []
filtered_labels = []

for f, l in zip(all_files, all_labels):
    if l in exclude_classes:
        continue
    if l in combine_classes:
        l = combine_classes[l]
    filtered_files.append(f)
    filtered_labels.append(l)

print(f"After filtering: {len(filtered_files)} audio files, {len(set(filtered_labels))} classes.")

# Feature extraction
print("Extracting MFCC features (40 MFCC + Delta + Delta-Delta)...")
X = []
for f in filtered_files:
    feat, _, _, _ = extract_mfcc_with_deltas(f, n_mfcc=40, include_delta=True, include_delta_delta=True)
    X.append(feat)
X = np.array(X)
y = np.array(filtered_labels)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute class weights
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))

# Classifier
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
example_report = None

print(" Starting 5-fold Stratified Cross-Validation...")
for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_acc.append(acc)

    if example_report is None:
        example_report = classification_report(y_test, y_pred, zero_division=0)

print(f" Average accuracy (5-fold CV): {np.mean(cv_acc):.3f}")
print(" Sample classification report:")
print(example_report)

# Final training on all data
print(" Training final classifier on full dataset...")
clf.fit(X_scaled, y)

# Save model and scaler
output_dir = "./hearemotions_combined_model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(clf, os.path.join(output_dir, "combined_sound_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "combined_scaler.pkl"))
print(f"Model and scaler saved to: {output_dir}")

def clean_label(label):
    if label.startswith("animal_"):
        label = label[len("animal_"):]
    elif label.startswith("baby_"):
        label = label[len("baby_"):]
    for suffix in ["_test", "_train"]:
        if label.endswith(suffix):
            label = label[:-len(suffix)]
    return label

def mock_gemma_response(prediction_label):
    explanations = {
        "dog_bark": "The dog is barking.",
        "dog_growl": "The dog is growling.",
        "dog_grunt": "The dog is grunting.",
        "cat": "The cat is making sounds.",
        "unwell": "The baby feels unwell.",
        "hungry": "The baby is hungry.",
        "laugh": "The baby is laughing – joy!",
        "silence": "No sound detected.",
        "discomfort": "The baby is uncomfortable.",
        "belly_pain": "The baby has stomach pain.",
        "cry": "The baby is crying.",
        "whine": "The baby is whining.",
        # add more explanations here
    }
    cleaned_label = clean_label(prediction_label)
    if cleaned_label in explanations:
        return explanations[cleaned_label]
    else:
        print(f"No explanation found for label: {cleaned_label}")
        print(f"Available labels: {list(explanations.keys())}")
        return "No explanation available."

emoji_dict = {
    "hungry": "🍽️👶",
    "unwell": "🤒👶",
    "laugh": "😄👶",
    "silence": "🔇",
    "bark": "🐶🔊",
    "growl": "🐶😠",
    "grunt": "🐶💨",
    "dog_bark": "🐶🔊",
    "dog_growl": "🐶😠",
    "dog_grunt": "🐶💨",
    "cat": "🐱",
    "cry": "😭👶",
    "whine": "😢👶",
}

# --- Random audio demo ---
random_audio = random.choice(filtered_files)
print(f"\n Randomly selected file for prediction:\n{random_audio}")

try:
    display(Audio(random_audio))
except:
    print("Audio playback not available.")

mfcc_example, mfcc_matrix, y_audio, sr_audio = extract_mfcc_with_deltas(random_audio, n_mfcc=40, include_delta=True, include_delta_delta=True)
mfcc_example_scaled = scaler.transform(mfcc_example.reshape(1, -1))
predicted_label = clf.predict(mfcc_example_scaled)[0]
predicted_proba = clf.predict_proba(mfcc_example_scaled)[0]
confidence = np.max(predicted_proba)

print("\n HearMyPet Result:")
print(f"Detected sound: {predicted_label.replace('_', ' ')}")
print(f"Meaning: {mock_gemma_response(predicted_label)}")
print(f"Emoji: {emoji_dict.get(clean_label(predicted_label), '')}")
print(f"Prediction confidence: {confidence:.2f}")

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_matrix, sr=sr_audio, x_axis='time')
plt.colorbar()
plt.title('MFCC Heatmap of selected audio file')
plt.tight_layout()
plt.show()

# Gemma 3n API integration (commented out)
"""
# To enable Gemma 3n explanations, first add your API key to Kaggle Secrets as 'GEMMA3N_API_KEY'
import os
import requests

API_KEY = os.environ.get('GEMMA3N_API_KEY')

if API_KEY is None:
    print(" API key not found. Please add your Gemma3n API key to Kaggle Secrets.")
else:
    gemma_url = "https://api.gemma3n.example/v1/explain"  # replace with actual endpoint
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "label": predicted_label,
        "language": "en"
    }
    response = requests.post(gemma_url, headers=headers, json=payload)
    if response.status_code == 200:
        explanation = response.json().get("explanation", "")
        print("Gemma 3n explanation:")
        print(explanation)
    else:
        print(f"API error {response.status_code}: {response.text}")
"""

print("\nThank you for using HearMyPet!")
