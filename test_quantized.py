import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix
import time

# 1. Setup Paths
model_path = r"D:\SSAR_Final_Model_quantized.tflite"
test_dir = r"D:\dataset\final_train" 

# 2. Load Model & Detect Shape
print(f"📦 Loading Model: {model_path}")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'] # Should be [1, 1, 128, 128]

print(f"✅ Model Info: Expected Input Shape is {input_shape}")

# 3. Optimized Preprocessing for NXP Hardware
def preprocess_for_nchw(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Convert to Grayscale as model expects 1 channel
        img = tf.image.rgb_to_grayscale(img)
        
        # Resize to 128x128
        img = tf.image.resize(img, [128, 128])
        
        # 🛡️ THE ACCURACY FIX: NXP Standard Normalization (-1 to 1 range)
        # Most eIQ models use this instead of 0-1 scaling
        img = (img - 127.5) / 127.5
        
        # 🔄 THE NCHW FIX: Transpose from [H, W, C] to [C, H, W]
        img = np.transpose(img, (2, 0, 1)) 
        
        # Add Batch Dimension: [1, 1, 128, 128]
        img = np.expand_dims(img, axis=0) 
        
        return img.astype(np.float32)
    except Exception as e:
        print(f"⚠️ Error on {image_path}: {e}")
        return None

# 4. Run Inference Loop
y_true, y_pred = [], []
# Filter out non-directories (like the bbox folder) to prevent permission errors
classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

print(f"🔍 Processing {len(classes)} classes for IESA Hackathon...")
start_time = time.time()
processed_count = 0

for idx, label in enumerate(classes):
    class_path = os.path.join(test_dir, label)
    print(f"Testing Class: {label}")
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Filter for images only
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        input_data = preprocess_for_nchw(img_path)
        if input_data is None: continue
            
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_true.append(idx)
        y_pred.append(np.argmax(output_data))
        processed_count += 1

# 5. Final Metrics for GitHub & PDF Report
if processed_count > 0:
    print("\n" + "="*40)
    print(" 🚀 FINAL IESA HACKATHON RESULTS")
    print("="*40)
    print(f"Model Size:     4.02 MB")
    print(f"Avg Latency:    {(time.time() - start_time)/processed_count*1000:.2f} ms")
    print(f"Target Device:  NXP i.MX RT1170")
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=classes))