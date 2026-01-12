import os
import csv
from ultralytics import YOLO

# 1. Load your expert model
model = YOLO(r'C:\Users\Admin\runs\detect\pistachio_v2_improved\weights\best.pt')

# 2. Define your paths
input_folder = r'C:\Users\Admin\Pistachio_for_test'  # Folder with images to count
output_csv = 'pistachio_counts.csv'

# 3. Prepare the CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Total Nuts', 'Kirmizi Count', 'Siirt Count'])

    # 4. Loop through every image in the folder
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, img_name)
            
            # Run the AI prediction
            results = model.predict(source=img_path, conf=0.5, save=False, verbose=False)
            
            # Reset counters for this image
            k_count = 0
            s_count = 0
            
            for r in results:
                for box in r.boxes:
                    # Get the class name
                    cls_name = model.names[int(box.cls[0])]
                    if cls_name == 'Kirmizi':
                        k_count += 1
                    else:
                        s_count += 1
            
            total = k_count + s_count
            
            # Write results to CSV and Print to screen
            writer.writerow([img_name, total, k_count, s_count])
            print(f"Processed {img_name}: Found {k_count} Kirmizi, {s_count} Siirt.")

print(f"\nFinished! All counts saved to {output_csv}")