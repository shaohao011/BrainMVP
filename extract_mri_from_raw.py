import os
import glob
import json
import zipfile
import pydicom
import io
import shutil
import tempfile
import dicom2nifti
import pandas as pd
import re
import unicodedata
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ---
RAW_DATA_ROOT = '/mnt/disk1/aiotlab/namth/datasets/DienBien'
IMAGING_DIR = os.path.join(RAW_DATA_ROOT, 'imaging')
PATIENTS_DIR = os.path.join(RAW_DATA_ROOT, 'patients')

# Thư mục đích mới (Sạch 100%, chứa đủ T1, T2...)
NEW_OUTPUT_DIR = '/mnt/disk1/aiotlab/minhnn225043/EDA_PROCESS_STROKE_DATA/DienBien_MRI_Full_Raw'
OUTPUT_JSON = 'jsons/dataset_DienBien_MRI_Full.json'

# Mapping nhãn
LABEL_MAP = {
    'I67': 0, 'I68': 0, 'I69': 0, # Class 0: Negative
    'I61': 1,                     # Class 1: Xuất huyết
    'I63': 2,                     # Class 2: Nhồi máu
    'I64': 3                      # Class 3: Unlabeled (Lấy luôn để dùng sau này)
}

# --- HÀM HỖ TRỢ ---
def clean_string(s):
    if not isinstance(s, str): return "Unknown"
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    return re.sub(r'_+', '_', s).strip('_').upper()

def identify_mri_sequence(desc, num_slices):
    """Phân loại chuỗi xung dựa trên mô tả"""
    d = str(desc).lower()
    # Loại bỏ ảnh rác, ảnh định vị
    if num_slices < 10: return None 
    if any(x in d for x in ['localizer', 'scout', 'topo', 'dose', 'screen', 'mpr']): return None
    
    # Ưu tiên DWI và FLAIR trước
    if 'adc' in d: return 'ADC' # Tách ADC riêng
    if 'dwi' in d or 'diff' in d: return 'DWI'
    if 'flair' in d or 'dark-fluid' in d: return 'FLAIR'
    if 't1' in d: return 'T1'
    if 't2' in d: return 'T2'
    
    return None

# --- 1. LOAD DATABASE BỆNH NHÂN TỪ CSV GỐC ---
print("1. Đang đọc thông tin bệnh nhân từ CSV gốc...")
csv_files = glob.glob(os.path.join(PATIENTS_DIR, 'downloaded_I*.csv'))
patient_db = {}

for f in csv_files:
    icd_code = os.path.basename(f).replace('downloaded_', '').replace('.csv', '')
    if icd_code not in LABEL_MAP: continue
    
    try:
        df = pd.read_csv(f)
        # Chỉ lấy MRI (Lọc ngay từ CSV để tiết kiệm thời gian)
        df = df[df['Máy chụp'].astype(str).str.contains('MR|Cộng hưởng từ', case=False, na=False)]
        
        df['ZipName'] = df['path'].apply(lambda x: os.path.basename(str(x).replace('\\', '/')))
        
        for _, row in df.iterrows():
            z_name = row['ZipName']
            # Lưu thông tin cần thiết
            patient_db[z_name] = {
                'ICD': icd_code,
                'Label': LABEL_MAP[icd_code],
                'Real_ID': str(row['Mã bệnh nhân']),
                'Real_Name': clean_string(row['Họ tên'])
            }
    except Exception as e:
        print(f"Lỗi đọc CSV {f}: {e}")

print(f"-> Tổng số ca MRI cần xử lý: {len(patient_db)}")

# --- 2. TRÍCH XUẤT VÀ CONVERT (CHẠY 1 LẦN ĂN NGAY) ---
print("\n2. Bắt đầu trích xuất T1, T2, FLAIR, DWI từ Zip gốc...")
if not os.path.exists(NEW_OUTPUT_DIR):
    os.makedirs(NEW_OUTPUT_DIR)

processed_samples = [] # Danh sách để tạo JSON

# Lấy list file zip thực tế
all_zips = glob.glob(os.path.join(IMAGING_DIR, '*.zip'))
# Chỉ xử lý những zip có trong DB (tức là MRI)
target_zips = [z for z in all_zips if os.path.basename(z) in patient_db]

for z_path in tqdm(target_zips):
    z_name = os.path.basename(z_path)
    info = patient_db[z_name]
    
    # Tạo folder riêng cho từng bệnh nhân để dễ quản lý
    # Path: Output / ClassX / PatientID_StudyID / Modality.nii.gz
    # Lấy StudyID từ tên Zip (số dài thứ 2)
    try:
        study_id = z_name.split('_')[1]
    except:
        study_id = "Unknown"
        
    patient_folder_name = f"{info['Real_ID']}_{study_id}"
    class_folder_name = f"Class{info['Label']}_{info['ICD']}"
    
    save_dir = os.path.join(NEW_OUTPUT_DIR, class_folder_name, patient_folder_name)
    
    # Dictionary lưu đường dẫn các file đã convert của BN này
    patient_files = {'image': [], 'label': info['Label'], 'id': patient_folder_name, 'modalities': []}
    
    # Nếu folder đã tồn tại và có đủ file thì có thể skip (Resume logic), nhưng ở đây ta làm lại cho chắc
    # if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0: ...
    
    try:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        with zipfile.ZipFile(z_path, 'r') as z:
            # Gom file theo series folder (0000, 0001...)
            series_dict = {}
            for f in z.namelist():
                if f.endswith('.dcm'):
                    d = os.path.dirname(f)
                    series_dict.setdefault(d, []).append(f)
            
            for s_folder, files in series_dict.items():
                # Đọc header để xác định loại ảnh
                try:
                    with z.open(files[0]) as dcm_io:
                        dcm = pydicom.dcmread(io.BytesIO(dcm_io.read()), stop_before_pixels=True)
                        desc = getattr(dcm, 'SeriesDescription', 'Unknown')
                        modality_type = identify_mri_sequence(desc, len(files))
                        
                        if modality_type:
                            # Tên file đích: Modality.nii.gz (VD: FLAIR.nii.gz)
                            # Đơn giản hóa tên file để DataLoader dễ tìm
                            target_filename = f"{modality_type}.nii.gz"
                            target_path = os.path.join(save_dir, target_filename)
                            
                            # Convert
                            if not os.path.exists(target_path):
                                with tempfile.TemporaryDirectory() as tmp:
                                    for f in files: z.extract(f, path=tmp)
                                    src_dcm = os.path.join(tmp, s_folder)
                                    dicom2nifti.convert_directory(src_dcm, tmp, compression=True, reorient=True)
                                    generated = glob.glob(os.path.join(tmp, '*.nii.gz'))
                                    if generated:
                                        shutil.move(generated[0], target_path)
                            
                            # Thêm vào danh sách dataset nếu file tồn tại
                            if os.path.exists(target_path):
                                patient_files['image'].append(target_path)
                                patient_files['modalities'].append(modality_type)
                except:
                    pass
        
        # Chỉ thêm vào dataset nếu trích xuất được ít nhất 1 modal
        if len(patient_files['image']) > 0:
            # Sắp xếp lại thứ tự ưu tiên: FLAIR, DWI, T1, T2
            # Để đảm bảo tính nhất quán khi đưa vào Model
            sorted_images = []
            priority = {'FLAIR': 0, 'DWI': 1, 'T1': 2, 'T2': 3, 'ADC': 4}
            
            # Zip path và modality lại để sort
            zipped = list(zip(patient_files['image'], patient_files['modalities']))
            zipped.sort(key=lambda x: priority.get(x[1], 99))
            
            patient_files['image'] = [x[0] for x in zipped]
            patient_files['modalities'] = [x[1] for x in zipped] # Lưu để debug
            
            processed_samples.append(patient_files)
            
    except Exception as e:
        print(f"Lỗi xử lý zip {z_name}: {e}")

# --- 3. TẠO JSON ---
print("\n3. Tạo file JSON...")

# Tách riêng Class 3 (Unlabeled) ra khỏi Train/Val/Test set chính
labeled_samples = [s for s in processed_samples if s['label'] != 3]
unlabeled_samples = [s for s in processed_samples if s['label'] == 3]

print(f"- Số mẫu có nhãn (0, 1, 2): {len(labeled_samples)}")
print(f"- Số mẫu chưa nhãn (3): {len(unlabeled_samples)}")

# Chia tập train
try:
    train_data, test_val_data = train_test_split(labeled_samples, test_size=0.3, random_state=42, stratify=[x['label'] for x in labeled_samples])
    val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=42, stratify=[x['label'] for x in test_val_data])
except:
    print("Cảnh báo: Không thể Stratify split (do số lượng class quá ít). Split ngẫu nhiên.")
    train_data, test_val_data = train_test_split(labeled_samples, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

json_output = {
    "training": train_data,
    "validation": val_data,
    "test": test_data,
    "unlabeled": unlabeled_samples # Lưu thêm vào để dùng sau
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(json_output, f, indent=4)

print(f"-> HOÀN TẤT. File JSON lưu tại: {OUTPUT_JSON}")
print(f"-> Dữ liệu NIfTI sạch lưu tại: {NEW_OUTPUT_DIR}")

# Thống kê nhanh
print("\n--- THỐNG KÊ CHI TIẾT ---")
for cls in [0, 1, 2, 3]:
    count = len([s for s in processed_samples if s['label'] == cls])
    print(f"Class {cls}: {count} ca")

# Kiểm tra mẫu xem có đủ T1, T2 không
print("\n--- KIỂM TRA MẪU ---")
if len(labeled_samples) > 0:
    sample = labeled_samples[0]
    print(f"ID: {sample['id']}")
    print(f"Label: {sample['label']}")
    print(f"Modalities: {sample['modalities']}")