import torch
import os

# Cấu hình đường dẫn
CKPT_PATH = 'pretrained_weights/BrainMVP_uniformer.pt'

def analyze_checkpoint():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Không tìm thấy file: {CKPT_PATH}")
        return

    print(f"🔍 Đang phân tích kỹ file: {CKPT_PATH}")
    try:
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # 1. Kiểm tra các keys tầng ngoài cùng
    print("\n1. CẤU TRÚC TẦNG NGOÀI (Top-level Keys):")
    print(list(ckpt.keys()))

    if 'state_dict' not in ckpt:
        print("⚠️ Không tìm thấy 'state_dict'. Dừng phân tích.")
        return

    state_dict = ckpt['state_dict']
    all_keys = list(state_dict.keys())
    print(f"-> Tổng số tham số (Layers): {len(all_keys)}")

    # 2. Phân tích Tiền tố (Prefix Analysis)
    print("\n2. PHÂN TÍCH TIỀN TỐ (PREFIX):")
    first_key = all_keys[0]
    print(f"   Key mẫu đầu tiên: '{first_key}'")
    
    prefix = ""
    if "module.encoder.uniformer." in first_key:
        prefix = "module.encoder.uniformer."
    elif "encoder.uniformer." in first_key:
        prefix = "encoder.uniformer."
    elif "uniformer." in first_key:
        prefix = "uniformer."
    
    print(f"-> Tiền tố chung (cần cắt bỏ khi load): '{prefix}'")

    # 3. Soi lớp đầu vào (Input Layer)
    print("\n3. PHÂN TÍCH LỚP ĐẦU VÀO (Patch Embedding):")
    # Thường tên là patch_embed1.proj.weight
    input_keys = [k for k in all_keys if "patch_embed1.proj.weight" in k]
    
    if input_keys:
        k = input_keys[0]
        w = state_dict[k]
        print(f"   Tên layer: {k}")
        print(f"   Kích thước (Shape): {w.shape}")
        # Conv3d shape: [Out_Channels, In_Channels, D, H, W]
        in_channels = w.shape[1]
        print(f"   => Số kênh đầu vào gốc (In_Channels): {in_channels}")
        
        if in_channels == 1:
            print("   ✅ KẾT LUẬN: Model Pre-train trên từng ảnh đơn (1 kênh).")
            print("      Cần kỹ thuật 'Weight Inflation' để dùng cho input 4 kênh (FLAIR, DWI, T1, T2).")
        elif in_channels == 4:
            print("   ✅ KẾT LUẬN: Model khớp với input 4 kênh. Không cần sửa.")
        else:
            print(f"   ⚠️ LƯU Ý: Số kênh là {in_channels}, cần kiểm tra lại chiến lược.")
    else:
        print("❌ Không tìm thấy lớp Patch Embedding đầu vào!")

    # 4. Soi lớp đầu ra (Prediction Head)
    print("\n4. PHÂN TÍCH LỚP ĐẦU RA (Head):")
    # Thường tên là head.weight hoặc fc.weight hoặc decoder...
    head_keys = [k for k in all_keys if "head" in k or "fc" in k or "decoder" in k]
    
    # Lấy 5 key cuối cùng trong danh sách (thường là layer cuối)
    print("   Các layer cuối cùng trong model:")
    for k in all_keys[-5:]:
        print(f"   - {k} | Shape: {state_dict[k].shape}")

    # 5. Kiểm tra Template (Đặc sản của BrainMVP)
    print("\n5. KIỂM TRA TEMPLATE (Data Distillation):")
    if 'module.rep_template' in state_dict:
        print(f"   Tìm thấy 'module.rep_template'. Shape: {state_dict['module.rep_template'].shape}")
        print("   (Đây là khuôn mẫu não học được, có thể dùng để điền khuyết thiếu nếu cần)")
    elif 'rep_template' in ckpt:
        print(f"   Tìm thấy 'rep_template' ở tầng ngoài. Shape: {ckpt['rep_template'].shape}")
    else:
        print("   Không tìm thấy Template.")

if __name__ == "__main__":
    analyze_checkpoint()