# VIC — Video Individual Counting

ระบบนับจำนวนบุคคลไม่ซ้ำที่เดินผ่านกล้องตลอดวิดีโอ โดยไม่ต้องใช้ชุดข้อมูลเฉพาะทาง (zero-shot)

---

## การทำงานของระบบ

ระบบแบ่งเป็น 5 ขั้นตอนต่อเนื่องกัน แต่ละขั้นรับ output จากขั้นก่อนหน้า

```
entrance.mov (หรือวิดีโอใดก็ได้)
    │
    ▼  Step 1 — Head Point Detection
    │  โมเดล FasterRCNN-ResNet50-FPN-v2 (COCO pretrained) ตรวจจับตำแหน่งหัวของแต่ละคน
    │  ในทุก frame เป็น point (cx, cy) แทนที่จะใช้ bounding box เต็ม
    │
    ▼  Step 2 — Appearance Descriptor Extraction
    │  ตัดภาพรอบหัวแต่ละคน (128x128 px) แล้วส่งผ่าน ConvNext-S (ImageNet pretrained)
    │  ได้ descriptor 768 มิติต่อคนต่อ frame → บันทึกเป็น descriptors_all_frames.pt
    │
    ▼  Step 3 — ICG Intra-frame Attention (optional)
    │  ใช้ dot-product self-attention ภายใน frame เดียวกัน
    │  เพื่อสร้าง context-aware descriptor และ visualize attention map
    │  (ขั้นตอนนี้เป็น optional ไม่จำเป็นสำหรับการนับคน)
    │
    ▼  Step 4 — DPI (Displacement Prior Injector)
    │  track แต่ละคนข้ามเฟรมโดยใช้ Hungarian matching ด้วย 2 signal รวมกัน:
    │    - Appearance: cosine similarity ของ raw descriptor
    │    - Displacement Prior: Gaussian ที่ตำแหน่งที่คาดไว้จาก constant-velocity model
    │  ผล: แต่ละคนได้รับ track ID ที่ต่อเนื่องข้ามเฟรม → dpi_tracks.pt
    │
    ▼  Step 5 — OMPM (One-to-Many Pairwise Matching)
       เปรียบแต่ละคนในเฟรมปัจจุบันกับ memory buffer ของคนที่เคยเห็นมาแล้ว
       (One-to-Many: คน 1 คนใน memory match กับคนหลายคนในเฟรมใหม่ได้)

       ถ้า best score >= threshold  →  Pedestrian  (คนเดิม ไม่นับซ้ำ)
       ถ้า best score <  threshold  →  Inflow      (คนใหม่ บวกเข้า counter)

       ผลลัพธ์สุดท้าย: จำนวน Inflow = จำนวนบุคคลไม่ซ้ำทั้งหมดในวิดีโอ
```

---

## ความต้องการของระบบ

| รายการ | ข้อกำหนด |
|---|---|
| OS | Windows 10/11 หรือ Linux |
| Python | 3.10 ขึ้นไป (ทดสอบบน 3.13.5) |
| GPU | NVIDIA พร้อม CUDA 12.4 (แนะนำ VRAM >= 4 GB) |
| CPU-only | ใช้ได้ แต่ Step 1-2 จะช้ามาก (~5-10x) |
| RAM | >= 8 GB |
| พื้นที่ดิสก์ | >= 2 GB (สำหรับ model weights ที่โหลดอัตโนมัติ) |

---

## การติดตั้งบนเครื่องใหม่

### 1. ติดตั้ง Python และ Conda (ถ้ายังไม่มี)

แนะนำใช้ [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda create -n vic python=3.11
conda activate vic
```

### 2. ติดตั้ง PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

> ถ้าไม่มี GPU หรือใช้ CUDA version อื่น ให้ดู https://pytorch.org/get-started/locally/ แล้วเลือก config ที่ตรงกัน

### 3. ติดตั้ง dependencies ที่เหลือ

```bash
pip install timm==1.0.27 opencv-python==4.13.0 numpy==2.3.1 scipy==1.16.3 matplotlib==3.10.6
```

หรือใช้ไฟล์ requirements.txt (ต้องติดตั้ง PyTorch ก่อนในข้อ 2 แล้วค่อยรัน):

```bash
pip install -r requirements.txt
```

### 4. ตรวจสอบ GPU

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

---

## วิธีใช้งาน

### กรณี A — ใช้ผลลัพธ์ที่คำนวณไว้แล้ว (เร็วที่สุด)

ไฟล์ `.pt` ในโฟลเดอร์ `models/` คือผลที่คำนวณจาก `entrance.mov` ไว้แล้ว  
สามารถโหลดและใช้งานได้ทันทีโดยไม่ต้องรัน Step 1-2 ใหม่

```python
import torch

# โหลด OMPM results (ผลลัพธ์สุดท้าย)
ompm = torch.load("models/ompm_results.pt", weights_only=False)

# นับจำนวนคนทั้งหมด
inflow_count = sum(
    1 for rows in ompm.values()
    for r in rows if r["label"] == "Inflow"
)
print("จำนวนบุคคลไม่ซ้ำ:", inflow_count)

# ดูข้อมูลต่อ frame (เช่น frame 300)
for r in ompm[300]:
    print(r["label"], r["uid"], r["pos"], round(r["best_score"], 3))
```

```python
# โหลด DPI tracks
dpi = torch.load("models/dpi_tracks.pt", weights_only=False)

# ดู track ใน frame 300
for r in dpi[300]:
    print(f"track ID={r['tid']}  pos={r['pos']}  score={r['score']:.3f}")
```

```python
# โหลด descriptors (ใช้สำหรับ re-run Step 3-5 กับ video ใหม่)
descs = torch.load("models/descriptors_all_frames.pt", weights_only=False)
frame_data = descs[300]   # frame 300
print("persons:", len(frame_data["points"]))
print("desc shape:", frame_data["descs"].shape)   # (N, 768)
```

### กรณี B — รัน pipeline ทั้งหมดกับวิดีโอใหม่

1. เปิด `full_pipeline.ipynb` ใน Jupyter Notebook หรือ VS Code
2. แก้ path วิดีโอใน **Cell 2** (Imports & Project Paths):

```python
VIDEO_IN = ROOT / "ชื่อวิดีโอของคุณ.mp4"   # เปลี่ยนตรงนี้
```

3. รัน cells ตามลำดับ:

| Cells | ขั้นตอน | เวลาโดยประมาณ (RTX 3050) |
|---|---|---|
| 1–5 | Imports + GPU setup | < 1 นาที |
| 6–7 | Step 1: โหลด FasterRCNN | < 1 นาที |
| 8–10 | Step 2: โหลด ConvNext-S | < 1 นาที |
| **12** | **รัน Step 1+2 ทุก frame** | **~10 นาที** |
| 14 | ตรวจสอบผล | < 1 นาที |
| 23 | Step 4: DPI class | < 1 นาที |
| 24 | รัน DPI tracking | < 1 นาที |
| 27 | Step 5: OMPM run | < 1 นาที |
| 28 | กราฟ inflow timeline | < 1 นาที |
| 29 | สร้างวิดีโอ OMPM | ~3 นาที |

> **Cell 12 ใช้เวลานานที่สุด** เพราะต้องรัน FasterRCNN + ConvNext-S ทุก frame  
> ถ้ารันแล้วครั้งหนึ่ง ไฟล์ `descriptors_all_frames.pt` จะถูกบันทึกไว้  
> ครั้งต่อไปข้ามไป Cell 23 ได้เลย

### กรณี C — ข้าม Step 1-2 โดยใช้ descriptors ที่มีอยู่แล้ว

ถ้ามีไฟล์ `models/descriptors_all_frames.pt` อยู่แล้ว ให้ copy ไปที่ `output/features/` แล้วรัน cells 14, 23, 24, 27, 28, 29 เท่านั้น

```bash
# Windows
copy models\descriptors_all_frames.pt output\features\descriptors_all_frames.pt

# Linux/Mac
cp models/descriptors_all_frames.pt output/features/descriptors_all_frames.pt
```

---

## โครงสร้างโฟลเดอร์

```
VIC-Counting/
├── full_pipeline.ipynb              # notebook หลัก (Steps 1-5)
├── README.md
├── requirements.txt
├── models/
│   ├── ompm_results.pt              # ผลลัพธ์สุดท้าย: label + uid + score ต่อ frame
│   ├── dpi_tracks.pt                # DPI track ID ต่อ frame
│   └── descriptors_all_frames.pt   # ConvNext-S 768-dim descriptors ทุก frame
└── results/
    └── ompm_tracking.mp4            # วิดีโอ annotated (P=Pedestrian, IN=Inflow)
```

---

## Parameters ของแต่ละโมเดล

### Step 1 — FasterRCNN Head Detection

| Parameter | ค่า | คำอธิบาย |
|---|---|---|
| `PERSON_CLASS` | `1` | COCO label index สำหรับ "person" |
| `SCORE_THRESH` | `0.90` | confidence threshold ต่ำกว่านี้ตัดทิ้ง |
| `BATCH_DETECT` | `4` | จำนวน frames ต่อ inference batch |
| head offset | top `12%` ของ bbox | `cy = y1 + (y2-y1) × 0.12` แทนที่ center จริง |
| backbone | ResNet-50 + FPN v2 | pretrained บน COCO 80 classes |

ลด `SCORE_THRESH` (เช่น `0.7`) ถ้าตรวจจับคนได้น้อยเกินไป  
เพิ่ม `SCORE_THRESH` (เช่น `0.95`) ถ้ามี false positive มาก

---

### Step 2 — ConvNext-S Descriptor Extraction

| Parameter | ค่า | คำอธิบาย |
|---|---|---|
| `PATCH_SIZE` | `128` px | ขนาด crop รอบหัวแต่ละคน (centered at head point) |
| `CONVNEXT_INPUT` | `224` px | input size ที่ ConvNext-S ต้องการ |
| `DESCRIPTOR_DIM` | `768` | มิติของ feature vector ที่ได้ |
| padding | `edge` | เมื่อ crop ออกนอกขอบภาพ ใช้ค่า pixel ขอบ |
| normalization mean | `[0.485, 0.456, 0.406]` | ImageNet mean (RGB) |
| normalization std | `[0.229, 0.224, 0.225]` | ImageNet std (RGB) |
| classification head | ตัดออก (`num_classes=0`) | ใช้เฉพาะ feature extractor |

---

### Step 3 — ICG Intra-frame Attention (optional)

| Parameter | ค่า | คำอธิบาย |
|---|---|---|
| `ICG_TEMPERATURE` | `1.0` | scaling ของ softmax: `scale = sqrt(D) × temp` |
| attention type | dot-product self-attention | Q = K = V = L2-normalized descriptors |
| output | context descriptor (N, 768) | L2-normalized weighted sum |

ลด temperature (เช่น `0.1`) ได้ attention ที่ sharp กว่า (เน้นคนที่คล้ายที่สุด)  
เพิ่ม temperature ได้ attention ที่กระจายสม่ำเสมอขึ้น

---

### Step 4 — DPI (Displacement Prior Injector)

| Parameter | ค่า | คำอธิบาย |
|---|---|---|
| `DPI_SIGMA` | `60` px | Gaussian sigma รอบตำแหน่งที่ทำนาย (original-res) |
| `DPI_ALPHA` | `0.45` | สัดส่วน spatial prior vs appearance (`0`=appearance only) |
| `DPI_MATCH_THR` | `0.30` | combined score ต่ำกว่านี้ = spawn new track |
| `DPI_MAX_LOST` | `30` frames | track หายไปนานเกินนี้ถูกลบ (~1 วินาทีที่ 30fps) |
| `DESC_EMA_BETA` | `0.80` | EMA weight ของ descriptor เดิม (สูง = เชื่อประวัติมากกว่า) |
| velocity model | constant velocity | `pred = pos(T) + (pos(T) − pos(T−1))` |
| matching | Hungarian algorithm | optimal one-to-one assignment |

**สูตร score:**
```
score(track_i, det_j) = (1 − 0.45) × cosine_norm(desc_i, desc_j)
                      +      0.45  × exp(−dist(pred_i, pos_j)² / 2×60²)
```

---

### Step 5 — OMPM (One-to-Many Pairwise Matching)

| Parameter | ค่า | คำอธิบาย |
|---|---|---|
| `OMPM_MATCH_THR` | `0.45` | threshold ตัดสิน Pedestrian vs Inflow |
| `OMPM_SIGMA` | `80` px | Gaussian sigma สำหรับ spatial component |
| `OMPM_ALPHA` | `0.45` | สัดส่วน spatial vs appearance |
| `OMPM_MEMORY` | `5` frames | จำนวน frames ย้อนหลังที่เก็บใน reference group |
| `OMPM_MAX_AGE` | `20` frames | memory entry expire หลังจากไม่ถูก match นานเท่านี้ |
| descriptor EMA | `0.8 × old + 0.2 × new` | อัปเดต memory descriptor เมื่อ match ได้ |
| matching type | One-to-Many | คน 1 คนใน memory match กับหลายคนในเฟรมใหม่ได้ |

**สูตร score:**
```
score(curr_i, mem_j) = (1 − 0.45) × (cosine(desc_i, mem_j) + 1) / 2
                     +      0.45  × exp(−dist(pos_i, mem_j.pos)² / 2×80²)

ถ้า max(score) >= 0.45  →  Pedestrian
ถ้า max(score) <  0.45  →  Inflow  (inflow_count++)
```

---

## ปรับแต่งพฤติกรรมการนับ

| ตัวแปร | Cell | ค่าเริ่มต้น | ผลเมื่อปรับ |
|---|---|---|---|
| `SCORE_THRESH` | 6 | `0.90` | ต่ำลง = ตรวจจับคนได้มากขึ้น |
| `OMPM_MATCH_THR` | 27 | `0.45` | ต่ำลง = รับ match ง่ายขึ้น (Inflow น้อยลง) |
| `OMPM_SIGMA` | 27 | `80` px | ต่ำลง = เข้มงวดกับตำแหน่งมากขึ้น |
| `OMPM_ALPHA` | 27 | `0.45` | สูงขึ้น = เชื่อ position มากกว่า appearance |
| `OMPM_MAX_AGE` | 27 | `20` frames | สูงขึ้น = จำคนได้นานขึ้น ทน occlusion ดีขึ้น |
| `DPI_SIGMA` | 23 | `60` px | ต่ำลง = เข้มงวดกับการเคลื่อนที่ของ DPI |
| `SLOW_FACTOR` | 29 | `4` | ปรับความเร็ววิดีโอ (`1` = realtime) |

---

## Pretrained Models

ระบบใช้ pretrained weights ที่โหลดอัตโนมัติจากอินเทอร์เน็ตครั้งแรกที่รัน:

| Model | Source | ขนาด |
|---|---|---|
| FasterRCNN-ResNet50-FPN-v2 | torchvision (COCO) | ~170 MB |
| ConvNext-Small | timm (ImageNet-1k) | ~197 MB |

weights จะถูก cache ใน `~/.cache/torch/` และ `~/.cache/huggingface/` โดยอัตโนมัติ

---

## References

[1] H. Lu, X. Zhu, W. Zhang, Y. Li, X. Bai.
**"Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors."**
arXiv:2601.01192, 2026.
https://arxiv.org/abs/2601.01192

> แนวคิดหลักของโปรเจคนี้ (VIC, OMPM, DPI, One-to-Many matching, Displacement Prior)
> อ้างอิงจากงานวิจัยข้างต้น
