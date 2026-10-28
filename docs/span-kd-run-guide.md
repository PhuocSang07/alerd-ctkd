# Hướng dẫn chạy: chưng cất CE + KD có trọng số theo span trên dữ liệu math của TSD-KD

Thiết lập: teacher Qwen2.5-14B-Instruct (đóng băng, bf16), student Qwen2.5-1.5B-Instruct (LoRA r16). Dữ liệu là 22.543 mẫu Math CoT của TSD-KD, lọc còn 21.347 mẫu không quá 1.024 token. Mặc định chấm gain và huấn luyện trên **2.000 mẫu đầu** (theo thứ tự của dữ liệu). Giải thích method và kết quả thí điểm nằm ở `docs/span-kd-experiments.md`; đánh giá chi tiết code ở `plans/reports/review-261028-1200-span-kd-method-and-experiment-setup.md`.

## 1. Chuẩn bị

Cần một GPU 80 GB trở lên (A100-80GB hoặc H200) và Python 3.10+.

```
pip install -r requirements.txt
export HF_HOME=/đường/dẫn/cache      # tùy chọn: nơi cache mô hình (khoảng 40 GB cho hai mô hình)
```

Dữ liệu (`phuocsang/UltraInteract-math-Qwen2.5-14B-Instruct-Gen`) và hai mô hình đều công khai, không cần token. Không đặt token vào file hay lệnh; nếu dùng mô hình cần quyền thì đặt `HF_TOKEN` trong môi trường.

Xem số lượng mẫu và độ dài token trên CPU, không tốn GPU:

```
NUM_SAMPLES=2000 bash scripts/prepare_tsdkd.sh
```

## 2. Chạy toàn bộ một lệnh

```
DATASET=tsdkd bash scripts/span_kd_run_all.sh
```

Lệnh này lần lượt chạy ba bước dưới đây với các giá trị mặc định cho `tsdkd`:

| Mặc định | Giá trị |
|---|---|
| Số mẫu | `LIMIT=2000` (đặt `LIMIT=` rỗng để lấy cả 21.347 mẫu) |
| Huấn luyện | 2 epoch, batch hiệu dụng 64, micro-batch 8, lr 1e-4 cosine, β = 0,5, top-k 50, seed 10 |
| Cấu hình | `sft kd-token kd-token-mean gain-ce gain-kd gain-both` |
| Đánh giá | student zero-shot và từng cấu hình trên GSM8K, MATH500, AIME24, AIME25; prompt TSD-KD |

Thư mục sinh ra:

- `work-tsdkd-limit2000/Qwen2.5-1.5B-Instruct/`: `train-sra.jsonl`, `spans-Qwen2.5-14B-Instruct.jsonl`, `span-gains.json`.
- `outputs/span-kd-tsdkd/<cấu hình>/`: checkpoint đã gộp LoRA (bf16), `train_stats.jsonl`, `spans.jsonl`.
- `results/span-kd-tsdkd/<cấu hình>/<bộ>/summary.json`: pass@1, pass@k, độ dài, tỷ lệ cắt cụt.

## 3. Chạy từng bước

**Bước 1: chấm gain.** Mỗi token của lời giải được chấm một lần cho teacher và một lần cho student (khoảng 372.600 lần chấm mỗi mô hình cho 1.000 mẫu). Chi phí lớn nhất là teacher: khoảng 38 lần chấm mỗi giây trên A100-80GB, tức khoảng 5 giờ cho 2.000 mẫu, cộng khoảng 2 giờ cho student. Con số trên H200 chưa đo.

```
DATASET=tsdkd LIMIT=2000 SPAN_MODE=word GPU_MEMORY_UTILIZATION=0.85 \
  STUDENTS=Qwen/Qwen2.5-1.5B-Instruct TEACHER=Qwen/Qwen2.5-14B-Instruct \
  bash scripts/score_gain.sh work-tsdkd
```

Kết quả chấm chỉ được lưu khi cả bước hoàn tất, vì vậy hãy chạy bước này trong `tmux` hoặc `nohup`. Muốn chia nhiều GPU, dùng `SHARD=i/n` (mỗi lần một phần) rồi gộp bằng `scripts/modal_score_gain.py`.

**Bước 2: huấn luyện.**

```
DATASET=tsdkd DATA_DIR=work-tsdkd-limit2000/Qwen2.5-1.5B-Instruct EPOCHS=2 BATCH_SIZE=64 MICRO_BATCH_SIZE=8 \
  bash scripts/span_kd_train.sh
```

Chọn cấu hình bằng `EXPERIMENTS="sft kd-token gain-both"`. Bảng đầy đủ các cấu hình nằm ở đầu `scripts/span_kd_train.sh`. Các cấu hình dùng trung bình theo mẫu, riêng `kd-token-mean` dùng trung bình theo token toàn batch như baseline khác.

**Bước 3: đánh giá.**

```
ZERO_SHOT=1 bash scripts/span_kd_eval.sh
```

`PROMPT_STYLE=tsdkd` (mặc định) khớp prompt huấn luyện; `PROMPT_STYLE=alerd` cho student huấn luyện trên GSM8K. Sinh 4.096 token mới, temperature 0,6, top-p 0,9; GSM8K và MATH500 lấy 3 mẫu mỗi bài, AIME 16 mẫu.

## 4. Dùng lại kết quả và chỉnh bộ nhớ

- Đã có gain rồi, chỉ muốn train lại: `SKIP_SCORING=1 DATASET=tsdkd bash scripts/span_kd_run_all.sh` (cần cùng `LIMIT`).
- Hết bộ nhớ khi train: giảm `MICRO_BATCH_SIZE` (8 → 4 → 2); batch hiệu dụng vẫn là 64 nhờ tích lũy gradient. Còn dư bộ nhớ thì tăng lên 16. Micro-batch 8 là ước lượng cho H200 141 GB và chưa được chạy thử.
- Hết bộ nhớ khi chấm gain: giảm `GPU_MEMORY_UTILIZATION` hoặc `MAX_MODEL_LEN`.
- Đổi cặp mô hình: `STUDENT=` và `TEACHER=`; hai mô hình phải dùng chung tokenizer.

## 5. Kiểm tra nhanh trên CPU

```
python -m pytest tests -q
```

Bộ test dùng mô hình ngẫu nhiên rất nhỏ nên không tải mô hình thật.

## 6. Lưu ý khi báo cáo kết quả

- Student dùng LoRA, còn các baseline của bài TSD-KD fine-tune toàn bộ; cần ghi rõ.
- Các cấu hình trọng số chuẩn hóa loss theo mẫu. `kd-token-mean` là KD chuẩn hóa theo token toàn batch, dùng để so với baseline chạy bằng code khác.
- Bộ đánh giá chưa có GSM-Plus, Minerva Math, MMLU-Pro-Math, MMLU-STEM, SciQ, MBPP và BBH.
- Chưa có kết quả nào trên dữ liệu TSD-KD; các số trong `docs/span-kd-experiments.md` là thí điểm 1.000 bài GSM8K.
