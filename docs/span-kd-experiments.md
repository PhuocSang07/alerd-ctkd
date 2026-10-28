# Chưng cất CE + KD có trọng số theo span: setup, cách chạy, kết quả

Tài liệu này gom toàn bộ thí nghiệm "nhánh A" (student và teacher dùng chung tokenizer): Qwen2.5-14B-Instruct dạy Qwen2.5-1.5B-Instruct, trọng số của từng span cho loss CE và loss KD lấy từ answer gain. Số liệu bên dưới đo trên GSM8K train 1.000 bài; phần s1K chưa chạy xong (xem mục 6).

## 1. Ý tưởng và thành phần

- **Span theo từ** (`word_spans.py`): lời giải được cắt theo từ (số có dấu phân cách như `1,200` hay `3.5` là một span, một dãy `\w+` là một span, một ký tự khác là một span). Dấu cách phía trước thuộc từ sau; dấu xuống dòng thuộc từ cuối dòng. Mỗi token thuộc đúng một span và không span nào vắt qua dòng. Teacher dùng đúng span của student, dịch theo hiệu độ dài prompt.
- **Answer gain** (`answer_gain.py`, `token_gain.py`): sau mỗi token của lời giải, đo log-xác suất của đáp án cuối khi ghép thêm câu dẫn `\n\nTherefore, the final answer is \boxed{`. Gain của một span là hiệu hai giá trị ở hai đầu span. Chấm bằng student (đóng băng) và bằng teacher. Hai tín hiệu: `student` và `diff` = teacher − student. Xuất ra `span-gains.json`.
- **Trọng số** (`span_kd_train.py`, `allocation.py`): mỗi mẫu có tổng khối lượng 1 chia cho các span, mỗi span chia đều cho token của nó. Có bốn kiểu: `token` (mọi token như nhau), `uniform` (mọi span như nhau), hoặc trọng số ALERD (Eq. 3-5, τ=1, clip 2, λ=0,5, chuẩn hoá theo từng lời giải) từ gain `student`, `teacher` hoặc `diff`. Trọng số của CE và của KD chọn độc lập. Token EOS thuộc span cuối.
- **Loss**: `(1−β)·CE + β·KD`, KD là KL forward trên top-k logits của teacher, chuẩn hoá lại cả hai phân phối trên cùng tập top-k, nhân τ² (đúng mô tả của RIVAD, chế độ OFF).

## 2. Cấu hình đã dùng

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Teacher / student | Qwen2.5-14B-Instruct (bf16, đóng băng) / Qwen2.5-1.5B-Instruct | bài của anh Phong |
| LoRA dropout, weight decay, clip gradient, seed | 0,05; 0,01; 1,0; 10 | bài của anh Phong |
| Giới hạn token student / teacher | 1.024 / 2.560 | bài của anh Phong |
| Dạng loss | forward KL trên top-k, τ² | bài của anh Phong |
| β, top-k, nhiệt độ | 0,5; 50; 1 | **tôi tự chọn** |
| LoRA rank / alpha, mọi projection | 16 / 16 | **tôi tự chọn** (mặc định của repo) |
| Learning rate | 1e-4, cosine về 1e-5, warm-up 10% | **tôi tự chọn** |
| Epoch, batch hiệu dụng | 3; 32 (anh Phong: 128) | **tôi tự chọn** |
| Dữ liệu | 1.000 bài đầu GSM8K train, lời giải gốc | **khác** anh Phong (UltraInteract, lời giải do teacher sinh) |
| Prompt | chỉ dẫn `Please reason step by step, and put your final answer within \boxed{}.` + câu hỏi, khuôn chat Qwen | giao thức ALERD cũ; **chưa khớp** anh Phong (bài không nêu mẫu prompt) |

Thống kê span (GSM8K train, tokenizer Qwen): trung vị 64 span mỗi mẫu, p95 130; trung vị 1 token mỗi span, p95 3; không span nào vắt qua bước; không mẫu nào bị cắt ở 512 token.

## 3. Chín cấu hình trọng số

| Tên | CE | KD | β | λ |
|---|---|---|---|---|
| `sft` | token | - | 0 | 0,5 |
| `kd-token` | token | token | 0,5 | 0,5 |
| `kd-uniform` | uniform | uniform | 0,5 | 0,5 |
| `gain-ce` | student | token | 0,5 | 0,5 |
| `gain-kd` | token | diff | 0,5 | 0,5 |
| `gain-both` | student | diff | 0,5 | 0,5 |
| `gain-student` | student | student | 0,5 | 0,5 |
| `gain-diff` | diff | diff | 0,5 | 0,5 |
| `gain-both-lam1` | student | diff | 0,5 | 1,0 |

Bộ năm cấu hình mặc định khi chạy tiết kiệm: `sft`, `kd-token`, `gain-ce`, `gain-kd`, `gain-both`.

## 4. Cách chạy

**Trên máy có GPU (vLLM, một A100-80GB đủ chứa teacher 14B và student):**

```
# chấm gain, train năm cấu hình, đánh giá
DATASET=gsm8k LIMIT=1000 bash scripts/span_kd_run_all.sh

# hoặc từng bước
SPAN_MODE=word LIMIT=1000 STUDENTS=Qwen/Qwen2.5-1.5B-Instruct TEACHER=Qwen/Qwen2.5-14B-Instruct bash scripts/score_gain.sh
DATA_DIR=work-limit1000/Qwen2.5-1.5B-Instruct bash scripts/span_kd_train.sh      # EXPERIMENTS="sft kd-token ..." để chọn
ZERO_SHOT=1 bash scripts/span_kd_eval.sh                                          # OUT_ROOT / RESULTS_DIR đổi được
```

`scripts/span_kd_run_all.sh` nhận `DATASET=gsm8k|s1k`, `LIMIT=N`, `SKIP_SCORING=1` (dùng lại gain đã chấm). Các biến `EPOCHS`, `BATCH_SIZE`, `MICRO_BATCH_SIZE`, `LEARNING_RATE`, `KD_TOP_K`, `SEED`, `MAX_TOKENS` chỉnh được qua môi trường. Đầu ra: `outputs/span-kd-<dataset>/<tên>/` (checkpoint LoRA đã gộp, bf16, `train_stats.jsonl`, `spans.jsonl`) và `results/span-kd-<dataset>/<tên>/<bộ>/summary.json`.

**Trên Modal** (cần `modal setup`; các script này không dùng `*.sh`):

```
modal run --detach scripts/modal_score_gain.py --limit 1000                       # GSM8K, một GPU
modal run --detach scripts/modal_score_gain.py --dataset s1k --shards 3          # s1K, nhiều GPU rồi gộp
modal run --detach scripts/modal_train_kd.py --work work-limit1000 --tag e1k      # train + eval, tối đa 2 GPU cùng lúc
modal run --detach scripts/modal_eval_ckpt.py --tag e1k-ckpt                      # eval lại checkpoint với 4096 token
```

**Đánh giá** (`benchmark_eval.py`): GSM8K test (1.319 bài) và MATH500 với 3 mẫu mỗi bài, AIME24 và AIME25 (30 bài mỗi bộ) với 16 mẫu; nhiệt độ 0,6, top-p 0,9, tối đa 4096 token mới; chấm bằng Math-Verify. pass@1 là trung bình trên mọi mẫu; pass@3 là ước lượng không chệch khi có hơn 3 mẫu. Với giới hạn 1.024 token cũ, MATH500 bị cắt 8-11% và AIME24 27-34%, nên các số tin cậy là số 4096 token.

## 5. Kết quả (dữ liệu GSM8K train 1k, một seed)

GSM8K test pass@1 (1.024 token) và ba bộ khó (4096 token), pass@1 / pass@3, tính theo %:

| Cấu hình | GSM8K p@1 | MATH500 | AIME24 | AIME25 |
|---|---|---|---|---|
| student zero-shot | 74,48 | 54,27 / 67,80 | 2,71 / 5,38 | 1,46 / 4,13 |
| `sft` | 57,44 | 32,33 / 50,00 | 0,21 / 0,62 | 0,00 / 0,00 |
| `kd-token` | 69,65 | 49,13 / 65,00 | 1,88 / 4,41 | 0,62 / 1,88 |
| `kd-uniform` | 69,47 | 50,80 / 68,20 | 3,12 / 7,05 | 1,04 / 2,65 |
| `gain-ce` | 68,54 | 49,53 / 66,00 | 2,71 / 5,81 | 1,25 / 3,67 |
| `gain-kd` | 70,51 | 51,27 / 68,40 | 1,88 / 4,77 | 1,88 / 5,29 |
| `gain-both` | 68,81 | 49,87 / 67,40 | 2,08 / 4,71 | 0,83 / 2,26 |
| `gain-student` | 69,88 | 50,80 / 68,00 | 2,08 / 4,17 | 0,42 / 1,25 |
| `gain-diff` | 68,51 | 49,47 / 66,60 | 1,67 / 4,67 | 0,42 / 1,25 |
| `gain-both-lam1` | 68,94 | 50,87 / 68,20 | 2,71 / 6,12 | 1,88 / 5,37 |

Nhận xét (chỉ những điều số liệu hỗ trợ):

- Mọi cấu hình train đều thấp hơn zero-shot; SFT thuần giảm nhiều nhất (MATH500 32,3), thêm KD giữ lại phần lớn (khoảng 49-51). Độ dài sinh giảm theo: GSM8K từ 310 token (zero-shot) xuống 97 (SFT) và khoảng 195-222 (KD). Giả thuyết: lời giải GSM8K ngắn kéo model về lối viết ngắn; chưa kiểm chứng bằng dữ liệu do teacher sinh.
- Các cấu hình KD sát nhau (GSM8K 68,5-70,5; MATH500 49,1-51,3). Chạy lặp lại cùng cấu hình cho GSM8K lệch từ 0 đến 0,6 điểm, và sai số chuẩn của một seed cỡ 1 điểm, nên **chưa kết luận được** trọng số gain tốt hơn token hay uniform. AIME24/25 chỉ đúng 0-3% trên 30 bài, các khác biệt là nhiễu.
- Tín hiệu lúc train (log mỗi span của `kd-token`, 69.922 span): CE và KD của span tương quan 0,36; KD gần như không tương quan với gain `diff` (−0,02); span có gain student cao có CE cao hơn (nhóm thấp nhất 0,44, nhóm cao nhất 0,83), span có `diff` cao có CE thấp hơn (0,72 xuống 0,53). CE trung bình giảm 0,641 → 0,563 và KD 0,372 → 0,313 qua 3 epoch.

## 6. Chi phí đã đo và những việc chưa xong

- Chấm gain GSM8K 1k (95.751 lần chấm mỗi model, A100-80GB): student 431 giây, **teacher 14B 2.502 giây** (khoảng 38 lần/giây). Train mỗi cấu hình 2-9 phút (gồm nạp model; SFT nhanh nhất vì không tải teacher). Chưa đo riêng thời gian eval.
- **s1K chưa có gain.** Dữ liệu `simplescaling/s1K-1.1`, cột `deepseek_attempt` (không dùng phần suy nghĩ dài), còn 830 bài sau khi lọc theo có `\boxed` và giới hạn 1.024/2.560 token; 397.369 span, khoảng 473.000 lần chấm cho mỗi model (gấp 5 lần GSM8K 1k, ước 3,5 giờ cho teacher trên một GPU). Hai lần chấm 6 và 3 shard bị dừng: lần đầu vì workspace Modal hết hạn mức chi tiêu, lần sau tôi dừng vì tốn credit. Kết quả shard chỉ được lưu khi shard chạy xong nên mất hết. `answer_gain.py` nay in tiến độ; **chưa sửa** việc lưu kết quả giữa chừng.
- Chưa làm: prompt và dữ liệu của anh Phong (UltraInteract, lời giải do teacher sinh), nhiều seed, on-policy, các benchmark còn lại (GSM-Plus, MBPP, SciQ, MMLU, BBH).

## 6b. Dữ liệu TSD-KD (toán) và cách chạy

Dữ liệu: bộ Math CoT đã lọc `phuocsang/UltraInteract-math-Qwen2.5-14B-Instruct-Gen` (22.543 mẫu, public; lọc từ `Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen`, 79.751 mẫu), prompt lấy nguyên từ bản phát hành (không thêm câu lệnh ALERD), output là lời giải do teacher 14B sinh (off-policy, theo README của tác giả TSD-KD). Chỉ lấy Math CoT, không lấy Math PoT (22.718 mẫu, prompt có công cụ Python). Lọc theo tiền tố `Solve the following math problem step-by-step.` còn 22.582 mẫu Math CoT, bỏ 39 mẫu không có `\boxed{}` còn **22.543**; bỏ mẫu dài hơn 1.024 token (prompt + lời giải, cùng tokenizer cho teacher và student) còn **21.347**. Độ dài prompt + lời giải: trung vị 455, p95 843 token.

Chi phí chấm gain (ước từ tốc độ đã đo; chưa chạy): lời giải trung bình khoảng 350 token, nên 21.347 mẫu có khoảng 7,5 triệu lần chấm mỗi model, tức khoảng 55 giờ cho teacher (38 lần/giây) và hơn 20 giờ cho student. **Không chấm toàn bộ.** Dùng `LIMIT=N` (N mẫu đầu sau khi lọc): 1.000 mẫu khoảng 350.000 lần chấm mỗi model (teacher ước 2,5 giờ, student ước 1 giờ trên một A100-80GB).

```bash
NUM_SAMPLES=1000 bash scripts/prepare_tsdkd.sh      # CPU: tải, lọc, đếm số mẫu và độ dài token (bỏ NUM_SAMPLES = lấy hết)
DATASET=tsdkd LIMIT=1000 bash scripts/span_kd_run_all.sh   # chấm gain (GPU) -> train 5 cấu hình (LoRA, 2 epoch, batch 64) -> eval
DATASET=tsdkd bash scripts/span_kd_run_all.sh              # không đặt LIMIT: dùng toàn bộ 21.347 mẫu vừa 1.024 token
```

`LIMIT=K` với `DATASET=tsdkd` là **K mẫu đầu theo thứ tự của bộ dữ liệu**, lấy sau khi bỏ mẫu dài hơn 1.024 token; không đặt `LIMIT` thì lấy hết. (Với `gsm8k` và `s1k`, `LIMIT=K` cũng là K mẫu đầu.) Thư mục làm việc là `work-tsdkd-limitK` hoặc `work-tsdkd`. Nguồn dữ liệu đổi bằng `TSDKD_SOURCE=<id trên Hub hoặc file jsonl>`.

Các bước chạy riêng lẻ, theo thứ tự: `scripts/score_gain.sh` (lọc, chuẩn bị, chấm gain student và teacher, tạo chỉ mục span và `span-gains.json`), `scripts/span_kd_train.sh` (huấn luyện, chọn cấu hình bằng `EXPERIMENTS`), `scripts/span_kd_eval.sh` (đánh giá). `scripts/span_kd_run_all.sh` chạy cả ba; `SKIP_SCORING=1` dùng lại gain đã có.

Với `DATASET=tsdkd`, `scripts/score_gain.sh` giới hạn 1.024 token cho cả hai mô hình; `span_kd_run_all.sh` đặt epoch 2, batch hiệu dụng 64 với micro-batch 8 (cho H200 141 GB; chưa chạy thử trên GPU, hạ `MICRO_BATCH_SIZE` nếu hết bộ nhớ), 1.000 mẫu đầu (`LIMIT=` để lấy hết), thêm cấu hình `kd-token-mean` (KD chuẩn hóa theo token toàn batch như baseline, các cấu hình khác chuẩn hóa theo mẫu) và student zero-shot khi eval; eval dùng prompt TSD-KD (`PROMPT_STYLE`). Cài thư viện bằng `pip install -r requirements.txt`. Student vẫn dùng LoRA như trước (learning rate 1e-4, khác với fine-tune toàn bộ ở bài TSD-KD). Bộ eval vẫn là GSM8K, MATH500, AIME24, AIME25; các benchmark còn lại (GSM-Plus, Minerva Math, MMLU-Pro-Math, ...) chưa có trong `benchmark_eval.py`.

## 7. Dữ liệu tái sử dụng

Xem `plans/reports/artifacts-index-261026-1250.md`: gain, chỉ mục span, kết quả train/eval của GSM8K 1k nằm ở `work/archive/gsm8k-1k/` (máy) và volume Modal `alerd-ctkd-results` của workspace cũ (cùng 9 checkpoint đã gộp LoRA). Gain gắn với đúng văn bản (prompt + lời giải) qua mã băm: đổi prompt hoặc dữ liệu thì phải chấm lại.
