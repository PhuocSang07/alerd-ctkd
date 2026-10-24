# ALERD-CTKD: phase 2, 4, 5, 6, 7 (phần chạy được trên CPU)

Phase 1 và 3 nằm trong `cook-261025-0044-alerd-ctkd-phase-1-3.md`. Không sửa file nào trong
D:\SpectralGuidedLearning; `sgl` chỉ được gọi như thư viện.

## Kết quả

Toàn bộ code của phase 2, 4, 5, 6, 7 đã viết và kiểm trên CPU: `python -m pytest tests` cho 89 test đạt và 1 bị bỏ
qua (Qwen3 cần transformers mới) với transformers 4.44.2, và 90 test đạt với transformers 4.53.3. Không có gì được
kiểm trên GPU, và không có số huấn luyện hay eval nào ở đây là kết quả thật.

| Phase | Nội dung | Kiểm bằng |
|---|---|---|
| 2 | `layer_capture.py`: SDPA mọi layer, hook lấy hàng attention của token thật cuối (RoPE, GQA, pad); bỏ `[:, -1]`; fp32 + autocast bf16; LoRA mọi projection; gradient checkpointing; KL chỉ trên hàng `lm_head` vocab chung; CE che prompt và giám sát EOS; checkpoint HF | hàng attention từ hook = `outputs.attentions` eager (Qwen2, Llama, Qwen3), có pad; KL vocab chung = cắt logit đầy đủ; checkpoint nạp lại bằng `AutoModelForCausalLM` và cho cùng logit |
| 4 | `span_weights.py`, `kd_losses.py`, `llm_train.py`: nguồn `attention`, `uniform`, `alerd`, `random`, `predictability-easy/hard`; collator đọc gain, kiểm hash và số span; log phân vị ω và budget | `attention` cho đúng ba loss của mã SRA gốc (allclose); `alerd` với λ=0 bằng `uniform` (allclose); hash hoặc số span lệch thì dừng; gradient chỉ tới student và projection |
| 5 | `scripts/alerd_ctkd_eval.sh` gọi `sgl.eval.evaluate` với protocol ALERD (không viết evaluator mới) | chỉ kiểm cú pháp bash |
| 6 | `scripts/alerd_ctkd_train.sh`, `alerd_ctkd_run_all.sh`, `collect_results.py` (bảng LaTeX như Table 2) | mọi method sinh tham số mà `Arguments` nhận; bảng biên dịch được bằng pdflatex |
| 7 | ablation Table 3–5 trong script và `collect_results.py --table ablations`; `analyze_weights.py` (Spearman trong mẫu so với hai xếp hạng ngẫu nhiên, attention dồn vào đâu, ví dụ tô màu) | dữ liệu tổng hợp và dump thật từ một lần train CPU |

Ngoài ra: `score_predictability.py` (điểm cho nguồn predictability), `scripts/score_gain.sh` (dữ liệu và gain theo từng
student), README mới, và không còn token HF trong script nào (đọc từ biến môi trường `HF_TOKEN`).

## Chạy thử với dữ liệu và tokenizer thật

Tôi định chạy một lần train CPU trên dữ liệu s1K thật với tokenizer thật của R1-Distill-Qwen-1.5B và gpt2, model nhỏ
ngẫu nhiên. Lần chạy bị hệ thống dừng vì máy thiếu bộ nhớ (logit trên vocab 151k × chuỗi 2k trên CPU), nên không có kết
quả và tôi không chạy lại. Phần dữ liệu thật đã kiểm riêng ở phase 1 (chỉ mục span trên 914 mẫu với hai tokenizer thật).
Còn lại chưa kiểm: vòng train đầy đủ với tokenizer thật, nên bước đầu tiên trên GPU nên là vài step để bắt lỗi sớm.

## Chỉ kiểm được trên CPU

- Toán và luồng dữ liệu: hàng attention, các loss, trọng số, span, chỉ mục, checkpoint, bảng, phân tích. Mô hình thử là
  model nhỏ khởi tạo ngẫu nhiên.
- Chưa biết: bộ nhớ đỉnh và thời gian mỗi step với chuỗi ~2.2k token, KL trên (số span × vocab chung) có vừa bộ nhớ không,
  loss có giảm trong 50 step không, gain thật, chất lượng student sau train, số eval.

## Lệnh GPU cho user, theo thứ tự

Đặt `HF_TOKEN` (nếu model cần) và `SGL_SRC=<đường dẫn>/SpectralGuidedLearning/src`.

1. Dữ liệu, gain, predictability cho ba student (cần vLLM): `bash scripts/score_gain.sh`
   Đầu ra `work/<student>/{train-segmented.jsonl, gain.json, predictability.json, train-sra.jsonl}`. Xem thống kê gain
   (tỉ lệ gain dương ở bước chứa đáp án, độ tập trung ở 3 bước đầu) và so với ALERD trước khi train.
2. Chọn teacher và đặt `TEACHER`, `TEACHER_LAYER` (đếm từ 1), `TEACHER_DIM`. Tokenizer khác Qwen, kiến trúc kiểu Llama.
3. Pilot một student (R1-1.5B), một seed:
   `DUMP_SPAN_WEIGHTS=1 bash scripts/alerd_ctkd_train.sh r1-qwen-1.5b alerd-ctkd`
   Lần đầu sẽ dựng `spans-<teacher>.jsonl` và dừng nếu mẫu nào mất `\boxed{}` do cắt (khi đó tăng `MAX_LEN`). Xem
   `outputs/r1-qwen-1.5b/alerd-ctkd/train_stats.jsonl`: `cosine`, `geometry`, `kl`, `ce` giảm và không NaN, cùng
   `w_p10/p50/p90` và `budget`. Nên chạy vài step trước để đo bộ nhớ và thời gian.
4. Eval zero-shot (đối chiếu 36.92 / 49.75 của ALERD), teacher, rồi các checkpoint:
   `bash scripts/alerd_ctkd_run_all.sh r1-qwen-1.5b main` (zero-shot, teacher, SRA, SRA-uniform, ALERD-CTKD).
5. Phân tích trọng số (CPU, sau bước 3): `python analyze_weights.py --dump ... --spans ... --data ... --output ...`
6. Sau khi pilot ổn: `bash scripts/alerd_ctkd_run_all.sh qwen25-7b main`, `... qwen3-8b main`, rồi
   `bash scripts/alerd_ctkd_run_all.sh r1-qwen-1.5b ablations`.
7. Bảng: `python collect_results.py --results-dir results --table main --output tables/main.tex` và `--table ablations`.
   Các dòng SFT và ALERD (black-box) lấy từ eval của `sgl` với tag `<student>-sft`, `<student>-alerd`; dòng thiếu bị
   bỏ qua và báo ra stderr.

## Quyết định và lệch so với plan

- Lớp cuối của teacher và student đi qua norm cuối trước khi so sánh và trước `lm_head`, giống `output_hidden_states` của
  HF. Mã SRA gốc dùng đầu ra layer chưa qua norm cho student; chỉ khác khi layer distill là layer cuối.
- Hàng attention tính ngoài autocast. Trong autocast, lệnh chiếu q/k chạy dưới `no_grad` làm cache của autocast giữ bản
  bf16 không có grad của trọng số, forward thật dùng lại bản đó: gradient về trọng số mất âm thầm và gradient
  checkpointing báo lỗi. Đã có test chặn lỗi này.
- Bỏ student và teacher không phải decoder kiểu Llama (GPT-2, OPT) và vòng eval Dolly trong train. Ba script cũ
  `mistral_tinyllama.sh`, `qwen2_5_gpt2_1_5B.sh`, `qwen2_5_opt.sh` chỉ được gỡ token HF và không còn chạy được như trước.
- Student không thêm BOS khi token hoá (chat template đã có), khớp cách sgl train và eval. Tokenizer giữ EOS gốc, chỉ
  đặt pad khi thiếu, và `padding_side = right` bị ép.
- Collator lấy `example_ids` là `id` trong jsonl (bằng chỉ số khi không có `id`).
- CE trung bình theo token trong từng micro-batch rồi chia cho số bước tích luỹ, không phải trung bình theo token của cả
  batch hiệu dụng như `Z` của ALERD; khác biệt nhỏ, chưa sửa.
- Chưa bật cắt gradient: plan không yêu cầu.

## Câu hỏi còn mở

1. Teacher: cần chốt để có `TEACHER_LAYER` và `TEACHER_DIM`.
2. Với Qwen2.5-7B và Qwen3-8B, KL trên khoảng 1.8k span × vocab chung có thể rất tốn bộ nhớ; nếu hết bộ nhớ, cần chia KL theo khối span.
