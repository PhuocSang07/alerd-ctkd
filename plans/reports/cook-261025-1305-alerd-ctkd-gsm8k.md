# ALERD-CTKD trên GSM8K: Qwen2.5-7B-Instruct dạy GPT2-XL và Llama-3.2-1B-Instruct

Đã đổi code trong `D:\SRE` theo plan mới (data GSM8K, hai student, attention eager, gain theo span với ba nguồn,
> Cập nhật: phần gain, eval và chấm điểm của `sgl` đã được đưa vào repo này (`answer_gain.py`, `benchmark_eval.py`, `answer_scoring.py`); không còn dùng `SGL_SRC`.
eval GSM8K + MATH500). Không sửa `D:\SpectralGuidedLearning`; `sgl` chỉ được import. Mọi thứ chạy được trên CPU đã
kiểm; không có bước GPU nào được chạy, nên chưa có số train, gain hay eval thật.

## Kết quả kiểm tra

`python -m pytest tests` với `SGL_SRC` đặt: **223 đạt, 0 hỏng, 0 bỏ qua** (transformers 4.44.2). Với transformers 4.53.3,
không đặt `SGL_SRC`: 217 đạt, 1 bỏ qua; một test (`test_cli_clean_then_build`) bị loại khỏi lần chạy đó vì thư mục
transformers tạm làm hỏng tương thích numpy/pandas của `datasets`, không phải lỗi mã (test này đạt ở 4.44.2). Lần chạy
4.53.3 diễn ra trước khi thêm `contains_answer` và test `run_sgl_eval`.

Chạy thử trên dữ liệu và tokenizer thật (chỉ tokenize, không tải mô hình):

| | GPT-2 + Qwen2.5 | Llama-3.2 + Qwen2.5 |
|---|---|---|
| Số mẫu | 7,473 | 7,473 |
| Token student, trung vị / p95 / max | 142 / 243 / 430 | 182 / 287 / 470 |
| Token teacher (Qwen, có chat template), trung vị / p95 / max | 189 / 307 / 492 | 189 / 307 / 492 |
| Số bước mỗi mẫu (dòng lời giải), trung vị / max | 4 / 10 | 4 / 10 |
| Span mỗi mẫu, trung vị / p95 / max | 68 / 139 / 284 | 77 / 156 / 298 |
| Span vắt qua biên bước | 0 / 559,782 | 0 / 627,266 |
| Mẫu bị cắt ở `max_len=512`, hay mất `\boxed{}` | 0, 0 | 0, 0 |
| Số probe gain (một probe mỗi biên token) | 591,662 | 627,267 (teacher: 719,399) |

Tokenizer Llama lấy từ bản sao công khai `unsloth/Llama-3.2-1B-Instruct` (bản gốc của Meta cần `HF_TOKEN`), với
transformers 4.53.3 vì tokenizer.json mới không đọc được ở 4.44.2. Không token nào vắt qua ranh giới prompt/output ở
cả ba tokenizer. Mẫu dài nhất của teacher là 492 token, sát giới hạn 512 nhưng chưa bị cắt.

## File đã sửa

| File | Thay đổi |
|---|---|
| `prepare_sra_data.py` | Viết lại: `clean` (GSM8K → bỏ `<<…>>`, `#### N` → `The final answer is \boxed{N}.`, bỏ dấu phẩy hàng nghìn, mỗi dòng một bước) và `build` (bọc prompt theo từng mô hình, kiểm không token nào vắt qua ranh giới prompt/output). Bỏ đường s1K. |
| `prompts.py` (mới) | Một nơi dựng prompt cho data, gain, train và eval: chat template nếu tokenizer có, chuỗi thô nếu không (GPT-2). Cố định dòng ngày của template Llama. |
| `data_utils.py`, `span_index.py` | Teacher có prompt riêng (`teacher_prompt`); toạ độ ký tự của teacher được dịch theo hiệu độ dài hai prompt khi căn span. Cả hai phía token hoá với `add_special_tokens=False`. `spans.jsonl` thêm `teacher_hash`. Collator chạy được không có teacher (SFT). |
| `utils.py`, `teacher_llm.py`, `student.py` | Quay về `output_attentions=True` với eager attention; hàng attention lấy ở token thật cuối từ `attention_mask`. Xoá `layer_capture.py` và `tests/test_last_row_attention.py`. |
| `token_gain.py` (mới) | Dựng bản ghi "mỗi token là một bước" cho `sgl.signals.answer_gain`, đóng gói `gain-tokens-<model>.json` (hash, tên tokenizer, vị trí bắt đầu, gain từng token, gain từng dòng), đọc và kiểm tra file. |
| `span_weights.py`, `allocation.py` | Gain theo span từ tổng gain token; ba nguồn `student|teacher|diff`; hai đơn vị `span|step`; `random`, `predictability`, `answer-only` giữ nguyên. Thêm `contains_answer` (khớp nguyên số, "72" không khớp "172" hay "72.5"). |
| `arguments.py`, `llm_train.py`, `run_distill_llm.py` | Tham số `gain_source`, `gain_unit`, `student_gain_file`, `teacher_gain_file` (bỏ `gain_file`); `max_len` mặc định 512; gradient checkpointing tắt mặc định; `hard_label_loss_weight 1.0` là SFT: không nạp, không chạy teacher; kiểm context và chỉ số layer. |
| `run_sgl_eval.py` (mới) | Thêm GSM8K test vào `sgl.eval` và dựng prompt giống lúc train; `sgl` không bị sửa. |
| `score_predictability.py` | Đọc `line_steps` của bản ghi mới. |
| `collect_results.py` | Cột GSM8K và MATH500, hai student, bỏ dòng ALERD black-box và số zero-shot của ALERD, trung bình theo seed; thêm khối ablation nguồn gain và đơn vị gain. |
| `analyze_weights.py`, `analyze_gain.py` (mới) | Bảng trọng số theo loại span (số, phép tính, từ nối, từ, câu đáp án). `analyze_gain.py` thống kê gain theo vị trí, loại span có gain lớn nhất và Spearman student–teacher. |
| `scripts/score_gain.sh`, `alerd_ctkd_train.sh`, `alerd_ctkd_eval.sh`, `alerd_ctkd_run_all.sh` | Viết lại cho hai student mới; `GAIN_SOURCE`, `GAIN_UNIT`, `MAX_TOKENS`; thư mục ra `outputs/<student>/<method>/seed-<seed>`; thiếu file gain thì dừng với mã 3 và tên file. |
| `README.md` | Viết lại, có ghi chú định dạng prompt cho GPT-2. |

## Test

Thêm: `test_prepare_data.py` (chuyển đổi GSM8K, prompt, ranh giới prompt/output), `test_eager_attention.py`,
`test_token_gain.py`, `test_span_weights.py`, `test_analyze_gain.py`, `test_run_sgl_eval.py`. Viết lại các file còn lại
cho cấu hình mới. Đối chiếu từng yêu cầu kiểm chứng của plan:

- Hàng attention eager ở token thật cuối đúng khi batch có pad, với Qwen2, Llama và GPT-2 nhỏ (so với từng chuỗi chạy riêng); `[:, -1]` là hàng sai cho chuỗi ngắn hơn batch.
- Span khớp giữa chạy một mẫu và chạy trong batch có pad, với ba cặp tokenizer (byte-level/byte-level, kiểu GPT-2 không có pad/byte-level, byte-level/word-piece), kể cả khi prompt của teacher dài khác prompt của student.
- Tổng gain các span trong một bước bằng gain của bước đó, cho cả student và teacher (khi không có span vắt qua biên bước, và đúng như vậy ở dữ liệu thật). `diff` bằng hiệu hai nguồn theo từng span. Thiếu file, hash lệch, vị trí bắt đầu lệch, thiếu id: đều dừng với lỗi nói rõ file nào.
- Toy table ALERD cho 0.61 / 0.70 / 1.69; Σω = 1 mỗi mẫu; λ = 0 cho ω = 1/N với cả ba nguồn gain và cả hai đơn vị.
- `attention` cho đúng ba loss của mã SRA gốc (allclose); CE của đường SFT (sdpa) bằng CE của đường KD (eager) trên cùng trọng số và batch.
- Chạy train đủ cho Qwen2 và GPT-2 nhỏ (full fine-tuning), LoRA + gradient checkpointing, bốn tổ hợp nguồn/đơn vị gain, SFT không nạp teacher; checkpoint HF nạp lại được.
- `run_sgl_eval.main` chạy qua `sgl.eval.evaluate` thật với `generate` giả: GSM8K được đăng ký, prompt là prompt train, `summary.json` đọc được bởi `collect_results.py`.
- Script shell chạy với `echo` thay Python: mọi phương pháp cho tham số mà `Arguments` nhận, thứ tự đánh giá của `run_all`, `MAX_TOKENS` 512/1024.

Các so sánh với `sgl` (prompt, probe, `extract_boxed`) cần `SGL_SRC`; không đặt thì được bỏ qua.

## Lệch so với plan, và điều tự quyết

1. **Teacher nhận prompt theo template của chính nó** (Qwen) thay vì văn bản của student. Plan nói "mỗi mô hình chỉ khác cách bọc template" và gain của teacher đo trên prompt Qwen, nên làm vậy để hai bên nhất quán; đổi lại collator phải dịch toạ độ ký tự của teacher.
2. **Bỏ đường s1K** trong `prepare_sra_data.py` (plan cho phép).
3. **Gain mỗi token là một "bước"** để dùng nguyên `sgl.signals.answer_gain`; gain bước và gain span đều là tổng gain token. Một bước/span vắt qua biên bước được gán vào bước chứa ký tự cuối (cùng quy tắc với chỉ mục span); ở dữ liệu thật không có trường hợp này.
4. **Câu "The final answer is \boxed{N}." là một bước** (bước cuối), nên gain của nó sẽ rất lớn. Đây chính là chỗ ablation `answer-only-unit` (giữ trọng số 1 cho các bước chứa đáp án) có nghĩa.
5. **Dòng ngày của template Llama được cố định** ("26 Jul 2024") cả lúc build data lẫn lúc eval, vì template mới của Llama in ngày hôm nay và sẽ làm prompt train khác prompt eval.
6. **Predictability và answer-only luôn ở đơn vị bước** dù `GAIN_UNIT` là gì (plan: "giữ nguyên"). `random` dùng đơn vị đã chọn.
7. **Phương pháp ablation nguồn gain cố định đơn vị `span`**; `alerd-ctkd-unit-step` giữ nguồn từ `GAIN_SOURCE`. Nguồn mặc định của `alerd-ctkd` là `student` cho tới khi pilot chọn.
8. **Eval không dùng cờ chat template của `sgl`**: `run_sgl_eval.py` thay `build_prompts` để dùng đúng prompt train, nên GPT-2 (không có template) đi qua chuỗi thô.
9. Hiệu chỉnh khác: batch 8 x 4 tích luỹ = 32 (plan: effective batch 32); `hard_label_loss_weight 0.5` cho các dòng KD giữ từ công thức SRA/ALERD; thư mục ra có `seed-<seed>`, tag eval là `<student>-<method>-s<seed>`.
10. Ba script legacy của SRE gốc (`mistral_tinyllama.sh`, `qwen2_5_gpt2_1_5B.sh`, `qwen2_5_opt.sh`) không đụng tới; chúng dùng dữ liệu Dolly không có trong repo và không thuộc đường chạy mới. Student OPT không được hỗ trợ.

## Chỉ kiểm được trên CPU (chưa biết)

- Lệnh `python -m sgl.signals.answer_gain` với vLLM (tham số lấy từ nguồn của nó, đã kiểm định dạng bản ghi đầu vào và hàm `extract_boxed`/probe, chưa chạy vLLM); thời gian và bộ nhớ của ~0.6–0.7 triệu probe mỗi mô hình.
- Bộ nhớ đỉnh và tốc độ train: attention eager cho cả teacher 7B (28 layer) lẫn GPT2-XL (48 layer) ở batch 8 và ~500 token; KL trên span x từ vựng chung. Nếu hết bộ nhớ: giảm `BATCH_SIZE` và tăng `GRAD_ACCUM`.
- Loss có giảm không, giá trị gain thật (có dương ở bước chứa đáp án không), điểm eval, tỉ lệ bị cắt trước `\boxed{}`.
- Math-Verify: gói `math_verify` không có ở máy này nên test eval dùng hàm chấm giả; chấm thật chạy trên máy GPU.
- Tokenizer Llama-3.2 thật cần `HF_TOKEN`: đã kiểm bằng bản sao công khai, chưa bằng repo gated.
- GPT-2: vài prompt MATH500 dài tới 850 token, nên với ngữ cảnh 1,024 chỉ còn ~170 token sinh cho vài bài; số GPT-2 trên MATH500 sẽ thấp.

## Lệnh GPU, theo thứ tự

Đặt trước: `export HF_TOKEN=...` và `export SGL_SRC=<đường dẫn>/SpectralGuidedLearning/src`.

1. Data và gain (vLLM): `bash scripts/score_gain.sh`
   Ra `work/<student>/{train-sra.jsonl,gain-tokens-<student>.json,predictability.json}` và `work/Qwen2.5-7B-Instruct/gain-tokens-Qwen2.5-7B-Instruct.json`.
2. Thống kê gain (CPU), xem trước khi train:
   `python analyze_gain.py --spans work/Llama-3.2-1B-Instruct/spans-Qwen2.5-7B-Instruct.jsonl --data work/Llama-3.2-1B-Instruct/train-sra.jsonl --student-gain work/Llama-3.2-1B-Instruct/gain-tokens-Llama-3.2-1B-Instruct.json --teacher-gain work/Qwen2.5-7B-Instruct/gain-tokens-Qwen2.5-7B-Instruct.json --output plans/reports/analysis-gain.md`
   (`spans-…jsonl` được dựng ở lần chạy train đầu tiên; có thể tạo trước bằng `python span_index.py --data … --student-tokenizer … --teacher-tokenizer Qwen/Qwen2.5-7B-Instruct --max-len 512 --output …`.)
3. Thử bộ nhớ và lỗi riêng của GPT-2, một seed: `bash scripts/alerd_ctkd_train.sh gpt2-xl sra`; xem `outputs/gpt2-xl/sra/seed-42/train.log` và `train_stats.jsonl` (loss giảm, không NaN, `kl` trên số).
4. Pilot trên Llama-3.2-1B, một seed (`main` gồm zero-shot, teacher, SFT, SRA, SRA-uniform và `alerd-ctkd` với
   `GAIN_SOURCE=student` mặc định; hai phương pháp sau thêm hai nguồn còn lại):
   `SEEDS=42 bash scripts/alerd_ctkd_run_all.sh llama-3.2-1b main alerd-ctkd-gain-teacher alerd-ctkd-gain-diff`
   Chọn nguồn gain tốt nhất; **báo user trước khi chốt**.
5. Phân tích trọng số: `DUMP_SPAN_WEIGHTS=1 bash scripts/alerd_ctkd_train.sh llama-3.2-1b alerd-ctkd` rồi `python analyze_weights.py --dump outputs/llama-3.2-1b/alerd-ctkd/seed-42/span_weights.jsonl --spans … --data … --output plans/reports/analysis-weights.md --html example.html`.
6. Bảng chính, ba seed: `GAIN_SOURCE=<đã chọn> bash scripts/alerd_ctkd_run_all.sh gpt2-xl main` và `… llama-3.2-1b main`.
7. Ablation trên Llama-3.2-1B: `GAIN_SOURCE=<đã chọn> bash scripts/alerd_ctkd_run_all.sh llama-3.2-1b ablations`.
8. Bảng: `python collect_results.py --results-dir results --table main --output tables/main.tex` và `--table ablations --ablation-student llama-3.2-1b`.

## Câu hỏi còn mở

1. Nguồn gain mặc định cho bảng chính: tạm là `student` (như ALERD) cho đến khi pilot cho số; cần user chốt sau pilot.
2. Layer distill: teacher layer 28 (cuối) với student layer cuối (48 cho GPT2-XL, 16 cho Llama-3.2-1B). Plan chưa nêu; nếu muốn layer giữa thì đặt `TEACHER_LAYER` và sửa `LAYER` trong `scripts/alerd_ctkd_train.sh`.
3. `hard_label_loss_weight 0.5` (CE và KD cân bằng, như SRA) có đúng ý với plan "CE giữ đều" không? Plan chỉ nói CE không có trọng số span; hệ số 0.5 giữ từ công thức SRA.
4. Với GPT2-XL, nên dùng `MAX_TOKENS=512` như mặc định, hay thấp hơn để tránh cắt các prompt MATH500 dài?
