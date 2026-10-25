# Span theo từ và gain theo span trên GSM8K (Qwen2.5-14B-Instruct → Qwen2.5-1.5B-Instruct)

Phần CPU xong và chạy được; lần chạy thử 200 bài trên Modal (A100-80GB) đi hết đường từ dữ liệu, chấm gain của student và
teacher, đến span theo từ và hai tín hiệu. Chưa chạy toàn bộ 7.473 bài và chưa có số đo chi phí đủ 1.000 bài (xem "Chưa làm được").
Không sửa `D:\SpectralGuidedLearning`; `sgl` không còn được dùng (gain, eval và chấm điểm đã nằm trong repo này).

## Việc đã làm

- **Span theo từ** (`word_spans.py`, mới). Từ là số có dấu phân cách (`1,200`, `3.5`), một chuỗi `\w+`, hoặc một ký tự khác;
  khoảng trắng phía trước thuộc từ sau. Tách dòng trước khi tách từ, và dấu xuống dòng thuộc từ cuối của dòng, nên không span
  nào vắt qua dòng (bước). Biên span là chỗ biên token trùng biên từ, tính bằng đúng `longest_common_subsequence` /
  `sample_span_bounds` của `data_utils.py`; token vắt qua nhiều từ thì gộp các từ đó. Phần đuôi sau chỗ khớp cuối được gộp vào
  span cuối, nên mọi token của lời giải thuộc đúng một span. Teacher dùng cùng span với student, dịch theo hiệu độ dài prompt
  (kiểm lỗi nếu hai bên không cùng tokenizer).
- **Chế độ mới** `--span-mode word` ở `span_index.py`, `span_mode` ở `LLMDataCollator`, `--span_mode` ở `Arguments` và
  `llm_train.py`. Một hàm duy nhất (`word_pair_bounds`) cho cả chỉ mục offline và collator, nên hai bên trùng nhau.
- **Hai tín hiệu** (`token_gain.py signals`): gain span của student (cho CE) và teacher − student (cho KL), kiểm hash văn bản và
  khoảng token của từng file gain. Xuất `span-gains.json`: `{id: {hash, teacher_hash, student, teacher, diff}}`.
- **Chấm gain**: `scripts/score_gain.sh` thêm `SPAN_MODE=word`, `LIMIT=N` (đo chi phí, ghi vào `<WORK_DIR>-limitN`, bỏ predictability),
  `GPU_MEMORY_UTILIZATION`, in thời gian từng lần chấm. `answer_gain.py` in số probe và thời gian, tắt thanh tiến trình, thêm
  `--max-num-batched-tokens` (32768) và `--max-num-seqs` (512). `span_index.py` in thêm token mỗi span và số probe ước tính.
- **Chạy trên Modal**: `scripts/modal_score_gain.py` (A100-80GB, vLLM 0.10.1.1, `transformers==4.55.4`, kết quả lưu vào volume
  `alerd-ctkd-results`). `.gitattributes` giữ LF cho file `.sh`.
- `requirements.txt` có thêm `math-verify`; README có mục "Same-tokenizer word spans".

## Test

`python -m pytest tests` (HF offline): **243 passed, 0 failed, 0 skipped**. Test mới:

- `tests/test_word_spans.py` (16 test): các ca từ (`1,200`, `3.5`, `\boxed{72}`, xuống dòng); câu mẫu ra đúng 11 span với tokenizer
  Qwen2.5; `$1,200` và `3.5` là một span; mọi token thuộc đúng một span, không span rỗng, đuôi khoảng trắng được phủ; không span
  nào vắt qua bước; span offline bằng span của collator (token của student và teacher, hash, số span) trên batch có pad; từ chối
  `span_mode` lạ và hai tokenizer khác nhau; tổng gain span bằng `b_cuối − b_đầu` và bằng gain tính trực tiếp; `signals` đúng và dừng
  khi gain chấm trên văn bản khác.
- `tests/test_scripts.py` (+2 test): `SPAN_MODE=word LIMIT=5` chấm student và teacher, bỏ predictability, dựng chỉ mục và xuất tín hiệu;
  `SPAN_MODE` lạ bị từ chối.

## Thống kê (GSM8K train, 7.473 bài, chỉ tokenize, Qwen2.5-1.5B; tokenizer của 1.5B, 7B và 14B giống hệt nhau, đã so sha1 bốn file)

| | span theo từ | span SRA (token) |
|---|---|---|
| span mỗi mẫu | trung vị 64, p95 130, max 250 (trung bình 70,4) | trung vị 88, p95 178, max 347 |
| token mỗi span | trung vị 1, p95 3, max 14 (trung bình 1,37); 79% span là một token | 1 |
| tổng token lời giải / tổng span | 719.399 / 526.360 | 719.399 / 719.399 |
| span vắt qua bước | 0 / 526.360 | 0 |
| bị cắt ở max_len 512 | 0 mẫu (dài nhất 492 token) | 0 |

Số probe cho mỗi mô hình chấm (mỗi biên một probe, kể cả điểm đầu): **726.872** nếu chấm ở mọi token (cách `answer_gain.py` đang làm,
cho phép tính gain mọi đơn vị từ cùng một lần chấm), **533.833** nếu chỉ chấm ở biên span (nhỏ hơn khoảng 27%; chưa cài chế độ này).

## Lần chạy thử trên Modal (200 bài đầu, A100-80GB)

Chạy xong từ đầu đến cuối, `returncode 0`. Lời giải 200 bài có 19.596 token, 14.390 span, 0 span vắt qua bước, 19.796 probe mỗi mô hình.

| mô hình chấm | thời gian (gồm nạp model) |
|---|---|
| student 1.5B | 109 s (133 s kể cả tokenize) |
| teacher 14B | 552 s (573 s), lần đầu nên gồm cả tải trọng số |

Gain mỗi span trên 14.390 span: student trung bình 0,044 (độ lệch chuẩn 2,19; 52,2% dương); teacher 0,017 (2,54; 49,5%);
teacher − student −0,027 (3,10; 48,8%). Tổng mỗi mẫu: student 3,16, teacher 1,22, hiệu −1,94. Tương quan Pearson giữa gain span của
student và của teacher là 0,145. Đây chỉ là mô tả trên 200 bài; chưa diễn giải thêm.

## Chưa làm được

- **Đủ 1.000 bài với teacher 14B**: lần chạy 1.000 bài (trước khi tăng tốc) chấm xong student (95.751 probe trong 457 s) nhưng teacher mới
  được 55% sau 23 phút (khoảng 45 probe/giây) thì máy cục bộ hết RAM và tiến trình `modal run` bị tắt; tôi tắt container còn lại. Vì chạy
  trước khi tăng `max_num_batched_tokens`, số này cao hơn thực tế sau tối ưu. Chưa có số thời gian của teacher 14B sau tối ưu (lần 200 bài
  gộp cả thời gian nạp và tải model, không tách được).
- **Toàn bộ 7.473 bài**: chưa chạy. Ước lượng thô theo lần 1.000 bài trước khi tối ưu: teacher 14B cỡ 5 giờ; cần đo lại sau tối ưu.
- Gain ở biên span chứ không ở mọi token: chưa cài; xem số probe ở trên.
- Eval, huấn luyện CE + KL có trọng số, on-policy, Gemma, UltraInteract, Dolly: ngoài phạm vi.
- Chưa đối chiếu với tokenizer Gemma2.

## Lệnh GPU tiếp theo

Cục bộ có vLLM (Linux) hoặc trên Modal:

```
# đo chi phí 1.000 bài (ghi vào work-limit1000)
STUDENTS=Qwen/Qwen2.5-1.5B-Instruct TEACHER=Qwen/Qwen2.5-14B-Instruct SPAN_MODE=word LIMIT=1000 \
  GPU_MEMORY_UTILIZATION=0.85 bash scripts/score_gain.sh
# toàn bộ GSM8K train
STUDENTS=Qwen/Qwen2.5-1.5B-Instruct TEACHER=Qwen/Qwen2.5-14B-Instruct SPAN_MODE=word \
  GPU_MEMORY_UTILIZATION=0.85 bash scripts/score_gain.sh
# trên Modal, chế độ detached để client cục bộ chết thì job vẫn chạy (--limit 0 là toàn bộ)
modal run --detach scripts/modal_score_gain.py --limit 1000
modal volume get --force alerd-ctkd-results work-limit1000/Qwen2.5-1.5B-Instruct/span-gains.json span-gains.json
```

Phiên điều phối đã dặn không chạy Modal nếu user chưa yêu cầu; user đã yêu cầu rõ trong phiên này. Chi phí Modal đã dùng: ba lần
dựng image và chạy (một lần lỗi `transformers`, một lần lỗi CRLF của file `.sh`, một lần bị tắt giữa chừng) và lần 200 bài.

## Câu hỏi còn mở

1. Có cần chế độ chấm gain chỉ ở biên span (giảm khoảng 27% probe) không, hay giữ chấm ở mọi token để còn ablation span một token và đơn vị bước?
2. Chạy toàn bộ 7.473 bài ngay, hay đo chi phí 1.000 bài sau tối ưu trước?
3. Phiên điều phối cần quyết định nhánh nào nhận `feat/modal-gain-scoring` (đã được push bởi phiên điều phối) và đường dẫn merge vào `main`.
