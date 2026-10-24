# ALERD-CTKD: phase 1 và phase 3

Branch `feat/alerd-ctkd` trong D:\SRE, chưa commit. Không sửa file nào trong D:\SpectralGuidedLearning.
Báo cáo này nằm ở D:\SRE\plans\reports vì tin nhắn cập nhật cấm ghi vào repo kia.

## Kết quả

Mã và test của cả hai phase xong; 18 test đạt (`python -m pytest tests`). Một mục thống kê của phase 3 chưa
chạy được (xem "Chưa làm được").

### Phase 1: dữ liệu, chỉ mục span, ánh xạ span → bước

- `prepare_sra_data.py`: lệnh `filter` giữ mọi dòng s1K-1.1 có `deepseek_attempt` khác rỗng và trích được
  `\boxed{}` cuối, không lọc theo `deepseek_grade`; lệnh `convert` đổi `train-segmented.jsonl` sang jsonl của
  SRA (`id`, `prompt`, `output`, `step_char_ends`).
- `scripts/score_gain.sh`: với từng student của ALERD (R1-Distill-Qwen-1.5B, Qwen2.5-7B-Instruct, Qwen3-8B) chạy
  `sgl.data.prepare --chat-template --no-enable-thinking --palign-prompt`, `sgl.signals.answer_gain` và
  `convert`. Đã kiểm tra cú pháp (`bash -n`), chưa chạy vì cần GPU và vLLM. Token HF lấy từ biến môi trường `HF_TOKEN`.
- `data_utils.py`: tách `sample_span_bounds` (tính span cho một mẫu) dùng chung cho collator và offline;
  `LLMDataset` trả `index`; collator trả `example_ids`; `N_SPAN` thành tham số `max_spans`, mặc định `None`
  (không subsample). Collator báo lỗi nếu `padding_side` không phải `right`.
- `student.py`: thêm một dòng bỏ `example_ids` trước khi gọi model để huấn luyện hiện tại không vỡ.
- `span_index.py`: ghi `spans.jsonl` (`id`, `hash`, `char_ends`, khoảng token hai phía, `step_of_span`,
  `crosses_step`).
- Collator giờ cắt offset theo độ dài thật (attention mask) trước khi ghép, để span không phụ thuộc phần pad.

Thống kê trên dữ liệu thật, chạy với `--no-enable-thinking --palign-prompt`, teacher thử là gpt2:

| Mục | Qwen2.5-0.5B-Instruct (chỉ để test) | R1-Distill-Qwen-1.5B |
|---|---|---|
| Dòng s1K-1.1 | 1000 | 1000 |
| Giữ lại (có `\boxed{}` trích được) | 914 | 914 |
| Bỏ vì không có `\boxed{}` | 86 | 86 |
| Giữ lại nhưng `deepseek_grade = "No"` | 315 | 315 |
| Token mỗi mẫu (prompt + lời giải) | trung vị 749, p95 1101, max 2135 | trung vị 728, p95 1080, max 2114 |
| Bước mỗi mẫu | trung vị 7, p95 16, max 24 | trung vị 7, p95 16, max 24 |
| Span mỗi mẫu | trung vị 528, p95 816, max 1796 | trung vị 530, p95 817, max 1797 |
| Span vắt qua biên bước | 117 / 495978 | 117 / 496892 |

`padding_side` của tokenizer R1-Distill-Qwen-1.5B là `left` (Qwen2.5-0.5B và gpt2 là `right`). Collator báo lỗi
với `left`, nên `load_tokenizer` trong `llm_train.py` phải đặt `padding_side = "right"` cho student ở phase 4.
Qwen2.5-7B-Instruct, Qwen3-8B và teacher thật chưa kiểm tra.

### Phase 3: gain và trọng số span

- `allocation.py`: Eq. 3–5 của ALERD với khối lượng là số span của bước; ba kiểu chuẩn hoá (`per-trace`,
  `corpus`, `none`); `span_weights` cho ω = w/N; các nguồn điểm `gain`, `random`, `predictability-easy/hard`,
  `answer-only-unit`.
- Test tái tạo toy table (0.61 / 0.70 / 1.69), tổng ω bằng 1 mỗi mẫu, λ=0 cho ω = 1/N.

## Lệch so với plan

`sgl.data.s1k` không dùng. Đầu ra của nó đi vào `sgl.data.prepare` theo nhánh "LIMO-style", nhánh này nối
thêm `\n</think>` và một câu "The final answer is \boxed{…}" vào lời giải, sai với ý đồ dùng nguyên
`deepseek_attempt`. Lệnh `filter` ghi dòng theo dạng `{question, instruction: "", output}`; nhánh alpaca của
`prepare` đọc nguyên văn `output`, không thêm gì (đã kiểm tra: không có `think` trong response). Nếu vẫn muốn
đi qua `sgl.data.s1k`, cần đổi `prepare`, tức là sửa repo kia.

## Chưa làm được

- Gain cho từng student: `sgl.signals.answer_gain` cần vLLM và GPU, máy này là Windows không có vLLM. Script
  `scripts/score_gain.sh` đã viết sẵn cho server. Nên tỉ lệ gain dương ở bước chứa đáp án và độ tập trung ở 3
  bước đầu chưa có số, và tôi không thay bằng dữ liệu giả.
- `corpus` và `none` không bảo đảm Σω = 1 mỗi mẫu; chỉ `per-trace` đảm bảo (đúng như Eq. 6 của ALERD).

## Câu hỏi còn mở

1. Teacher chưa chốt, nên `padding_side` và span của teacher thật chưa kiểm tra.
2. 86 dòng không có `\boxed{}` bị bỏ và 315 dòng giữ lại có lời giải chấm sai; cần xem có đủ cho bài không.
