# Dữ liệu tái sử dụng (gain, chỉ mục span, kết quả, checkpoint)

Cập nhật 26/10/2026. Không có file nào dưới đây nằm trong git (`work/` được bỏ qua); chúng nằm trong volume Modal và/hoặc bản sao cục bộ.

## Bản sao cục bộ, `D:\SRE\work\archive\gsm8k-1k\` (khoảng 416 MB)

GSM8K train, 1.000 bài đầu, Qwen2.5-14B-Instruct (teacher) và Qwen2.5-1.5B-Instruct (student), span theo từ.

| Đường dẫn | Nội dung |
|---|---|
| `gsm8k-clean.jsonl` | 1.000 bài đã làm sạch (lời giải, đáp án, `step_char_ends`) |
| `Qwen2.5-1.5B-Instruct/train-sra.jsonl` | dữ liệu train có prompt của student và của teacher |
| `Qwen2.5-1.5B-Instruct/spans-Qwen2.5-14B-Instruct.jsonl` | chỉ mục span theo từ (token của student và teacher, `step_of_span`) |
| `Qwen2.5-1.5B-Instruct/span-gains.json` | gain mỗi span: `student`, `teacher`, `diff` (đầu vào của `span_kd_train.py`) |
| `Qwen2.5-1.5B-Instruct/gain-tokens-*.json`, `gain-raw-*.json` | gain từng token của student |
| `Qwen2.5-14B-Instruct/gain-tokens-*.json`, `gain-raw-*.json` | gain từng token của teacher (42 phút A100 để chấm lại) |
| `train-e1k/<tên>/` | kết quả 9 cấu hình lần 1: `train_stats.jsonl`, `spans.jsonl.gz` (CE, KD, khối lượng theo span và epoch), `result.json`, eval GSM8K/MATH500/AIME24 ở 1.024 token |
| `train-e1k-math/<tên>/` | lần chạy lại có thêm MATH500 và AIME24 (1.024 token) |
| `train-e1k-ckpt/<tên>/` | lần chạy lưu checkpoint, cùng loại kết quả |
| `eval4096-e1k-ckpt/<tên>/` | eval 4096 token: MATH500, AIME24, AIME25 có pass@3, kèm sinh thô |

Ngoài ra: `work/pilot-200/span-gains.json` (200 bài đầu), `work/e1k/` (bản sao rời của một vài file trên).

## Volume Modal

Workspace cũ (profile `tranphuocsang912`), volume `alerd-ctkd-results`:
- `work-limit200/`, `work-limit1000/` như bảng trên.
- `work-limit1000/ckpt-e1k-ckpt/<tên>/`: **9 checkpoint** đã merge LoRA (bf16, khoảng 3 GB mỗi cái): `sft`, `kd-token`, `kd-uniform`, `gain-ce`, `gain-kd`, `gain-both`, `gain-student`, `gain-diff`, `gain-both-lam1`. Chưa tải về máy. Eval thêm benchmark bằng `scripts/modal_eval_ckpt.py` (chạy được trong workspace nào có volume đó).
- Volume `hf-cache`: trọng số Qwen2.5-1.5B và 14B.

Workspace mới (profile `phuocsang07`), volume `alerd-ctkd-results`, thư mục `work-s1k/` (đang tạo):
- `work-s1k-shard<i>of3/`: kết quả từng shard khi chấm gain.
- `work-s1k/Qwen2.5-1.5B-Instruct/{train-sra.jsonl, spans-Qwen2.5-14B-Instruct.jsonl, span-gains.json, gain-tokens-*.json}` và `work-s1k/Qwen2.5-14B-Instruct/gain-tokens-*.json`: sau khi gộp shard.
- `work-s1k/train-s1k/<tên>/` và `work-s1k/ckpt-s1k/<tên>/`: kết quả và checkpoint của 5 thực nghiệm s1K.

## Cách dùng lại

- Train lại một cấu hình trọng số khác mà không phải chấm gain: `python span_kd_train.py --train-data .../train-sra.jsonl --spans-file .../spans-*.jsonl --gains-file .../span-gains.json ...` (xem README).
- Chỉ đổi prompt hoặc dữ liệu thì phải chấm lại gain, vì gain gắn với văn bản (prompt + lời giải) qua mã băm.
