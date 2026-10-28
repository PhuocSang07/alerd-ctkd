# Rà soát code và thiết lập method, thực nghiệm (span-weighted CE + KD)

Ngày: 2026-10-28. Phạm vi: `prepare_sra_data.py`, `word_spans.py`, `span_index.py`, `token_gain.py`, `answer_gain.py`, `allocation.py`, `span_kd_train.py`, `benchmark_eval.py`, `scripts/*.sh`. Kết quả test CPU: 290 passed. Chưa chạy GPU cho dữ liệu TSD-KD.

## 1. Kết luận ngắn

Phần lõi (span, gain, trọng số, loss) triển khai đúng với mô tả method. Có 3 điểm lệch cần biết, 1 lỗi nhỏ đã sửa và 4 việc chưa làm (mục 6).

## 2. Method và nơi triển khai

| Bước | Định nghĩa | Code | Kết quả rà soát |
|---|---|---|---|
| Word span | Chia câu trả lời thành span từ (số có dấu phân cách, `\w+`, hoặc 1 ký tự khác), không vắt qua dòng; biên span là chỗ biên token trùng biên từ | `word_spans.py`, `span_index.py --span-mode word` | Đúng. Trên 143 mẫu thật: 282 span/mẫu (trung vị), 1 token/span (trung vị), p95 2, tối đa 12; 1/44.048 span vắt qua biên bước (dòng); các span liền nhau, không hở |
| Answer gain | `b_t = log p(đáp án | prompt, y_<=t, probe)`, probe `"\n\nTherefore, the final answer is \boxed{"`, chấm các token đáp án và `}`; `g_t = b_t - b_(t-1)`; gain span `[a,b)` = `b_b - b_a` | `token_gain.py records`, `answer_gain.py`, `token_gain.py pack`, `span_gains` | Đúng. Mỗi biên token một lần chấm (kể cả t=0); gain span cộng đúng theo tổng tiền tố; probe vượt context được bỏ và kế thừa giá trị trước (gain 0) |
| Hai tín hiệu | `u^S` (student, cho CE), `u^T - u^S` (cho KD) | `token_gain.py signals` (`student`, `teacher`, `diff`) | Đúng. Teacher chấm trên prompt của teacher, student trên prompt của student; hash của văn bản được kiểm tra khi huấn luyện |
| Trọng số | `r_n = exp(clip((z-mean)/std, -c, c)/tau)`; `w_n = (1-lambda) + lambda * N/sum(r) * r_n`; `omega_n = w_n/N` | `allocation.trace_weights` (khối lượng mỗi span bằng 1), `span_kd_train.span_omega` | Đúng và tổng `omega` của mẫu bằng 1. Mặc định tau=1, c=2, lambda=0,5. Nếu std nhỏ hơn eps thì trọng số đồng đều |
| Đối chứng | `token`: `omega_n = m_n/T`; `uniform`: `1/N` | `span_omega` | Đúng |
| Loss | `L = mean_mẫu sum_n [(1-beta) omega_ce_n CE_n + beta omega_kd_n KD_n]`, CE_n và KD_n là trung bình token trong span; EOS thuộc span cuối | `SpanKDCollator`, `weighted_loss` | Đúng. Vị trí logit `a-1 .. b-2` dự đoán span `[a,b)`, vị trí cuối dự đoán EOS |
| KD | KL thuận trên top-k của teacher, chuẩn hóa lại trên tập đó, nhân nhiệt độ bình phương | `distill_terms` | Đúng; k=50, nhiệt độ 1 |
| Student | LoRA r=16, alpha=16, dropout 0,05, mọi lớp tuyến tính của khối attention và MLP; adapter fp32, matmul bf16; gộp adapter rồi lưu bf16 | `load_models` | Đúng |

Công thức trong `docs/span-kd-proposal-vi.tex` đã được sửa cho khớp với code (trước đó viết khối lượng theo số token và trung bình theo token).

## 3. Thiết lập thực nghiệm (theo TSD-KD)

- **Cặp mô hình:** teacher Qwen2.5-14B-Instruct (đóng băng, bf16), student Qwen2.5-1.5B-Instruct (LoRA). Hai mô hình dùng chung tokenizer (kiểm tra bằng `get_vocab`); logit được cắt về kích thước từ vựng của tokenizer vì embedding hai mô hình có đệm khác nhau.
- **Dữ liệu:** `phuocsang/UltraInteract-math-Qwen2.5-14B-Instruct-Gen` (public): 22.543 mẫu Math CoT lọc từ `Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen` (79.751 mẫu), câu trả lời do teacher 14B sinh (theo README của tác giả TSD-KD, dùng cho chưng cất off-policy). Prompt lấy nguyên từ bản phát hành. Bỏ mẫu dài hơn 1.024 token (prompt + lời giải, cùng tokenizer) còn 21.347 mẫu. Không lấy Math PoT (22.718 mẫu). `LIMIT=K` giữ K mẫu đầu theo thứ tự; không đặt thì lấy hết.
- **Huấn luyện:** LoRA, AdamW, lr 1e-4 cosine xuống 1e-5, warm-up 10%, weight decay 0,01, clip gradient 1,0, 2 epoch, batch 64 (micro-batch 4), độ dài tối đa 1.024, beta 0,5, seed 10.
- **Cấu hình:** `sft`, `kd-token`, `gain-ce`, `gain-kd`, `gain-both` (mặc định); `kd-uniform`, `gain-student`, `gain-diff`, `gain-both-lam1` (phụ).
- **Đánh giá hiện có:** GSM8K, MATH500 (3 mẫu), AIME24, AIME25 (16 mẫu) ở 4.096 token; lấy mẫu temperature 0,6, top-p 0,9, repetition penalty 1,05, seed 42; báo cáo pass@1, pass@3 không chệch, tỷ lệ cắt cụt, độ dài trung bình. Chấm bằng Math-Verify.
- **Baseline khác:** chạy bằng mã nguồn riêng của từng bài (SKD, SeqKD, DistiLLM, MiniLLM, GKD, Speculative KD, TSD-KD, AMiD, CSD, ALERD).

## 4. Kiểm tra dữ liệu thật (CPU)

- `extract_boxed` (dùng khi chấm gain) và `last_boxed` (dùng khi lọc dữ liệu) cho cùng đáp án trên cả 22.543 mẫu (0 lệch).
- Đáp án ngắn: trung vị 2 ký tự, p95 12, tối đa 159; 10,7% không phải số.
- Đáp án xuất hiện lại nhiều lần trong lời giải ở 97,1% mẫu. Hệ quả: log-xác suất của đáp án có thể đã cao từ giữa lời giải, nên gain tập trung ở một số bước chứ không phân bố đều. Đây là tính chất của tín hiệu, không phải lỗi.
- Chi phí chấm gain: khoảng 372.600 lần chấm mỗi mô hình cho 1.000 mẫu (đo trên 143 mẫu thật).

## 5. Điểm lệch và rủi ro

1. **Loss trung bình theo mẫu, không theo token toàn batch.** Mỗi mẫu có tổng khối lượng 1 nên mẫu ngắn và mẫu dài đóng góp bằng nhau. Baseline thông thường (SFT/KD trong thư viện) lấy trung bình trên mọi token của batch, mẫu dài nặng hơn. Vì vậy `kd-token` của mình không đồng nhất với KD token-mean của các bài khác, và khác biệt giữa `kd-token` và baseline ngoài có thể một phần đến từ cách chuẩn hóa này.
2. **Prompt khi đánh giá khác prompt khi huấn luyện.** Huấn luyện dùng prompt TSD-KD ("Solve the following math problem step-by-step ... \boxed{Your Answer}"); `benchmark_eval.py` vẫn dùng câu lệnh ALERD ("Please reason step by step, and put your final answer within \boxed{}").
3. **LoRA so với fine-tune toàn bộ của baseline.** Baseline trong bài TSD-KD fine-tune toàn bộ, learning rate 5e-6; phương pháp này dùng LoRA. Khi so sánh phải ghi rõ; chênh lệch có thể đến từ LoRA.
4. **Gradient accumulation:** phần dư cuối epoch chia cho `accum` đầy đủ dù ít micro-batch hơn, nên bước cuối nhỏ hơn một chút. Ảnh hưởng không đáng kể.
5. **Beta chưa được chọn trên validation.** Proposal ghi sẽ chọn beta trên tập validation; code hiện đặt cố định 0,5 và chưa có bước tách validation.

## 6. Việc đã sửa trong lượt này và việc chưa làm

Đã sửa: `scripts/score_gain.sh` không còn chạy `score_predictability.py` khi `SPAN_MODE=word` mà không đặt `LIMIT` (bước đó không dùng ở nhánh word-span và sẽ tốn thêm GPU khi chấm toàn bộ dữ liệu).

Chưa làm:
- Kết quả chấm gain chỉ được lưu khi shard chạy xong, nên bị ngắt giữa chừng sẽ mất hết.
- Script Modal huấn luyện chưa có tham số seed.
- Bộ đánh giá thiếu GSM-Plus, Minerva Math, MMLU-Pro-Math, MMLU-STEM, SciQ, MBPP, BBH.
- Các script shell mới (`span_kd_*.sh`, `prepare_tsdkd.sh`, nhánh `DATASET=tsdkd` của `score_gain.sh`) chưa chạy trên GPU thật; chỉ được kiểm tra bằng test với `PYTHON=echo` và bước chuẩn bị dữ liệu trên CPU.
- Chưa có kết quả thực nghiệm nào trên dữ liệu TSD-KD; kết quả duy nhất hiện có là thí điểm 1.000 bài GSM8K (LoRA, một seed), trong đó mọi cấu hình chưng cất đều thấp hơn student zero-shot và các cấu hình KD chênh nhau trong phạm vi nhiễu.

## 7. Câu hỏi mở

- Dùng prompt nào khi đánh giá: prompt TSD-KD hay ALERD?
- Chuẩn hóa loss: giữ trung bình theo mẫu hay đổi sang trung bình theo token toàn batch để khớp baseline?
- Số mẫu chấm gain: 1.000 hay 2.000 mẫu đầu?
