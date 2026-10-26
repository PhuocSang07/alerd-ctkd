# Thực nghiệm huấn luyện KD có trọng số theo span, 1.000 bài GSM8K (Qwen2.5-14B → Qwen2.5-1.5B)

Chấm gain 1.000 bài xong (student 431 s, teacher 14B 2.502 s, 95.751 probe mỗi model). Chín cấu hình trọng số được train song song trên Modal (A100-80GB mỗi cấu hình) và đánh giá GSM8K test (1.319 bài, 3 mẫu, nhiệt độ 0,6).

Tham số chung: LoRA rank 16 / alpha 16 / dropout 0,05, learning rate 1e-4 (cosine về 1e-5), batch hiệu dụng 32, 3 epoch (96 bước), β 0,5, top-k 50, nhiệt độ 1, seed 10, `max_len` 1.024 / 2.560. β, top-k, nhiệt độ, rank, learning rate và epoch là giá trị tôi tự chọn (bài RIVAD không nêu).

| cấu hình | CE | KD | pass@1 | pass@3 |
|---|---|---|---|---|
| student zero-shot | - | - | 74,48 | 87,19 |
| sft | token | - | 57,44 | 76,35 |
| kd-token | token | token | 69,65 | 85,52 |
| kd-uniform | uniform | uniform | 69,47 | 85,14 |
| gain-ce | student | token | 68,54 | 85,14 |
| gain-kd | token | diff | 70,33 | 85,22 |
| gain-both | student | diff | 69,37 | 85,90 |
| gain-student | student | student | 69,98 | 85,37 |
| gain-diff | diff | diff | 68,74 | 84,38 |
| gain-both-lam1 | student (λ=1) | diff (λ=1) | 68,94 | 84,76 |

Thời gian train mỗi cấu hình 4-9 phút (gồm nạp model).

## Nhận xét

- Mọi cấu hình đều thấp hơn student zero-shot. Dữ liệu 1.000 bài GSM8K có lời giải ngắn kiểu GSM8K, khác cách suy luận mặc định của model Instruct. SFT thuần giảm nhiều nhất (57,4); thêm KD từ teacher giữ lại phần lớn (khoảng 69-70).
- Các cấu hình KD chênh nhau 68,5-70,3 pass@1, trong khoảng nhiễu của một seed (sai số chuẩn khoảng 1,2 điểm). Chưa thể nói trọng số gain tốt hơn hay kém hơn trọng số token/uniform.
- Tín hiệu trong lúc train (`spans.jsonl.gz`, epoch đầu của kd-token, 69.922 span): CE và KD của span tương quan 0,36; CE có tương quan thấp với gain của student (0,06); KD gần như không tương quan với gain `diff` (-0,02). CE trung bình giảm 0,641 → 0,563 và KD 0,372 → 0,313 từ epoch 0 đến epoch 2. Span có gain student cao có CE cao hơn (quintile thấp nhất 0,44, cao nhất 0,83); span có `diff` cao có CE thấp hơn (0,72 → 0,53).

## Chưa làm

- Nhiều seed cho từng cấu hình; một seed không đủ để phân biệt các cấu hình.
- Chấm gain đủ 7.473 bài (ước khoảng 5 giờ cho teacher 14B), dữ liệu UltraInteract, đánh giá các benchmark khác của anh Phong, on-policy.

Kết quả đầy đủ trong volume Modal `alerd-ctkd-results`, thư mục `work-limit1000/train-e1k/`.
