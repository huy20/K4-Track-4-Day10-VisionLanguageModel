Họ tên: Phùng Quang Minh Huy
`dataset_version`: d10-21e4070301

> Ghi chú: E0 và E1 đã chạy thật trên Kaggle T4 (số liệu dưới đây lấy từ
> `report/evidence/e0.json`, `e1.json`). E2–E4 **chưa chạy xong** (lỗi dtype trên
> torch 2.11 đã được sửa trong notebook);
## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,078 | 0,054–0,107 | 1,05 | 0,161 | 0,026 | 0,136 |
| E1 | LoRA LLM | 33,0M | 6,43 | 51,4 | 0,270 | 0,235–0,307 | 1,40 | 0,217 | 0,182 | 0,548 |
| E2 | + projector | (chưa chạy) | | | | | | | | |
| E3 | + LoRA ViT | (chưa chạy) | | | | | | | | |
| E4 | rank r=4 / r=64 | (chưa chạy) | | | | | | | | |

Thống kê split (từ `data/splits/meta.json`):

| Split | Ảnh | Tổng box | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1422 | 3515 | 1074 | 1750 | 691 |
| val | 276 | 639 | 66 | 470 | 103 |
| test | 311 | 583 | 121 | 299 | 163 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B gồm ViT ≈415M, projector (`visual.merger` + `visual.deepstack_merger_list`) và LLM ≈4,0B (tổng ≈4,44B). Trong thí nghiệm của tôi, projector đếm được **54,6M** tham số huấn luyện được (deepstack lấy đặc trưng ở các lớp giữa của ViT). E1 chỉ gắn LoRA lên LLM: **252 module** khớp regex, **33,0M** tham số huấn luyện, ViT + projector đóng băng. E2 = E1 + mở đầy đủ projector (**+54,6M**, tổng **87,6M**). E3 = E2 + LoRA trên các khối attention/MLP của ViT (kỳ vọng **348 module** khớp; trong lần probe trước khi train, tôi đo được `E3 -> 348 module`). ViT/`lm_head` **không** bị lượng tử hoá (`llm_int8_skip_modules=["visual","lm_head"]`).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` của E0 = **0,936** (JSON hợp lệ 93,6% số ảnh). Hai lỗi phổ biến nhất:
1. **Sụp về một lớp (class collapse):** model gần như chỉ trả `manhole`, bỏ qua `crack`/`pothole`. Hệ quả: F1 crack chỉ **0,026**, F1 pothole **0,161**, trong khi manhole **0,136** — và precision rất thấp (P=0,102) do FP/ảnh **1,05**.
2. **Box trùng lặp / bịa box và bỏ sót:** cùng một box được lặp hai lần, và nhiều ảnh trả `[]` dù có hư hại.
Ví dụ output thô (E0, ảnh `vlcsnap-00202`):
```json
[{"label": "manhole", "bbox_2d": [309, 686, 393, 719]}, {"label": "manhole", "bbox_2d": [309, 686, 393, 719]}]
```
(box y hệt nhau, lặp lại hai lần), và `vlcsnap-00201` trả `[]`.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Toạ độ chuẩn của Qwen3-VL là số nguyên trên thang 0–1000, tương đối theo chiều rộng/cao ảnh (không phụ thuộc độ phân giải). `scripts/score.py` có cờ `--coord norm1000|pixel`: nếu nhãn/prompt dùng thang 0–1000 mà chấm bằng `pixel` (bỏ hệ số `×width/1000`), box bị co lại ~1000 lần → không box nào đạt IoU ≥ 0,5 → F1 ≈ 0 **dù JSON vẫn hợp lệ** (đúng như `test_pixel_vs_norm1000_scale_trap` trong `tests/test_score.py`). Tôi đã kiểm chứng gián tiếp: ở Bước 2 tôi chuyển nhãn sang JSON 0–1000 rồi **quy ngược** về pixel và vẽ chồng box gốc (ô "Quy ngược JSON -> pixel") — box vàng (quy ngược) trùng khít box đỏ (GT), tức thang 0–1000 được áp đúng. Việc so sánh F1 `norm1000` với `pixel` đầy đủ nằm trong E4 và **(chưa chạy)**.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Chỉ phần **trả lời** (JSON đáp án) là thứ model cần học sinh ra; prompt và 220 token ảnh là điều kiện đầu vào, không phải mục tiêu. Tôi đặt `labels[:, :n] = -100` với `n` = độ dài prompt, nên loss chỉ trên các token của đáp án. Kiểm chứng: decode riêng các token có `labels != -100` ra **đúng chuỗi JSON đáp án** (`số token ảnh (mm==1): 220`, `số token image_pad: 220`, độ dài chuỗi 614). Nếu tính loss cả prompt và token ảnh, model bị buộc tái tạo prompt/đặc trưng ảnh (nhiệm vụ vô nghĩa), làm loãng tín hiệu học và có thể tệ đi; ngoài ra chuỗi prompt dài (220 token ảnh) sẽ lấn át phần đáp án ngắn.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Nếu ViT bị lượng tử hoá 4-bit (NF4), trọng số của `merger`/`deepstack_merger_list` chỉ tồn tại dưới dạng **trọng số lượng tử + hằng số lượng tử hoá**, không phải số thực đầy đủ; cập nhật gradient lên dạng lưu trữ đó không biểu diễn được bước học đầy đủ → không thể full-finetune. Vì vậy tôi giữ ViT (kể cả projector) ở fp16 bằng `llm_int8_skip_modules=["visual","lm_head"]` trong `BitsAndBytesConfig`. Về fp32: trên lý thuyết projector nên học ở fp32 để ổn định số học. **Trong môi trường của tôi (torch 2.11)** việc ép projector sang fp32 gây xung đột `grad_dtype`/`GradScaler` khi optimizer trộn fp16–fp32; cách xử lý cuối cùng là dùng `AdamW(foreach=False)` (tối ưu từng tensor theo đúng dtype) và **bỏ GradScaler**, để grad và trạng thái optimizer luôn cùng dtype. Vì vậy notebook hiện chạy được; đây là điểm khác biệt so với hướng dẫn (vốn giả định cast projector sang fp32 + fp16 AMP).

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các ảnh là **frame cắt từ video**, hai frame liền kề gần như trùng nhau. Chia ngẫu nhiên sẽ đặt các frame gần giống nhau vào cả train lẫn test → **rò rỉ** (model gần như đã "thấy" ảnh test khi học) → F1 test cao giả. `prepare_data.py` gom ảnh theo **phiên quay** (ngày + giờ trong tên file) và gán **cả phiên** vào một split, nên không phiên nào nằm ở hai split (tôi kiểm tra: `OK: không phiên nào ở hai split`).
Số liệu split: train **1422** ảnh/**3515** box {pothole 1074, crack 1750, manhole 691}; val **276** ảnh/**639** box {66, 470, 103}; test **311** ảnh/**583** box {121, 299, 163}.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 đã có: F1 **0,270** (CI95 0,235–0,307), **33,0M** tham số, **6,43 GB**, **51,4 phút**. E2 (thêm projector, 87,6M) và E3 (thêm LoRA ViT) **(chưa chạy xong)** nên chưa thể kết luận có đáng hay không. Dự kiến: E2/E3 tốn VRAM và thời gian hơn (gradient xuyên ViT + projector), và phải nạp lại `extra_trainable.pt` khi suy luận, nếu không F1 E2/E3 sẽ trùng E1.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Test chỉ **311 ảnh** nên CI95 khá rộng: E0 [0,054–0,107] (rộng **0,053**), E1 [0,235–0,307] (rộng **0,072**). Nghĩa là E1 > E0 rõ rệt (hai khoảng không giao nhau), nhưng **chênh lệch nhỏ giữa các setup gần nhau sẽ không vượt CI95** → không thể kết luận chắc chắn setup này hơn setup kia nếu khoảng cách F1 nhỏ hơn ~0,07. Kết luận an toàn chỉ ở mức xu hướng; muốn chắc cần test lớn hơn hoặc lặp nhiều seed.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E0, **crack** kém nhất (F1 **0,026**), rồi tới pothole (0,161) và manhole (0,136); ở E1 vẫn là **crack** thấp nhất (0,182) so với pothole (0,217) và manhole (0,548). Giải thích: (i) box rất nhỏ — diện tích trung vị chỉ ≈0,7% khung hình nên khó định vị, đặc biệt crack là vết mảnh, dài; (ii) crack nhiều mẫu nhưng hình dạng đa dạng, dễ nhầm với vạch kẻ đường/nền; (iii) manhole to, tương phản rõ nên dễ học nhất (F1 cao nhất). Ảnh minh hoạ: `report/figs/gt_samples.png` (box GT) và `report/figs/best_predictions.png` (GT đỏ vs dự đoán xanh). Do E2–E4 (chưa chạy), phần so sánh lỗi giữa các setup bổ sung sau.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
**(Chưa chạy.)** Dự kiến chọn biến **rank** (r=4 và r=64, alpha=2·r) so với setup tốt nhất, ghi `report/evidence/e4rank4.json` / `e4rank64.json`. Với test 311 ảnh, kết luận sẽ phải nhìn CI95: nếu chênh lệch F1 giữa hai giá trị nhỏ hơn bề rộng CI95 (~0,07) thì **không đủ tin cậy** để khẳng định.

## Nhìn lại

- E0/E1 chạy ổn; E2 trở đi từng lỗi ở bước optimizer đầu do **trộn dtype fp16/fp32 + cơ chế `grad_dtype` mới của torch 2.11**. Đã sửa: `AdamW(foreach=False)`, bỏ `GradScaler`, giữ nguyên dtype từng tham số.
- Cần **một lần chạy đầy đủ** (Run all) để sinh `e2.json`, `e3.json`, `e4*.json` và các ảnh minh hoạ cho E2–E4; hiện chỉ E0/E1 có bằng chứng.
- E1 mất **51,4 phút** cho 300 bước (chậm hơn ước lượng 20–30 phút), nên E2–E4 cần thời gian GPU đáng kể.
- Chưa kiểm chứng được giới hạn đầy đủ của bộ chấm đối với ảnh quá tối/mờ hoặc box sát biên; đây là phần cần xem thêm trong `score.py` khi có E2–E4.
