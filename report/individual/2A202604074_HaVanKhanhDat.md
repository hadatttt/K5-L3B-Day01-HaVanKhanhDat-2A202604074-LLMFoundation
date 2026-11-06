# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** xem tên file
- **Họ và tên:** xem tên file
- **SEED:** 517

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.893 | 0.824 | 0.12 | 0 |
| 2. BoW + MLP | 0.956 | 0.919 | 0.38 | 0 |
| 3a. Encoder + kNN | 0.914 | 0.950 | 64.01 | 0 |
| 3b. Encoder + LogReg | 0.957 | 0.947 | 39.26 | 0 |
| 3c. Encoder + MLP head | 0.946 | 0.947 | 57.04 | 0 |
| 4a. LLM nhỏ zero-shot (Groq) | chỉ đo SUB | 0.878 | 450.07 | chưa có giá |
| 4b. LLM lớn zero-shot (Groq) | chỉ đo SUB | 0.872 | 411.55 | chưa có giá |
| 4c. LLM nhỏ few-shot (Groq) | chỉ đo SUB | 0.927 | 535.00 | chưa có giá |

Bảng giá trong notebook chưa có Groq nên cột USD của LLM không có nghĩa là miễn phí. SUB chỉ có 40 tin, nên chênh dưới khoảng 0.05 F1 được coi là hòa.

## 3. Dự đoán và thực tế

1. **`llm_models`.** Mình đoán model lớn hơn model nhỏ 0.02–0.05 F1 và LLM thắng 3b. Thực tế model lớn 0.872, model nhỏ 0.878, còn 3b đạt 0.947. Cả hai model LLM đều đúng 35/40 tin và chỉ khác loại lỗi, nên coi như hòa; model lớn không tốt hơn. Chi phí không so sánh được vì bảng giá trong notebook chưa có giá Groq nên hiện $0.
2. **`hard_eval`.** Mình đoán prompt injection là nhóm khó nhất với LLM. Thực tế cả 4a và 4d đều đạt 100% ở nhóm này, còn nhóm khó nhất là tin ham chứa từ khóa spam (50% ở 4a). LLM có vẻ cảnh giác quá mức với các từ như "FREE" trong tin ngắn thiếu ngữ cảnh.
3. **`tradeoff`.** Mình đoán 3c nằm trên đường Pareto với p95 khoảng 35 ms và 3b có thể bị lấn át. Thực tế đường Pareto gồm 1, 2, 3b và 3a, tức là mình đoán ngược ở 3b và 3c. 3c bị 3b lấn át vì hai cách có F1 SUB bằng nhau (0.947) mà 3b nhanh hơn (39 ms so với 57 ms). Dự đoán này được viết theo số của lần chạy trước, trong khi p95 thay đổi giữa các lần chạy.

## 4. Phân tích 50 case khó

BoW + MLP tụt nhiều nhất, từ F1 0.919 trên SUB xuống 0.368 trên 50 case khó. Nhóm yếu nhất của nó là spam giọng tự nhiên (0%), tiếp theo là né từ khóa (17%) và thao túng/lừa đảo (25%). Hai cột không so thẳng được vì khác tỉ lệ spam, nhưng mức tụt vẫn cho thấy điểm test thường chưa chứng tỏ model hiểu spam.

- **Case #13 (spam giọng tự nhiên):** "Hope you're well! Quick reminder that your photos from the event are ready, view & order prints…". BoW + MLP đoán sai vì chỉ nhìn token rời rạc, mà tin này không có từ khóa spam quen thuộc.
- **Case #2 (ham chứa từ khóa, sai ở 4d):** "FREE pizza in the office kitchen right now, come quick before it's gone". LLM thấy "FREE" và lời giục trong tin ngắn không có ngữ cảnh nên cảnh giác quá mức và đoán spam.

Tổng thể, F1 trên 50 case khó của BoW + LogReg là 0.333, 3a 0.741, 3b 0.625, 3c 0.653, 4a 0.906 và 4d 0.918.

## 5. Thử thách cá nhân

**Đề #2:** đổi vocab thành 100 và 5000 từ, train lại, báo F1 test và giải thích xu hướng.

**Cách làm:** viết hàm `_run_vocab_challenge`, tạo vocab bằng `build_vocab(texts_tr, max_size=...)`, vector hóa train và test, train lại model Phần 1 rồi đo kết quả.

| Vocab | F1 train | Precision test | Recall test | F1 test | Train (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.845 | 0.942 | 0.699 | 0.802 | 0.1 |
| 2000 | 0.915 | 1.000 | 0.806 | 0.893 | 3.6 |
| 5000 | 0.922 | 1.000 | 0.806 | 0.893 | 8.4 |

F1 test tăng rõ từ vocab 100 lên 2000, rồi đứng yên ở 5000. Với 100 từ, model thiếu nhiều từ đặc trưng của spam. Các từ thêm từ 2000 lên 5000 đều hiếm hơn, nên có thể không đủ sức đổi nhãn trên tập test, trong khi thời gian train tăng khoảng 2.3 lần. Vì vậy mình chọn vocab 2000.

## 6. Phương án cho sản phẩm thật

Mình chọn **3b. Encoder + LogReg**. Đề không cho ngưỡng chất lượng nên mình tự chọn F1 test ≥ 0.95. Ở lần chạy cuối, đạt p95 < 50 ms có BoW + LogReg (0.12 ms), BoW + MLP (0.38 ms) và 3b (39 ms). Trong đó BoW + MLP (0.956) và 3b (0.957) đạt ngưỡng F1, và 3b tốt hơn rõ trên 50 case khó (0.625 so với 0.368).

**Đánh đổi:** 3b chỉ đạt khoảng 50 tin/s trên Colab, thấp hơn mức trung bình 116 tin/s, nên cần ít nhất 3 máy như vậy hoặc GPU, và nhiều hơn cho giờ cao điểm. p95 của 3b cũng khá gần ngưỡng và dao động giữa các lần chạy, nên cần đo lại trước khi triển khai. LLM không đạt yêu cầu độ trễ (412–535 ms). Nếu dùng giá gpt-6-luna để ước tính, LLM tốn khoảng 3,150 USD/tháng.

**Hướng mở rộng:** 3b xử lý mọi tin trước; tin có p trong khoảng [0.3, 0.7] mới gửi LLM với prompt phòng thủ, và giữ nhãn của 3b nếu LLM lỗi. Với giả định 4% số tin cần LLM, chi phí khoảng 126 USD/tháng; tỉ lệ này cần đo lại trên tập val.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4: `call_openai` | Viết `call_openai` dùng `provider_kwargs` cho Groq | Viết hàm gọi Chat Completions, truyền `temperature`, `top_p` và `**provider_kwargs(max_tokens)`, trả về nội dung phản hồi và độ trễ | Đọc lại code, chạy ô kiểm tra 4.1 + 4.4 đạt, gọi thử Groq thành công |
| Phần 4: `llm_classify_batch` | Sửa lỗi 429 khi chạy batch | Gợi ý giãn cách giữa các lần gọi và tăng thời gian chờ khi thử lại | Thêm `time.sleep` 2.5 giây cho Groq, đặt `retry_with_backoff` với `max_retries=4`, `base_delay=15`; chạy lại trên 40 tin SUB với `invalid=0` |

**Điều đã học:** Mình nghĩ prompt injection sẽ là nhóm khó nhất với LLM, nhưng cả 4a và 4d đều đạt 100%. Nhóm khó nhất thật ra là tin ham chứa từ khóa spam (50% ở 4a). Mình cũng đoán model lớn tốt hơn model nhỏ, nhưng F1 gần như bằng nhau (0.872 và 0.878), trong khi thêm 4 ví dụ lại tăng lên 0.927. Mức tăng này chỉ là 2 tin nên cần thử trên nhiều tin hơn. Mình hiểu ra rằng cần kiểm tra bằng dữ liệu khó, và model lớn hơn chưa chắc tốt hơn.
