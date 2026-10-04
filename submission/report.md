1. Tôi làm lab trên GCP; báo cáo cũ ghi máy e2-medium; region/zone mặc định của source là us-central1/us-central1-a, chưa xác nhận cấu hình VM thực tế. Source nền: 55539f67d7c78b43afe334a2ec3271c4bfdbbe2d, kèm các sửa đổi chưa commit trong script startup GCP.
2. Dataset có 284.807 dòng, 492 giao dịch gian lận; chia stratified train/validation/test khoảng 70/10/20% (199.364/28.481/56.962 dòng), seed 42, dùng 2 threads.
3. Load dữ liệu mất 2,320 giây; training mất 2,188 giây; best iteration là 1, tạo mô hình nhỏ và góp phần làm inference nhanh.
4. Trên tập test: AUC 0,942199; Accuracy 0,983463; F1 0,158929; Precision 0,087084; Recall 0,908163. Tại threshold 0,5, recall cao nhưng precision/F1 thấp, cho thấy nhiều cảnh báo sai.
5. Latency 1 dòng trung vị 0,827 ms qua 200 lần; throughput batch 1.000 dòng khoảng 931.690 dòng/giây, tính bằng 1.000 chia thời gian trung bình của 10 lần predict_proba sau warm-up, dùng perf_counter.
6. Ảnh Screenshot_resource.png cho thấy CPU idle 77,8%, RAM dùng 511 MiB trên khoảng 3,8 GiB; ens4 RX 264.461.737 byte và TX 2.151.205 byte tích lũy. Chưa xác nhận ảnh chụp trong hay sau benchmark và ngày giờ chụp.
7. Ảnh Screenshot_gcp_billing_reports.png hiển thị ₫0 cho 01–03/10/2026 trong phạm vi tháng; chưa thể hiện rõ bộ lọc project và chi phí từng dịch vụ trong ngày lab. Chưa xác nhận thời điểm kiểm tra, chưa đủ bằng chứng để kết luận lab miễn phí.
8. Code và kết quả đã có trên laptop; chưa xác nhận thời điểm tải về, chưa có bằng chứng terraform destroy hoàn tất hoặc thời điểm dọn dẹp. Cần bổ sung bằng chứng chi phí và dọn dẹp để hoàn thiện bài nộp.
