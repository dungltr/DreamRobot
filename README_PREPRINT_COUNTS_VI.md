# DREAM preprint: quy ước đếm và nguồn gốc kết quả đã lưu

Ghi chú tài liệu cho snapshot `84a2798cd46565b3dd72f9210511a7f5f7f1cf1e`. Áp dụng sửa tài liệu trên nhánh hiện tại; không sửa code đã tạo số liệu, không di chuyển tag và không thay giá trị CSV gốc.

## Hai quy ước đếm

| World | Raw baseline / DREAM | Giảm raw count | Complete baseline / DREAM | Giảm complete count |
|---|---|---|---|---|
| A | 9 / 2 | 77.8% | 11 / 3 | 72.7% |
| B | 26 / 8 | 69.2% | 28 / 9 | 67.9% |
| C | 33 / 3 | 90.9% | 35 / 4 | 88.6% |

Headline của preprint là complete boundary-inclusive waypoint counts, cùng chính sách tính bảy trường. Raw counts ở đây dùng bản two-rate reproduction đã regularize trên selected clean recording. Cùng số trường không có nghĩa cùng semantic payload hoặc đã đo ROS 2 network traffic.

Các số 89.5/69.2/90.9% ở README cũ là legacy diagnostics, không phải headline hiện tại. Riêng World A còn khác quy ước yaw-test regularization, không chỉ thêm boundary. Cần sửa nhãn/diễn giải ở mục 2.3.1 và checklist cuối README/README_VN; không chỉ nối thêm note rồi để nguyên hai headline mâu thuẫn. Nếu giữ ví dụ cũ, gắn rõ baseline/cách đếm của nó.

`--compare` in raw ratio `1 - len(d_idx)/len(b_idx)`, không tự trở thành complete-count reduction. Không gán headline mới cho output đó nếu wrapper chưa dựng đúng complete sets. Trong nominal noise, mean complete counts World A là 20.0/3.0; 85% là tỷ số của mean hiển thị, không mặc định là mean của tỷ lệ từng run.

## Endpoint và nhóm đánh giá

PE ở 0.030 m trong estimate frame là đạt stopping tolerance, không phải độ chính xác vị trí thật. Two-state study tách PE_true và operational criterion 0.06 m. Corridor driver stop 0.06 m, offline clearance và physics bumper contact không được gộp thành một phép đo hoặc một mean chung.

## Clipping forward

Protocol forward gọi `reconstruct_noisy` với `vn=max(0.0, v+sigma_v*eta)` rồi tích phân. Gaussian mô tả nhiễu đầu vào trước clipping, không phải phân phối vận tốc sau clipping. Đây là lựa chọn preprocessing của benchmark; sampler tổng quát vẫn tích phân velocity đầu vào có dấu. Không suy ra clipping return chỉ từ hàm forward; cần đối chiếu riêng engine two-state.

## CSV lưu trữ khác summary CLI

CSV tham chiếu: `bench_data/two_state_runs.csv` tại commit `84a2798cd46565b3dd72f9210511a7f5f7f1cf1e`.

https://github.com/dungltr/DreamRobot/blob/84a2798cd46565b3dd72f9210511a7f5f7f1cf1e/bench_data/two_state_runs.csv

Command được ghi nhận để in aggregate results:

```sh
python3 run_gazebo_benchmark.py --noise-robustness --runs 50
```

Vòng sửa này không chạy command. Không nói command trực tiếp export CSV hoặc tái tạo byte-identical nếu chưa có exporter/config/log tương ứng. Nếu exporter chưa được lưu, ghi rõ provenance còn thiếu, không tự đặt tên script cho khớp.

Các trường CSV được tài liệu ghi nhận gồm world/sampler/seed/return_seed, N_sampler/N_stored, PE_true/PE_odom, completed/timed_out và forward_drift. Protocol có 50 realizations cho mỗi world/sampler; kiểm tra file đã lưu trước khi khẳng định số dòng hoặc tính lại thống kê. Utility preflight kèm theo chỉ đọc snapshot/CSV hiện có, không import/chạy benchmark.

Dùng full commit làm artifact identifier; kiểm tra tag dereference thay vì chỉ đọc release description. Sửa README bằng commit mới trên nhánh hiện tại, không move tag benchmark. Chủ sở hữu phải xác định license code/data thích hợp và quyền của thành phần bên thứ ba; repo public không tự cấp license sử dụng lại.
