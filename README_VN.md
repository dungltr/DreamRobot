# README — Hướng dẫn tự kiểm tra hệ thống DREAM (Robot tự phục hồi khi mất camera)

Tài liệu này dành cho người **không rành lập trình**. Bạn chỉ cần **sao chép – dán** các
lệnh dưới đây vào cửa sổ Terminal và so sánh kết quả hiện ra với phần
**“Kết quả mong đợi”**. Không cần sửa code.

---

## 0. Đây là dự án gì?

Hệ thống mô phỏng một con robot chạy bằng vSLAM. Khi **mất tín hiệu camera**, robot
phải **tự quay về điểm xuất phát** mà không nhìn thấy gì, chỉ dựa vào các **“mốc đường”**
(landmark) đã ghi trước đó.

DREAM là thuật toán **ghi mốc thông minh**: chỉ ghi mốc ở những chỗ cần thiết (khúc cua,
chỗ đổi tốc độ) và **bỏ qua** những đoạn thẳng đều, nhờ đó **giảm số mốc phải lưu**
mà vẫn quay về chính xác.

Mục tiêu của bản README này: giúp bạn **chạy lại các phép thử** và **tự kiểm tra** xem
những con số trong bản thảo (bài báo) có đúng với kết quả thực tế hay không.

---

## 1. Chuẩn bị một lần duy nhất

### 1.1 Yêu cầu đã cài sẵn (kiểm tra nhanh)

Mở Terminal và gõ lần lượt, nếu mỗi lệnh in ra một phiên bản thì đã sẵn sàng:

```bash
python3 --version        # mong đợi: Python 3.10.x
ros2 --help | head -1    # mong đợi: có chữ "usage: ros2"
ign gazebo --version     # mong đợi: "Gazebo Sim, version 6.18.x"
Xvfb -help 2>&1 | head -1  # mong đợi: có chữ "use: X"
```

Nếu lệnh nào báo `command not found` thì phần đó chưa cài (xem mục 7 để biết cách cài).

### 1.2 Mở đúng thư mục dự án

Mỗi lần mở Terminal mới, việc đầu tiên là đi vào thư mục dự án:

```bash
cd ~/codewhale_recoveryRobotDREAM
```

> Nếu dự án nằm ở chỗ khác, thay đường dẫn cho đúng. Bạn biết mình đã đúng chỗ khi
> lệnh `ls` (liệt kê file) hiện ra các tên như `run_gazebo_demo.sh`,
> `test_auto_recovery_sim.py`, `publish/`, `paper/`...

### 1.3 “Kích hoạt” môi trường ROS 2 (hãy gõ đầu mỗi phiên Terminal)

ROS giống như một “bộ công cụ” cần được bật lên trước khi dùng. Gõ 2 dòng này:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

> Sau khi gõ, màn hình **không in ra gì** là bình thường — tức là đã bật thành công.
> Hai dòng này chỉ có hiệu lực trong **cửa sổ Terminal hiện tại**. Nếu mở Terminal mới,
> phải gõ lại.
>
> **Mẹo dành cho người mới:** hãy coi 2 dòng này là thao tác “mở khóa” cần làm đầu
> mỗi phiên làm việc. Có vài lệnh (mục 2.2, 2.3) không bắt buộc cần, nhưng gõ thêm cũng
> **vô hại** — nên cứ gõ cho chắc, đỡ phải nhớ lệnh nào cần lệnh nào không.

### 1.4 Biên dịch lại toàn bộ (chỉ cần làm một lần, hoặc sau khi sửa code)

```bash
colcon build --packages-select recovery_interfaces vslam_recovery_core --symlink-install
```

Dấu hiệu thành công: dòng cuối cùng có dạng

```
Summary: 2 packages finished [...]
```

---

## 2. Các phép kiểm tra (chạy từ nhanh → chậm)

> Quy tắc chung: **chạy theo đúng thứ tự**. Các phép ở mục 2.1 → 2.3 chạy trong vài
> giây đến vài phút và **không cần mở màn hình ảo**. Phép 2.4 và 2.5 cần nhiều thời gian
> và/hoặc màn hình ảo.

---

### 2.1 Kiểm tra NHANH nhất: 4 chế độ Auto + mất camera ở 100% (không cần Gazebo)

Đây là phép kiểm tra “tính đúng đắn” cốt lõi: nó chạy **đúng luật điều khiển trong code**
trên một mô hình robot đơn giản, mất camera ở 100% quãng đường, rồi đo robot có quay về
được không.

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 test_auto_recovery_sim.py
```

**Kết quả mong đợi** (4 dòng `PASS` và dòng cuối `ALL PASS`):

```
Mode 1: PASS  n_lm=  9  final_err=0.030 m  wall_margin=0.267 m  obst_clear=0.293 m  coll=None
Mode 2: PASS  n_lm=  9  final_err=0.030 m  wall_margin=0.273 m  obst_clear=0.299 m  coll=None
Mode 3: PASS  n_lm=  4  final_err=0.030 m  wall_margin=0.308 m  obst_clear=0.204 m  coll=None
Mode 4: PASS  n_lm=  4  final_err=0.030 m  wall_margin=0.309 m  obst_clear=0.215 m  coll=None
RESULT: ALL PASS
```

Cách đọc:

- `n_lm` = số mốc đường DREAM đã ghi (càng ít càng tiết kiệm bộ nhớ).
- `final_err` = sai số khi quay về điểm xuất phát (đơn vị mét). `0.030` ≈ 3 cm.
- `coll=None` = **không va chạm** (nếu va chạm sẽ in `wall` hoặc `obstacle`).
- `wall_margin` / `obst_clear` = khoảng cách an toàn tối thiểu tới tường / vật cản (m).

**Đây là kết quả bạn đối chiếu với bảng “End-to-end corridor demonstration” trong bài
báo.** (Lưu ý: bảng trong bài báo dùng số liệu Gazebo thật là `0.060 m`; file này dùng mô
hình offline nên là `0.030 m` — cả hai đều là quay về chính xác, chỉ khác nguồn dữ liệu.)

---

### 2.2 Kiểm tra LÝ THUYẾT: định luật tỉ lệ (không cần robot)

Phép này xác nhận công thức toán học trong bài báo (`L ≈ 1 − c(κW)²`, `c ≈ 0.0168`).

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 dream_theory_check.py
```

**Kết quả mong đợi:** dòng gần cuối in ra `c ~= 0.01681`, và các giá trị cột
`(1-linear)/(kW)^2` đều nằm quanh `0.0168` (0.01679 → 0.01683).

> Con số `c ≈ 0.0168` này phải khớp với bài báo (trước đây từng viết nhầm `0.0167`,
> đã sửa lại — nếu bạn thấy tài liệu nào còn ghi `0.0167` thì đó là bản cũ).

---

### 2.3 Kiểm tra SỐ LIỆU benchmark trên dữ liệu Gazebo ĐÃ GHI

Thư mục `bench_data/` chứa **dữ liệu odometry thật** đã ghi từ Gazebo cho 3 thế giới:
`odom_A.csv` (hành lang thẳng), `odom_B.csv` (slalom), `odom_C.csv` (chữ S).
Các lệnh dưới đây **tính lại các bảng trong bài báo** từ chính dữ liệu đó.

> Muốn **tự ghi lại** odometry (thay vì chỉ tính lại từ các CSV đã lưu), các thế giới có
> sensor camera cần màn hình ảo:
> `xvfb-run -a python3 run_gazebo_benchmark.py`. Các lệnh offline bên dưới
> (`--compare`, `--mc`, `--ablate`) không cần màn hình.

#### 2.3.1 Bảng DR / sai số quay về (bảng chính)

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 run_gazebo_benchmark.py --compare
```

**Kết quả mong đợi** (chú ý cột cuối `PE_*_cl` đều bằng `0.030`):

```
[A] ... N_baseline=18 N_dream=2  DR= 88.9%  ... PE_base_cl=0.030 m  PE_dream_cl=0.030 m
[B] ... N_baseline=10 N_dream=6  DR= 40.0%  ... PE_base_cl=0.030 m  PE_dream_cl=0.030 m
[C] ... N_baseline=12 N_dream=7  DR= 41.7%  ... PE_base_cl=0.030 m  PE_dream_cl=0.030 m
```

Ý nghĩa: DREAM giảm số mốc 88.9% / 40.0% / 41.7% mà **sai số quay về vẫn 0.030 m**
(giống baseline). Đây chính là bảng “Landmark count, data reduction, and closed-loop
recovery error” trong bài.

#### 2.3.2 Bảng quét ngưỡng R²_th

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 sweep_r2_threshold.py
```

**Kết quả mong đợi:** với mọi ngưỡng tái tạo được (≥ 2 mốc), cột `PE_dream_cl` đều
`0.030 m`; chỉ những trường hợp chỉ còn `1` mốc mới in `inf`.

#### 2.3.3 Bảng Monte-Carlo (nhiễu) và ablation

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 run_gazebo_benchmark.py --mc --runs 20 --noise 0.02 --ablate
```

**Kết quả mong đợi:** các dòng `Monte-Carlo` in `PE_dream_cl=0.030±0.000 m`; các dòng
ablation in `PE_cl` = `0.030` (trừ trường hợp `regress_speed=False` ở thế giới A in `inf`).

#### 2.3.4 Bảng so sánh với 4 phương pháp khác (trên dữ liệu Gazebo)

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 benchmark_baselines_gazebo.py --mc
```

**Kết quả mong đợi:** in 3 khối (World A/B/C), mỗi khối 5 dòng. Dòng cuối mỗi khối là
`dream (R2_th=0.98)`. Đối chiếu với bảng “Offline baseline comparison” trong bài:
- World A: DREAM `N≈3.0`, `path_max≈0.006` (nhỏ hơn Nyquist `0.206` rất nhiều).
- World B: DREAM `N≈13.9`, `path_max≈0.069` (nhỏ hơn Nyquist `0.142`).
- World C: DREAM `N≈8.5`, `path_max≈0.130` (cỡ Nyquist `0.116`, nhưng ít mốc hơn).

---

### 2.4 Kiểm tra ĐẦY ĐỦ trên Gazebo thật (headless, chậm ~15 phút)

Phép này chạy **toàn bộ hệ thống ROS 2 thật** (Gazebo không giao diện + các node ROS)
để lái robot và quay về. Chạy cả 4 chế độ Auto:

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 run_headless_auto_test.py
```

**Kết quả mong đợi** (4 dòng `PASS`):

```
Mode 1: PASS  final_err=0.060 m  wall=0.267 m  obst=0.293 m  coll=None
Mode 2: PASS  final_err=0.060 m  wall=0.273 m  obst=0.299 m  coll=None
Mode 3: PASS  final_err=0.060 m  wall=0.309 m  obst=0.205 m  coll=None
Mode 4: PASS  final_err=0.060 m  wall=0.310 m  obst=0.214 m  coll=None
```

> Muốn chạy nhanh hơn (chỉ 1 chế độ), thêm `--mode 4` vào cuối lệnh.

---

### 2.5 Kiểm tra GIAO DIỆN (Gazebo GUI + RViz + Dashboard, tùy chọn)

Bước này mở **cửa sổ đồ họa** để bạn thấy tận mắt. Vì máy chạy không có màn hình vật lý
nên cần **màn hình ảo Xvfb** trước.

Mở một Terminal và gõ:

```bash
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp &
export DISPLAY=:99
```

Rồi trong cùng Terminal đó:

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
./run_gazebo_demo.sh
```

Dấu hiệu thành công: log hiện các dòng `Dashboard GUI started.`, `rviz2`, và `ign gazebo`
đang chạy; các topic `/odom_raw`, `/odom`, `/cmd_vel` xuất hiện.

Muốn **nhìn từ máy khác** (VNC), mở thêm một Terminal:

```bash
x11vnc -display :99 -forever -nopw -rfbport 5900
```

rồi dùng VNC client kết nối tới `địa_chỉ_máy:5900`.

---

## 3. Đối chiếu nhanh: “bài báo nói gì ↔ chạy lệnh nào”

| Nội dung trong bài báo | Lệnh kiểm tra | Con số cần khớp |
|---|---|---|
| Bảng DR / sai số quay về (3 thế giới) | `python3 run_gazebo_benchmark.py --compare` | DR `88.9/40.0/41.7%`, PE `0.030 m` |
| Bảng quét ngưỡng R²_th | `python3 sweep_r2_threshold.py` | PE `0.030 m` (trừ `inf`) |
| Bảng lý thuyết (c ≈ 0.0168) | `python3 dream_theory_check.py` | `c ~= 0.0168` |
| Bảng Monte-Carlo | `run_gazebo_benchmark.py --mc ...` | PE `0.030±0.000 m` |
| Bảng ablation | `run_gazebo_benchmark.py ... --ablate` | PE `0.030` (trừ `inf`) |
| Bảng so sánh 5 phương pháp | `python3 benchmark_baselines_gazebo.py --mc` | N / path_max như mục 2.3.4 |
| Bảng 4 chế độ Auto + 100% failure | `python3 test_auto_recovery_sim.py` (offline) hoặc `run_headless_auto_test.py` (Gazebo thật) | lỗi `0.030 m` (offline) / `0.060 m` (Gazebo), `coll=None` |

---

## 4. Cách tự đọc kết quả “đúng” hay “sai”

Một phép kiểm tra được coi là **ĐẠT** khi:

1. Robot **quay về** điểm xuất phát: `final_err` hoặc `PE_*_cl` nhỏ (≤ 0.06 m).
2. **Không va chạm**: `coll=None` (hoặc không xuất hiện chữ `wall`/`obstacle`).
3. **Số mốc DREAM nhỏ hơn baseline**: `N_dream < N_baseline` (đó là mục đích giảm dữ liệu).

Nếu bạn thấy:
- `FAIL` hoặc `coll=wall` / `coll=obstacle` → hệ thống có lỗi cần sửa.
- `PE = inf` → đường đi bị nén quá mức (chỉ còn 1 mốc), không tái tạo được.
- `ImportError ... librcl_action.so` → bạn **quên gõ 2 dòng `source ...`** ở mục 1.3.

---

## 5. Các file quan trọng (để bạn biết cái gì nằm ở đâu)

- `src/vslam_recovery_core/src/dream_core.py` — thuật toán DREAM (lõi).
- `src/vslam_recovery_core/src/dream_sampler_node.py` — node ROS 2 ghi mốc DREAM.
- `src/vslam_recovery_core/src/recovery_controller.cpp` — bộ điều khiển **quay về vòng kín**.
- `src/vslam_recovery_core/src/recovery_recorder_server.cpp` — node ghi mốc vào YAML.
- `src/vslam_recovery_core/src/recovery_dashboard_gui.py` — giao diện điều khiển (4 chế độ Auto).
- `test_auto_recovery_sim.py` — kiểm tra nhanh (offline).
- `run_headless_auto_test.py` — kiểm tra đầy đủ Gazebo headless.
- `run_gazebo_benchmark.py`, `sweep_r2_threshold.py`, `benchmark_baselines_gazebo.py` — sinh các bảng benchmark.
- `benchmark_footprint.py` — đo thời gian CPU và bộ nhớ mỗi lần cập nhật DREAM.
- `bench_data/odom_A/B/C.csv` — dữ liệu odometry Gazebo đã ghi.
- `publish/DREAM-ROS2-IEEE.tex` — bản thảo bài báo IEEE Access.
- `publish/DREAM-ROS2-IEEE.pdf` — bản PDF đã biên dịch.
- `paper/benchmark_results.md` — ghi chú chi tiết kết quả + lệnh tái lập.

---

## 6. Câu lệnh “tất cả trong một” (nếu bạn muốn nhanh)

Sao chép toàn bộ khối này và dán vào Terminal (sau khi đã làm mục 1.3):

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
echo "== 1) Offline 4 chế độ ==" && python3 test_auto_recovery_sim.py
echo "== 2) Lý thuyết ==" && python3 dream_theory_check.py
echo "== 3) Benchmark DR/PE ==" && python3 run_gazebo_benchmark.py --compare
echo "== 4) Quét ngưỡng ==" && python3 sweep_r2_threshold.py
echo "== 5) So sánh 5 phương pháp ==" && python3 benchmark_baselines_gazebo.py --mc
```

---

## 7. Cài đặt khi thiếu thành phần (chỉ khi mục 1.1 báo thiếu)

```bash
# ROS 2 Humble + Gazebo Fortress (làm theo trang chính thức của ROS 2 cho Ubuntu 22.04)
# Màn hình ảo Xvfb (để mở giao diện không cần màn hình thật):
sudo apt-get update
sudo apt-get install -y xvfb x11-utils
# Xem giao diện từ xa qua VNC (tùy chọn):
sudo apt-get install -y x11vnc
```

> Lưu ý: lệnh `sudo apt-get ...` cần mật khẩu quản trị (root). Nếu bạn đang ở trong một
> môi trường sandbox bị chặn `sudo` thì phải chạy các lệnh cài đặt này trên máy/container
> của chính bạn.

---

## 8. Nếu kết quả không khớp

Ghi lại **toàn bộ nội dung Terminal** (chụp màn hình hoặc copy) và báo lại, kèm theo:
1. Bạn đang ở bước số mấy.
2. Dòng lệnh bạn đã chạy.
3. 5–10 dòng cuối cùng hiện ra trên màn hình.

Như vậy người hỗ trợ sẽ biết chính xác chỗ sai mà không cần đoán.
