# Robust Reward Design Lab for MDPs in Cybersecurity

Một bộ **môi trường thực nghiệm + hệ thống ứng dụng nhỏ** cho đề tài:

**Robust Reward Design for MDPs in Cybersecurity: A Probabilistic Attack-Graph Case Study**

Bộ này giúp bạn làm được 4 việc trong một repo:

1. **Mô hình hóa attack graph thành MDP**
2. **Giải standard reward design theo mô hình Stackelberg**
3. **Giải robust reward design theo optimal interior-point / max-margin**
4. **Chạy thực nghiệm, quét budget, so sánh standard vs robust, và chạy app demo**

---

## 1. Bộ này có gì

### Mã nguồn chính
- `src/mdp_model.py`: lớp dữ liệu cho attack-graph MDP
- `src/standard_reward_design.py`: standard MILP cho reward design
- `src/robust_reward_design.py`: robust MILP max-margin
- `src/evaluation.py`: optimistic / pessimistic values, reward-perturbation, bounded rationality
- `src/graph_generator.py`: sinh attack graph dạng layered
- `src/visualization.py`: vẽ graph và plot thực nghiệm
- `src/run_case.py`: chạy một case cụ thể
- `src/run_experiments.py`: chạy toàn bộ suite thực nghiệm
- `app.py`: app Streamlit để demo như một hệ thống ứng dụng nhỏ

### Môi trường / kịch bản tấn công có sẵn
- `configs/paper_style_attack_graph.json`
- `configs/branching_enterprise_graph.json`
- `configs/lateral_movement_graph.json`

### Script chạy nhanh
- `scripts/setup_env.sh`
- `scripts/run_demo.sh`
- `scripts/run_all_experiments.sh`

---

## 2. Ý tưởng mô hình

### Attacker = follower
Attacker tối đa hóa reward của chính mình trên attack graph MDP.

### Defender = leader
Defender phân bổ reward giả tại các decoy / honeypot sites, với:
- miền can thiệp `D` = tập các state-action pairs được phép chỉnh reward
- ngân sách `C`

### Hai bài toán đã được dựng sẵn

#### Standard reward design
Giải:
- chọn `x` để tối đa hóa payoff của defender
- attacker phản ứng tối ưu theo reward đã bị chỉnh sửa

#### Robust reward design
Giải:
- giữ nguyên mức tối ưu `v1*`
- đồng thời tối đa hóa margin `c*`
- để nghiệm robust hơn trước:
  - nonunique best responses
  - perturbation trong attacker reward perception
  - bounded rationality

---

## 3. Cài đặt môi trường

### Linux / macOS
```bash
cd robust_reward_design_lab
bash scripts/setup_env.sh
source .venv/bin/activate
```

### Windows PowerShell
```powershell
cd robust_reward_design_lab
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

---

## 4. Chạy nhanh một demo

```bash
cd robust_reward_design_lab
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
bash scripts/run_demo.sh
```

Kết quả sẽ nằm trong:

```bash
results/paper_demo/
```

Các file tạo ra:
- `summary.json`
- `attack_graph.png`
- `tau_sweep.png`

---

## 5. Chạy toàn bộ thực nghiệm

```bash
cd robust_reward_design_lab
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
bash scripts/run_all_experiments.sh
```

Kết quả chính:
- `results/experiment_summary.csv`
- `results/<case_name>/summary.json`
- `results/budget_sweep/budget_sweep.json`
- `results/budget_sweep/budget_sweep.png`

---

## 6. Chạy app hệ thống ứng dụng

```bash
cd robust_reward_design_lab
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
streamlit run app.py
```

App cho phép:
- chọn scenario
- đổi budget `C`
- chạy standard và robust
- xem graph
- xem allocation
- xem bảng bounded-rational sweep

---

## 7. Các case đã dựng

### 7.1 paper_style_attack_graph
Case gần tinh thần bài JAIR:
- 1 true goal
- 2 decoy sites
- budget mặc định `C = 2.4`
- phù hợp nhất để chạy demo robust vs non-robust

### 7.2 branching_enterprise_graph
Kịch bản enterprise branching:
- nhiều nhánh do thám và lateral movement
- robust margin có thể nhỏ hoặc bằng 0
- hữu ích để minh họa trường hợp bài toán tối ưu nhưng khó có interior point rõ ràng

### 7.3 lateral_movement_graph
Kịch bản lateral movement:
- 1 domain-controller goal
- 2 decoy servers
- thường cho thấy robust allocation cải thiện pessimistic value rõ hơn

---

## 8. Những metric đã có

Trong code hiện có sẵn các metric sau:
- `v1*`
- optimistic defender value
- pessimistic defender value
- robust margin `c*`
- true-goal reach probability
- decoy-capture probability
- bounded-rational defender value theo `tau`
- reward-perception sensitivity sweep theo `epsilon`

---

## 9. Điều quan trọng về solver

Repo này dùng:
- **PuLP + CBC**
- không cần Gurobi
- phù hợp cho **small / medium tabular cases**

Điều đó có nghĩa là:
- rất hợp để làm **NCKH sinh viên / thesis prototype / thực nghiệm paper nhỏ**
- chưa nhắm tới production-scale enterprise graph cực lớn

---

## 10. Bạn nên dùng bộ này thế nào để viết bài

### Giai đoạn 1: Reproduce
Chạy 3 case có sẵn, lấy:
- `x_MILP`
- `x_IP`
- `v1*`
- `c*`
- gap giữa optimistic và pessimistic

### Giai đoạn 2: Add novelty
Mở rộng theo một trong hai hướng:
- **thiết kế miền can thiệp `D`**
- **quét budget `C` và tìm phase transition của robust solution**

### Giai đoạn 3: Nâng thành paper
Đưa vào bài:
- notation table
- assumption table attacker/defender
- 1 attack graph chuẩn
- 1 bảng transition probabilities
- 1 bảng `r1`, `r2`, `D`, `C`
- 1 bảng so sánh standard vs robust
- 1 hình budget sweep
- 1 hình bounded-rational sweep

---

## 11. Lưu ý khoa học

Bộ này được làm theo hướng **thực nghiệm chạy được**, không phải một bản sao hoàn toàn của code tác giả JAIR.

Nó phù hợp để:
- làm môi trường nghiên cứu
- dựng prototype
- mở rộng thêm case study cyber của riêng bạn
- tạo kết quả ban đầu cho báo cáo / paper

Nếu bạn muốn nâng tiếp, hướng tốt nhất là:
- thay các graph toy bằng graph sinh từ dữ liệu mạng thật hoặc generator thực tế hơn
- mở rộng `D` sang nhiều state-action pairs hơn
- thêm experiment sweep theo topology
- thêm comparison với heuristic allocation baselines

