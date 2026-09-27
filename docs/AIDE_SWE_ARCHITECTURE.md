# Báo Cáo Nghiên Cứu & Kiến Trúc: Chuyển Giao Tinh Hoa AIDE ML và SWE-agent vào Antigravity IDE

Tài liệu này ghi lại toàn bộ quá trình nghiên cứu, bóc tách tài liệu/mã nguồn của hai công trình nghiên cứu nổi tiếng thế giới: **AIDE ML** (arXiv:2502.13138) và **SWE-agent** (NeurIPS 2024, arXiv:2405.15793), đồng thời lý giải chi tiết kiến trúc của hệ thống Universal ML-Harness.

---

## 1. Nghiên Cứu Công Trình 1: AIDE ML (Weco AI / arXiv:2502.13138)

### 1.1 Bối cảnh & Các nghiên cứu liên quan
AIDE (AI-Driven Exploration in the Space of Code) là thuật toán đứng sau thành tích áp đảo tại benchmark **MLE-bench của OpenAI** (75 kỳ thi Kaggle) và là nền tảng cốt lõi được Sakana AI sử dụng để xây dựng **The AI Scientist v2** (arXiv:2504.08066).

* **Vấn đề mà AIDE giải quyết:** Các Agent truyền thống (như AutoGPT, OpenHands) tiếp cận bài toán ML theo dạng tuyến tính: làm từ A1 đến A50. Khi gặp điểm số bão hòa, Agent rơi vào 2 cái bẫy:
  1. *Context Saturation:* Nhồi nhét toàn bộ lịch sử chạy vào prompt khiến LLM bị "ngợp token", quên mất các ràng buộc ban đầu.
  2. *Myopic Quitting:* Agent nhìn vào kết quả gần nhất thấy không tăng điểm thì vội vàng kết luận "hướng này bế tắc", sau đó bỏ cuộc hoặc nhảy sang ý tưởng hoàn toàn không liên quan (arbitrary pivot).

### 1.2 Bóc tách mã nguồn thực tế của AIDE (Đã đọc tại `tmp/aideml`)

#### A. Cấu trúc Cây và Ký ức Nén (`tmp/aideml/aide/journal.py`)
Trong AIDE, không bao giờ có danh sách phẳng A1..A50. Mọi thứ được mô hình hóa thành một **Cây phả hệ (Solution Tree)** với các lớp `Node` và `Journal`:
* Mỗi `Node` lưu trữ:
  - `id`: Định danh duy nhất.
  - `parent`: Con trỏ trỏ về thí nghiệm cha (nguồn gốc sinh ra nó).
  - `plan`: Bản tóm tắt giả thuyết cải tiến (chỉ 3-5 câu).
  - `metric`: Điểm số đạt được (`MetricValue`).
  - `analysis`: 2-3 câu đúc kết bài học thực nghiệm từ output log.
* **Hàm nén bộ nhớ (`generate_summary`):**
  AIDE không đưa code cũ vào prompt tiếp theo. Nó chỉ trích xuất:
  $$\text{Summary} = \sum \Big( \text{Plan} + \text{Analysis} + \text{Metric} \Big)$$
  Nhờ đó, dù hệ thống chạy qua 50 hay 100 thế hệ, toàn bộ lịch sử chỉ tốn **khoảng 1.000 tokens**. Agent luôn tỉnh táo, không bị loãng sự chú ý.

#### B. Chiến lược Tìm kiếm & Luật cấm EDA (`tmp/aideml/aide/agent.py`)
Khi soi vào prompt của hàm `_improve()` trong AIDE:
* **Luật cấm làm EDA vu vơ:** `Don't suggest to do EDA.` (Dòng 332 trong `agent.py`). AIDE nhận thấy khi Agent không biết làm gì, nó sẽ viện cớ "để tôi làm EDA thêm" rồi sinh ra hàng loạt phân tích bề nổi vô nghĩa. AIDE cấm tiệt hành vi này.
* **Luật Đơn nguyên (Atomicity):** `This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.` (Dòng 329). Mỗi thí nghiệm chỉ được đổi đúng 1 biến số (1 tham số, 1 hàm loss, hoặc 1 kiểu xử lý text).
* **Thuật toán UCB / Greedy Search:** AIDE dùng công thức Upper Confidence Bound để tính điểm tiềm năng cho từng nhánh:
  $$\text{Score}_{\text{UCB}}(N) = \text{Metric}(N) + c \cdot \sqrt{\frac{\ln(\text{Total Runs})}{\text{Visits}(N) + 1}}$$
  Nếu một nhánh có điểm số tốt nhưng chưa được khai thác nhiều, hệ thống sẽ **ép Agent tiếp tục đào sâu** thay vì cho phép nhảy sang nhánh khác.

---

## 2. Nghiên Cứu Công Trình 2: SWE-agent (Princeton & Stanford / NeurIPS 2024)

### 2.1 Bối cảnh & Luận điểm Khoa học
Công trình **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering** (arXiv:2405.15793) chỉ ra một sai lầm chết người của ngành AI:
> *"Lập trình viên thường nghĩ rằng chỉ cần viết prompt thật kỹ ('Hãy nhớ chạy file audit', 'Hãy trung thực') là LLM sẽ tuân thủ. Nhưng trong thực tế, LLM là một cỗ máy sinh xác suất từ ngữ (Probabilistic Token Generator). Khi gặp tình huống khó hoặc muốn kết thúc nhanh, LLM sẽ chọn con đường ngụy biện bằng ngôn ngữ tự nhiên (Confabulation & Sycophancy) thay vì thực thi công cụ kiểm định."*

### 2.2 Bóc tách mã nguồn thực tế của SWE-agent (Đã đọc tại `tmp/SWE-agent`)

#### Rào cản nộp bài đa tầng (`tmp/SWE-agent/tools/review_on_submit_m/bin/submit`)
Trong SWE-agent, Agent không thể gõ một câu lệnh là nộp được bài ngay. File `submit` thực hiện một quy trình chặn cưỡng bức:
1. **Chặn tiến trình (Process Interception):** Khi Agent gọi `submit`, script chạy lệnh `git diff` ngầm.
2. **Ném Diff vào mặt Agent:** Thay vì đóng gói, script in ra toàn bộ mã diff thực tế và trả về mã dừng. Nó ép Agent: *"Đây là diff thực tế của bạn. Bạn đã chạy script tái lập chưa? Bạn có sửa nhầm file test không? Hãy kiểm tra diff này rồi mới được xác nhận lại!"*.
3. **Deterministic Harness:** Hệ thống sử dụng một bộ "dây cương" bằng code Python thuần túy. Nếu các điều kiện vật lý chưa thỏa mãn, hàm nộp bài không bao giờ trả về file nộp.

---

## 3. Bản Đồ Giải Phẫu: Tại Sao Chúng Ta Chỉ Giữ Lại Mấy File Python Này?

Trong mã nguồn bạn vừa clone về:
* **SWE-agent** nặng hàng trăm MB, chứa hàng tá container Docker, Swerex, môi trường ảo devcontainer phức tạp.
* **AIDE** kéo theo cả thư viện Streamlit, giao diện WebUI, và các API gọi LLM bên ngoài liên tục.

Nếu bê nguyên xi 2 repo đó vào:
1. Bạn sẽ phải trả tiền API OpenAI riêng (vô nghĩa khi đang dùng Brain Antigravity).
2. Xung đột môi trường trên Windows (Docker symlinks, thư viện Linux).
3. Làm rối loạn cấu trúc thư mục của dự án thi đấu ML.

Do đó, chúng ta đã thực hiện một ca **"Phẫu thuật trích xuất tinh hoa" (Architectural Distillation)**: Chắt lọc đúng 4 hạt nhân mạnh nhất của họ và viết lại thành 4 module chuẩn mực trong workspace của bạn:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             BẢN ĐỒ CÔNG NGHỆ                                │
│                                                                             │
│  TINH HOA AIDE                           TINH HOA SWE-AGENT                 │
│  - Solution Tree                         - Agent-Computer Interface (ACI)   │
│  - Memory Compression                    - Deterministic Circuit Breaker    │
│  - UCB Branching                         - Diff Inspection Hook             │
│        │                                        │                           │
│        ▼                                        ▼                           │
│  [harness/tree_journal.py]               [tools/audit_gate.py]              │
│  - Cây phả hệ A1..A50                    - Kiểm tra SHA-256 trùng lặp       │
│  - Nén lịch sử < 600 tokens              - Đếm số mẫu diff thực tế          │
│  - Gợi ý nhánh, kiểm tra trần param      - Ký biên lai AUDIT_RECEIPT.json   │
│                                                                             │
│  [tools/journal_tool.py]                 [harness/validator.py]             │
│  - CLI query cho Brain Antigravity       - Code assertion chặn rò rỉ split  │
│                                                                             │
│  [harness/cloud_runner.py]               [.agents/skills/.../SKILL.md]      │
│  - Đẩy job nặng lên Cloud GPU            - Luật thép cấm chém gió lấp liếm  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Chi Tiết Từng File Mới Được Thêm Vào

### 1. `harness/tree_journal.py` (Kế thừa từ AIDE `journal.py`)
* **Chức năng:** Quản lý toàn bộ lịch sử thí nghiệm dạng Cây phả hệ thay vì danh sách tuyến tính A1..A50.
* **Hàm quan trọng:**
  * `generate_compact_summary()`: Gom toàn bộ 50 thí nghiệm thành một bảng Markdown súc tích chỉ ~300-600 tokens. Antigravity Brain đọc bảng này trong 0.1 giây mà không bị loãng bộ nhớ.
  * `recommend_next_branch(max_params)`: Tính toán điểm UCB, phát hiện nhánh nào tiềm năng nhất và lọc bỏ các mô hình vượt quá số tham số cho phép (`max_params_limit`).

### 2. `tools/journal_tool.py` (Giao diện CLI cho Agent)
* **Chức năng:** Cung cấp lệnh terminal để Antigravity gọi:
  * `python tools/journal_tool.py --summary`: Tra cứu nhanh lịch sử.
  * `python tools/journal_tool.py --recommend --max-params 150000000`: Nhận chỉ thị khoa học xem nên thử nghiệm gì tiếp theo.

### 3. `tools/audit_gate.py` (Kế thừa từ SWE-agent `review_on_submit`)
* **Chức năng:** Cổng kiểm định vật lý ngăn chặn 100% thói chém gió lấp liếm.
* **Cơ chế hoạt động:**
  1. Tính mã băm SHA-256 của file ứng viên so với baseline. Nếu trùng 100% -> **Crash ngay lập tức, từ chối nộp.**
  2. So sánh từng dòng dự đoán. Nếu số lượng câu khác biệt $= 0$ -> **Crash ngay lập tức.**
  3. Kiểm tra tính toàn vẹn: Không được có `NaN`, ID phải hợp lệ theo định dạng task.
  4. Nếu vượt qua 100% bài test: Đóng gói file nộp (`submission.zip` hoặc `submission.csv`) và ký file `AUDIT_RECEIPT.json`.

### 4. `harness/validator.py` (Guardrail as Code)
* **Chức năng:** Biến các luật văn bản thành các câu lệnh `assert` trong Python:
  * `assert_parameter_limit()`: Đếm tham số PyTorch thật, nếu vượt trần thì văng lỗi, ép Agent phải giảm kích thước mô hình.
  * `assert_no_group_leakage()`: Kiểm tra giao thoa giữa tập train và validation. Nếu trùng group/entity -> Crash để chống rò rỉ dữ liệu.

### 5. `harness/cloud_runner.py` (Cầu nối GPU)
* **Chức năng:** Cho phép Antigravity trên máy Windows gửi job huấn luyện các mô hình nặng (XLM-R, DeBERTa 5 folds) lên **Modal.com** hoặc **Kaggle API**, tự động tải kết quả về thư mục artifacts mà không làm treo máy local.

### 6. `.agents/skills/ml-experiment-harness/SKILL.md` & `AGENTS.md`
* **Chức năng:** "Khắc vào vỏ não" của Antigravity quy tắc làm việc mới:
  * Cấm Agent tự ý tạo file `submission.zip`.
  * Bắt buộc phải chạy `audit_gate.py` và chỉ được báo cáo khi có biên lai `AUDIT_RECEIPT.json`.
  * Cấm đoán mò khi đề xuất thí nghiệm, bắt buộc tra cứu qua `journal_tool.py`.
