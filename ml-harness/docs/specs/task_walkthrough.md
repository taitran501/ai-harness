# StereoQueerEval: đọc bài toán qua dữ liệu hiện có

## 1. Một mẫu đầu vào là gì?

Mỗi dòng là một bình luận YouTube kèm ngữ cảnh của video:

```text
yt_title + yt_description + yt_comment
```

Mục tiêu không phải phân loại video. Mục tiêu là phân loại **bình luận**, trong
đó title và description là ngữ cảnh giúp hiểu bình luận đang nói tới chủ đề gì.

## 2. Ba subtask

### A — stereotype detection

Đầu ra là `yes` hoặc `no`: bình luận có chứa một khuôn mẫu/định kiến về cộng
đồng LGBTQIA+ hay không.

Ví dụ EN `training_EN_0010` được gán `stereotype=yes`, nhưng `hate_speech=no`:

> straight men fetish me, straight women make me into a predator, gay women
> think im straight, its like ive nowhere to belong

Điểm cần nhớ: có stereotype không đồng nghĩa chắc chắn là hateful. Một người
có thể mô tả trải nghiệm bị định kiến mà không tấn công LGBTQIA+.

### B — hate speech classification

Đầu ra có ba lớp:

- `yes_explicit`: công kích/thù ghét được nói trực tiếp;
- `yes_implicit`: thái độ thù ghét hoặc loại trừ được diễn đạt gián tiếp;
- `no`: không hateful hoặc có tính ủng hộ.

Ví dụ:

- EN `training_EN_0001`: “She's living in unrepentant defiance to God's moral
  law.” → `yes_explicit`, `individual_l`.
- EN `training_EN_0022`: “Keep it the way it was please” trong video về hạn chế
  hiến máu của người gay → `yes_implicit`, `group_g`.

A và B là hai nhãn khác nhau. Không được suy A từ B hoặc ngược lại; trong dữ
liệu có cả stereotype nhưng không hateful và hateful nhưng stereotype=no.

### C — target và identity

C chỉ áp dụng cho các dòng có `hate_speech != no`. Dòng không hateful luôn có
`target=none`.

Trường `target` là dạng rút gọn:

```text
group_l,g,lgbtqia+
```

phải được đọc thành:

```text
target_type = group
identities = [l, g, lgbtqia+]
```

Vì vậy C gồm hai phần:

1. đối tượng là `individual` hay `group`;
2. một hoặc nhiều identity code.

Không được coi toàn bộ chuỗi `group_l,g,lgbtqia+` là một class đơn. Các code
identity cần đối chiếu glossary chính thức của task trước khi diễn giải ngữ
nghĩa trong bài viết.

## 3. Vì sao phải chia dữ liệu theo video?

Một video xuất hiện với nhiều comment. Nếu random-split theo comment, title và
description của cùng video có thể nằm ở cả train và validation. Khi đó điểm số
có thể phản ánh việc nhớ video thay vì hiểu comment. Pipeline hiện tại nhóm theo
`yt_title + yt_description` và tạo năm fold không giao nhau theo video.

## 4. Baseline hiện tại có ý nghĩa gì?

Baseline TF-IDF chỉ là control rẻ để trả lời ba câu hỏi:

1. comment-only đã mạnh đến đâu;
2. title có giúp không;
3. description có giúp không.

Nó chưa phải hệ thống dự thi và chưa đủ để kết luận mô hình nào tốt. Kết quả
hiện tại mới dùng một fold; cần chạy đủ năm fold rồi mới so sánh công bằng.

## 5. Câu hỏi nghiên cứu tối thiểu

Sau khi xác nhận scorer chính thức, pipeline nên kiểm tra:

1. Context video có cải thiện A/B so với comment-only không?
2. Context có ích hơn cho C, đặc biệt target type, không?
3. Mô hình đa ngôn ngữ có giữ được hiệu năng trên IT/NL và zero-shot sang
   Persian hay không?
4. Các nhãn hiếm như `A`, `Q`, `I`, `NB` có được dự đoán thật hay chỉ làm macro
   score dao động do support quá nhỏ?
