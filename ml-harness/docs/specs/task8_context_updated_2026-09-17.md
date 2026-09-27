# SemEval-2027 Task 8 - StereoQueerEval: Thông tin và kế hoạch chuẩn bị

> Cập nhật lần cuối: 17/09/2026 (giờ Việt Nam; kiểm tra web khoảng 22:51 ICT)
>
> Phạm vi: phần thi SemEval-2027 liên quan trực tiếp đến LGBTQIA+, tình hình thông tin đã công bố, quy định của cuộc thi, các nghiên cứu trước đây, phần còn thiếu và danh sách việc cần chuẩn bị.

## 1. Tóm tắt nhanh

Phần thi liên quan trực tiếp đến LGBT trong SemEval-2027 là **Task 8: StereoQueerEval - Identification of Stereotypes and Hate Speech Targeting the LGBTQIA+ Community**. Phần thi thuộc nhóm “Stereotypes, Framing, and Sentiment Analysis” và đã có tên trong danh sách chính thức của SemEval-2027.[^1]

Ý tưởng cốt lõi: phân loại một bình luận của người dùng YouTube, dùng **tiêu đề video và mô tả video làm ngữ cảnh**. Phần thi có ba phần:

1. **A - Phát hiện định kiến:** kiểm tra bình luận có hay không có định kiến về người LGBTQIA+.
2. **B - Phân loại lời nói thù ghét:** phân biệt lời thù ghét trực tiếp, lời thù ghét ẩn ý và nội dung không thù ghét hoặc mang tính ủng hộ.
3. **C - Xác định đối tượng và nhóm bản dạng:** với những bình luận mang tính thù ghét, xác định đối tượng là một người hay một nhóm, đồng thời xác định nhóm LGBTQIA+ được nhắc đến.

Theo bản đề xuất đã công bố, phần thi dự kiến có bốn ngôn ngữ: **English, Italian, Dutch, Farsi**. Người tham gia có thể chọn phần và ngôn ngữ mình muốn. Kết quả dự kiến được xếp hạng bằng **Macro F1**, tạo thành 12 bảng xếp hạng (4 ngôn ngữ x 3 phần). Đây là thông tin trong bản đề xuất, cần kiểm tra lại khi ban tổ chức phát hành hướng dẫn và cách chấm điểm chính thức.[^3][^4]

Tình hình vào ngày cập nhật **17/09/2026**:

- Task 8 **vẫn có tên trong danh sách chính thức của SemEval-2027** với cùng sáu thành viên ban tổ chức và địa chỉ liên hệ `stereoqueereval@gmail.com`.[^27]
- Tuy nhiên, bản danh sách Task 8 hiện tại trên SemEval chỉ hiển thị liên kết **contact organizers**; liên kết website của StereoQueerEval từng được thấy trong lần kiểm tra 09/09 **không còn được hiển thị ở mục Task 8**. Đây là thay đổi về trang danh sách, **không phải bằng chứng phần thi bị hủy**.[^27]
- Trang lịch chung của SemEval đang hiển thị mốc **Training data: 01/09/2026**; trong khi lần kiểm tra 09/09 của tài liệu này ghi nhận lịch chung từng hiển thị 08/09 và trang riêng của StereoQueerEval ghi **18/09/2026**. Vì vậy, lịch công khai hiện có dấu hiệu không đồng bộ; mốc 18/09 nên được xem là **điểm cần kiểm tra lại ngay khi đến ngày**, không phải cam kết đã xác nhận lại ngày 17/09.[^28]
- GitHub organization của StereoQueerEval hiện vẫn chỉ hiển thị **một repository công khai**, `stereoqueereval.github.io`, nhưng phần website trên hồ sơ organization lại trỏ tới `stereoqueereval-semeval2027.github.io`. Sự không nhất quán tên miền đã ghi nhận trước đây vẫn chưa được giải quyết rõ trên các nguồn được lập chỉ mục.[^29]
- Trong lần tìm kiếm công khai ngày 17/09, **chưa tìm thấy một trang Codabench công khai được lập chỉ mục, bộ dữ liệu huấn luyện, chương trình chấm điểm hay baseline mới của Task 8**. Đây là kết quả tìm kiếm âm tính, không chứng minh rằng tài nguyên riêng tư, chưa lập chỉ mục hoặc chỉ gửi qua email không tồn tại.
- Vì ngày 18/09/2026 là ngày kế tiếp theo giờ Việt Nam, ưu tiên vận hành trước mắt là kiểm tra lại website/repository/email/Codabench vào ngày đó và ghi lại chính xác nguồn, thời điểm, checksum và điều khoản nếu dữ liệu xuất hiện.

Kết luận hiện tại: vẫn có đủ thông tin để chuẩn bị quy trình nghiên cứu và baseline, nhưng **chưa đủ thông tin công khai đã xác nhận ngày 17/09 để cố định schema nộp bài, bảng nhãn, split, giấy phép, scorer của phần C hoặc địa chỉ competition platform**. Phần cập nhật deep research ở Mục 23 bổ sung các nghiên cứu 2025–2026 có ảnh hưởng trực tiếp đến chiến lược mô hình.

## 2. SemEval-2027 là gì?

SemEval là chuỗi hội thảo và cuộc thi chung của SIGLEX thuộc ACL. Mỗi cuộc thi cung cấp bộ dữ liệu và cách chấm điểm giống nhau để các nhóm có thể so sánh kết quả. SemEval-2027 được gọi là **The 21st International Workshop on Semantic Evaluation** và hiện có 11 phần thi.[^1][^9]

Task 8 nằm trong nhóm:

> **Stereotypes, Framing, and Sentiment Analysis**

Lời mời gửi đề xuất của SemEval yêu cầu một phần thi phải có vấn đề đủ mới và có ý nghĩa, bộ dữ liệu có giấy phép phù hợp, chất lượng gán nhãn tốt, công cụ kiểm tra tệp nộp, chương trình chấm điểm, mô hình tham khảo, trang web và kế hoạch duy trì dữ liệu sau cuộc thi. Đây là yêu cầu chung của SemEval; không có nghĩa là mọi phần việc của StereoQueerEval đã hoàn tất.[^9]

### 2.1. Lịch SemEval-2027 cấp hội thảo

Các trang SemEval-2027 đang có một số khác biệt về ngày dữ liệu mẫu/huấn luyện. Bảng dưới đây giữ nguyên nguồn và ghi rõ trạng thái thay vì tự chọn một ngày duy nhất.

| Mốc | Thông tin đã công bố | Diễn giải sử dụng |
|---|---|---|
| 13/04/2026 | Hạn nộp đề xuất, theo giờ Anywhere on Earth | Đã qua. |
| 01/06/2026 | Thông báo phần thi được chọn | Đã qua; Task 8 đã có trong danh sách chính thức. |
| 15/07/2026 | Lịch dự kiến: có dữ liệu mẫu | Đây là mốc trong lịch dự kiến của SemEval.[^9] |
| 08/08/2026 | Trang chính SemEval ghi có dữ liệu mẫu | Khác với mốc 15/07 trong trang thông báo.[^9] |
| 08/09/2026 | Lần kiểm tra 09/09 của tài liệu này ghi nhận lịch chung từng hiển thị 08/09, với mốc 01/09 bị gạch | **Dấu vết lịch sử** từ lần kiểm tra trước; trang SemEval được kiểm tra lại ngày 17/09 hiện hiển thị 01/09 thay vì 08/09.[^9][^28] |
| 18/09/2026 | Trang riêng của StereoQueerEval được kiểm tra ngày 09/09 ghi bắt đầu giai đoạn huấn luyện và có dữ liệu huấn luyện | Đây vẫn là mốc riêng gần nhất đã ghi nhận cho Task 8, nhưng **chưa xác nhận lại được ngày 17/09**; cần kiểm tra lại ngay 18/09.[^7] |
| 01/12/2026 | Có dữ liệu đánh giá; đây là hạn nội bộ, không công bố | Mốc nội bộ của ban tổ chức.[^9] |
| 10/01/2027 | Bắt đầu đánh giá | Mốc chung.[^9] |
| 31/01/2027 | Kết thúc đánh giá; ban tổ chức có thể kết thúc sớm hơn | Mốc muộn nhất chung.[^9] |
| 02/2027 | Nộp bài viết, dự kiến | Bao gồm bài mô tả phần thi và hệ thống.[^9] |
| 03/2027 | Thông báo kết quả, dự kiến | Mốc dự kiến.[^9] |
| 04/2027 | Nộp bản cuối để xuất bản, dự kiến | Mốc dự kiến.[^9] |
| Mùa hè 2027 | Hội thảo SemEval, tổ chức cùng một hội nghị NLP lớn | Hội nghị cụ thể chưa được nêu trên các trang đã kiểm tra.[^9] |

**Cách đọc lịch sau lần kiểm tra 17/09:** 18/09 là mốc riêng của StereoQueerEval đã được chụp lại trong lần kiểm tra 09/09; lịch chung hiện được truy cập ngày 17/09 lại hiển thị 01/09. Vì các nguồn đã thay đổi hoặc không đồng bộ, không chọn một ngày duy nhất làm “ngày chính thức” nếu chưa có thông báo trực tiếp của Task 8. Cần kiểm tra website, email ban tổ chức, repository và competition platform trước khi chốt kế hoạch chạy thử.[^28]

## 3. Nhận diện chính thức của Task 8

| Mục | Thông tin |
|---|---|
| Số phần thi | 8 |
| Tên ngắn | StereoQueerEval |
| Tên đầy đủ | Identification of Stereotypes and Hate Speech Targeting the LGBTQIA+ Community |
| Nhóm SemEval | Stereotypes, Framing, and Sentiment Analysis |
| Phạm vi | Định kiến, lời nói thù ghét, đối tượng và nhóm bản dạng LGBTQIA+ |
| Nguồn dữ liệu được mô tả | Bình luận YouTube, tiêu đề video và mô tả video |
| Trang web Task 8 đã dùng trong lần kiểm tra 09/09 | <https://stereoqueereval.github.io/>; lưu ý danh sách SemEval kiểm tra 17/09 hiện không còn hiển thị link website cạnh Task 8.[^27] |
| Trang tổ chức trên GitHub | <https://github.com/StereoQueerEval/> |
| Địa chỉ liên hệ chung | <code>stereoqueereval@gmail.com</code> |
| Trang Task 8 trên SemEval | <https://semeval.github.io/SemEval2027/tasks.html> |

Trang web của phần thi tự mô tả đây là **cuộc thi chung đầu tiên** tại SemEval-2027 về phát hiện định kiến và lời nói thù ghét nhắm đến cộng đồng LGBTQIA+. Mục tiêu là phát hiện tốt hơn ngôn ngữ gây hại bằng một bộ dữ liệu nhiều ngôn ngữ, bao phủ cả định kiến và lời nói thù ghét ẩn ý đối với nhiều nhóm bản dạng trong cộng đồng queer.[^2]

Lưu ý về nguồn thông tin: trong lần kiểm tra 09/09, trang SemEval từng dẫn tới <code>stereoqueereval.github.io</code> và địa chỉ đó được dùng làm nguồn Task 8 trong tài liệu này. Khi kiểm tra lại ngày 17/09, mục Task 8 trên danh sách SemEval chỉ còn liên kết liên hệ ban tổ chức, **không còn hiển thị website**; đồng thời GitHub organization vẫn trỏ hồ sơ website tới tên miền có hậu tố <code>-semeval2027</code>.[^27][^29] Vì vậy, không nên dùng việc “được SemEval link trực tiếp” như tiêu chí hiện thời để chọn giữa hai tên miền; cần ưu tiên thông báo mới từ ban tổ chức hoặc repository nếu xuất hiện.

## 4. Quy định của phần thi đã công bố

### 4.1. Dữ liệu được dùng

Mỗi mẫu dữ liệu được mô tả ở mức khái quát gồm:

- một bình luận trên YouTube;
- tiêu đề video;
- mô tả video.

Bình luận là phần cần phân loại; tiêu đề và mô tả được dùng để cung cấp thêm ngữ cảnh. Đây là điểm khác quan trọng so với cách chỉ đọc bình luận, vì một bình luận ngắn hoặc mơ hồ có thể cần tiêu đề và mô tả mới phân biệt được thảo luận mang tính ủng hộ, định kiến, lời nói thù ghét ẩn ý hay lời nói thù ghét trực tiếp.[^3][^4]

Chưa có thông tin chính thức về các điểm sau:

- tiêu đề và mô tả được đưa nguyên văn hay đã được làm sạch;
- tiêu đề và mô tả có luôn cùng ngôn ngữ với bình luận hay không;
- có kèm mã video, thông tin kênh, thời điểm hoặc mối quan hệ trả lời hay không;
- bình luận có giữ lại đường dẫn, biểu tượng cảm xúc, tên người dùng, thẻ đánh dấu hoặc phần ngôn ngữ trộn lẫn hay không;
- một video có thể xuất hiện trong nhiều phần dữ liệu hay không.

### 4.2. Ba phần của cuộc thi

| Phần | Bài toán | Nhãn hoặc kết quả đã công bố | Ý nghĩa |
|---|---|---|---|
| A - Stereotype Detection | Phân loại hai nhóm | Có định kiến / không có định kiến | Kiểm tra bình luận có chứa định kiến về LGBTQIA+ hay không. |
| B - Hate Speech Classification | Phân loại ba nhóm | Thù ghét trực tiếp / thù ghét ẩn ý / không thù ghét hoặc ủng hộ | Xác định loại lời nói trong bình luận nhắm đến LGBTQIA+. |
| C - Target & LGBTQIA+ Identity Identification | Gán nhiều nhãn và phân loại nhiều nhóm | Đối tượng là cá nhân hay nhóm, cùng nhãn nhóm bản dạng <code>L, G, B, T, Q, I, A, Q+, NB</code> | Về ý nghĩa, áp dụng cho bình luận mang tính thù ghét. |

Các mô tả trên lấy từ trang mô tả phần thi và bản đề xuất PDF.[^3][^4]

#### Phần A - phát hiện định kiến

Phần A không chỉ phát hiện lời nói thù ghét. Một bình luận có thể chứa định kiến về LGBTQIA+ nhưng chưa chắc được xếp là lời nói thù ghét trực tiếp hoặc ẩn ý. Vì vậy, A nên được xem là một bài toán riêng với B. Không nên tự suy ra nhãn của A từ nhãn “không thù ghét hoặc ủng hộ” của B, hoặc làm ngược lại, nếu hướng dẫn cuối cùng không quy định như vậy.

#### Phần B - lời nói trực tiếp, ẩn ý và nội dung không thù ghét

Phần B tách lời nói thù ghét trực tiếp và lời nói thù ghét ẩn ý, đồng thời có nhóm không thù ghét hoặc mang tính ủng hộ. Thiết kế này nhằm đo cả ngôn ngữ gây hại rõ ràng lẫn cách diễn đạt tinh vi hơn; không nên gộp hai loại lời nói thù ghét thành một nhãn trong quy trình cuối.

#### Phần C - đối tượng và nhóm bản dạng

Phần C kết hợp hai loại thông tin:

1. **Loại đối tượng:** một cá nhân hay một nhóm.
2. **Nhóm bản dạng LGBTQIA+ cụ thể:** <code>L, G, B, T, Q, I, A, Q+, NB</code>.

Bản đề xuất gọi C là bài toán có nhiều nhãn và nhiều nhóm, đồng thời mô tả đây là phần có thể được thực hiện sau B theo hai bước. Tuy nhiên, ban tổ chức **không bắt buộc hệ thống phải dùng kết quả của B**. Bảng giải thích đầy đủ các chữ viết tắt chưa được công bố trên trang mô tả phần thi. Đặc biệt cần xác nhận ý nghĩa chính thức của A và Q+, cũng như cách xử lý khi một bình luận nhắc đến nhiều nhóm bản dạng, trước khi huấn luyện hoặc chấm điểm.

### 4.3. Mối liên hệ giữa các phần và quyền chọn tham gia

Trang web và bản đề xuất nêu rõ:

- A có thể làm độc lập.
- B có thể làm độc lập.
- C được hình dung như phần có thể chạy sau B, nhưng không bắt buộc phải chạy B trước.
- Có thể tham gia bất kỳ tổ hợp nào: A, B, C, A+B, A+B+C, B+C và các tổ hợp khác.
- Có thể tham gia một hoặc nhiều ngôn ngữ.

Vì vậy, khi nghiên cứu có thể tách hai cách làm:

- **Dự đoán độc lập:** mỗi phần tự đưa ra kết quả.
- **Dự đoán theo hai bước:** dùng xác suất hoặc kết quả của B làm thông tin bổ sung cho C.

Khi so sánh hai cách làm, cần giữ nguyên cách chia dữ liệu, tài nguyên dành cho mô hình, số lần khởi tạo ngẫu nhiên và mức thử nghiệm. Không nên kết luận cách chạy hai bước tốt hơn chỉ vì nó được dùng thêm thông tin mà cách làm độc lập không có.

## 5. Ngôn ngữ và thách thức chính

Bản đề xuất mô tả bốn ngôn ngữ như sau:[^4]

| Ngôn ngữ | Vai trò được bản đề xuất mô tả | Điều cần chuẩn bị |
|---|---|---|
| English | Nhiều dữ liệu, nhiều thảo luận trên mạng và dễ so sánh với các cuộc thi khác | Dùng một cách làm tốt cho nhiều ngôn ngữ và một cách làm riêng cho tiếng Anh để đối chiếu; kiểm tra kỹ ngữ cảnh và lời nói ẩn ý. |
| Italian | Lượng dữ liệu trung bình, bối cảnh châu Âu và tiếp nối HODI 2023 | Có thể tham khảo các nghiên cứu HODI, nhưng không được mặc định rằng cách hiểu nhãn giống HODI. |
| Dutch | Lượng dữ liệu trung bình, bối cảnh châu Âu và hướng tới các tài nguyên mới về queer | Kiểm tra mức phù hợp của các mô hình đã học trước với tiếng Hà Lan và sự khác biệt giữa dữ liệu huấn luyện với dữ liệu cuộc thi. |
| Farsi | Ít dữ liệu, bối cảnh văn hóa khác biệt và ít được nghiên cứu về thảo luận liên quan đến queer | Chuẩn bị cách chuyển kiến thức giữa các ngôn ngữ, xem xét mức mất cân bằng và phân tích lỗi theo bối cảnh văn hóa. |

Bản đề xuất nói bộ dữ liệu sẽ “cân bằng”, nhưng ở thời điểm cập nhật chưa có số lượng mẫu, phân bố nhãn hoặc bảng cân bằng theo ngôn ngữ và phần thi. Vì vậy, “cân bằng” hiện chỉ là **mục tiêu được nêu trong bản đề xuất**, chưa phải chỉ số đã được kiểm chứng.

## 6. Đánh giá và tham gia

### 6.1. Chỉ số và cách xếp hạng đã được công bố

Bản đề xuất nêu:

- có một bảng xếp hạng cho mỗi ngôn ngữ trong mỗi phần;
- tổng cộng **12 bảng xếp hạng = 4 ngôn ngữ x 3 phần**;
- chỉ số chính dự kiến là **Macro F1**;
- cách làm được mô tả là tham khảo phương pháp của BRIGHTER 2025 và POLAR 2026.[^4]

Trang web của phần thi cũng nêu người tham gia phải nộp kết quả và báo cáo mô tả cách làm, gồm thuật toán, mô hình, tài nguyên, thí nghiệm và phân tích kết quả.[^5]

### 6.2. Những điểm chấm điểm chưa đủ rõ

Macro F1 đã được công bố ở mức khái quát, nhưng chưa có các thông tin sau:

- Phần A và B được tính Macro F1 theo từng nhóm như thế nào trong chương trình chấm điểm chính thức;
- Phần C được tính trên nhãn bản dạng, nhãn đối tượng, cặp kết hợp của hai loại nhãn hay nhiều điểm thành phần;
- phần C dùng ngưỡng cố định, ngưỡng được điều chỉnh trên dữ liệu phát triển hay quy tắc khác;
- dự đoán trống hoặc nhãn đúng bị bỏ trống được tính ra sao;
- bảng xếp hạng có tính trung bình qua các ngôn ngữ hay chỉ xếp riêng từng ngôn ngữ;
- cách xử lý khi hai kết quả bằng điểm, số lần nộp tối đa và cách chọn kết quả cuối;
- có báo cáo thêm độ chính xác, độ bao phủ, điểm theo từng nhóm, điều chỉnh ngưỡng hoặc đánh giá thủ công hay không.

Vì vậy, chưa nên gọi điểm tự tính trên máy cá nhân là “điểm chính thức” nếu chưa chạy đúng chương trình chấm điểm của ban tổ chức.

### 6.3. Nộp bài và báo cáo

Trang “How to participate” hiện mới mô tả quy trình ở mức sơ bộ:

- cuộc thi sẽ dùng Codabench, nhưng địa chỉ chưa xuất hiện;
- hướng dẫn về phần thi, dữ liệu, định dạng tệp nộp và cách nộp bài chưa xuất hiện;
- người tham gia cần nộp kết quả và báo cáo mô tả cách làm.

Trang “Nhận dữ liệu” nói bộ dữ liệu huấn luyện và kiểm tra được cấp sau khi điền biểu mẫu, sau đó ban tổ chức sẽ gửi hướng dẫn qua email. Trang cũng gợi ý người dùng Windows dùng WinRAR để giải nén và trỏ tới trang tổ chức trên GitHub để lấy chương trình chấm điểm và mô hình tham khảo. Tuy nhiên, các tệp đang được công khai chưa cho thấy kho riêng chứa những chương trình hoặc mô hình đó.[^5][^6][^13]

## 7. Ban tổ chức và liên hệ

### 7.1. Danh sách chính thức

Danh sách phần thi SemEval hiện liệt kê sáu ban tổ chức:[^1]

- Alessandra Teresa Cignarella - Ghent University
- Greta Damo - Université Côte d’Azur
- Marta Marchiori Manerba - University of Turin
- Marzieh Abdolmaleki - Ghent University
- Els Lefever - Ghent University
- Debora Nozza - Bocconi University

Bản đề xuất cung cấp vai trò chi tiết hơn:[^4]

- **Người phụ trách chính:** Alessandra T. Cignarella.
- **Đồng phụ trách:** Marta Marchiori Manerba, Greta Damo, Marzieh Abdolmaleki.
- **Cố vấn:** Debora Nozza, Els Lefever.
- **Hỗ trợ ngôn ngữ và kiểm tra dữ liệu:** Joke Daems, Joni Kruijsbergen.

Địa chỉ liên hệ chung trên trang web của phần thi: **stereoqueereval@gmail.com**.[^8]

### 7.2. Kinh nghiệm trước đây của nhóm

Ba thành viên ban tổ chức StereoQueerEval từng liên quan đến HODI 2023: Debora Nozza, Alessandra Teresa Cignarella và Greta Damo. Bản đề xuất cho biết phần thi mới dựa trên kinh nghiệm của nhóm HODI nhưng mở rộng sang định kiến, lời nói thù ghét ẩn ý, nhiều mức độ chi tiết hơn về nhóm bản dạng và nhiều ngôn ngữ.[^4][^10]

## 8. Vì sao phần thi này đáng chú ý?

StereoQueerEval không chỉ lặp lại việc phát hiện homophobia và transphobia theo hai nhóm. Những điểm mới được bản đề xuất nhấn mạnh gồm:

1. **Phát hiện định kiến riêng:** nhận ra định kiến ẩn hoặc gắn với bối cảnh văn hóa ngay cả khi bình luận chưa phải lời nói thù ghét rõ ràng.
2. **Phân biệt lời nói thù ghét trực tiếp và ẩn ý:** mở rộng từ lời nói thù ghét rõ ràng sang cách diễn đạt gây hại tinh vi hơn.
3. **Phân biệt nhiều nhóm bản dạng:** không xem LGBTQIA+ là một nhóm duy nhất.
4. **Dùng dữ liệu YouTube kèm ngữ cảnh:** tiêu đề và mô tả giúp giải thích bình luận.
5. **Nhiều ngôn ngữ và bối cảnh văn hóa:** English, Italian, Dutch và Farsi; trong đó bản đề xuất xem Farsi là ngôn ngữ có ít dữ liệu và bối cảnh văn hóa khác biệt.
6. **Quyền chọn phần thi linh hoạt:** cho phép chọn phần và ngôn ngữ thay vì bắt buộc phải làm toàn bộ các phần theo một chuỗi.

Các điểm trên là mục tiêu và cách thiết kế dự kiến; hiệu quả thực tế chỉ có thể đánh giá sau khi bộ dữ liệu, gán nhãn, cách chia dữ liệu và chương trình chấm điểm được phát hành.

## 9. Các cuộc thi và nghiên cứu liên quan trước đây

### 9.1. HODI tại EVALITA 2023

HODI là cuộc thi chung đầu tiên về phát hiện homotransphobia bằng tiếng Ý, do một nhóm có thành viên trùng với StereoQueerEval tổ chức. HODI dùng các bài đăng tiếng Ý và có hai phần:

- **Phần A:** phân loại hai nhóm: có tính thù ghét và không có tính thù ghét.
- **Phần B:** xác định đoạn văn giải thích lý do của quyết định.

HODI cung cấp 5.000 bài đăng để huấn luyện và 1.000 bài đăng để kiểm tra; khoảng 40% dữ liệu huấn luyện và 51% dữ liệu kiểm tra mang tính thù ghét. Phần A xếp hạng bằng F1 trung bình qua các nhóm. Phần B so sánh phần chữ được chọn với câu trả lời đúng bằng độ trùng nhau giữa ký tự hoặc từ, rồi tính F1. Bài tổng kết ghi nhận 19 lần nộp cho A và 5 lần nộp cho B, đến từ 8 nhóm ở 6 quốc gia.[^10]

Bài học có thể áp dụng cho Task 8:

- phần yêu cầu giải thích hoặc chọn đoạn chữ thường khó hơn và có thể thu hút ít người tham gia hơn;
- cần công bố rõ mức thống nhất giữa người gán nhãn và cách gộp các nhãn;
- quy định về mô hình và nhãn cần tách rõ lời nói thù ghét, phần giải thích và ngữ cảnh;
- không được mặc định cách hiểu nhãn của HODI giống StereoQueerEval.

### 9.2. LT-EDI tại EACL 2024

Cuộc thi chung LT-EDI 2024 tập trung vào homophobia và transphobia trong bình luận trên mạng xã hội, chủ yếu là YouTube, với 10 ngôn ngữ: English, Tamil, Malayalam, Telugu, Kannada, Gujarati, Hindi, Marathi, Spanish và Tulu. Mỗi bình luận được gán vào một trong ba nhóm; kết quả dùng Macro F1. Bài tổng kết ghi nhận hơn 60 nhóm tham gia qua CodaLab.[^11]

Đây là tiền lệ hữu ích cho:

- phân loại bình luận YouTube bằng nhiều ngôn ngữ;
- tình trạng chênh lệch số lượng giữa các nhóm và ngôn ngữ ít dữ liệu;
- việc dùng Macro F1 để xếp hạng;
- khả năng một cuộc thi về vấn đề xã hội thu hút nhiều nhóm nghiên cứu nếu cách tham gia và quyền nhận dữ liệu được công bố rõ.

### 9.3. LT-EDI 2025

LT-EDI 2025 có hai phần:

- phân loại bình luận thành nhiều nhóm: Homophobia, Transphobia và Non-anti-LGBT+ trên tám ngôn ngữ;
- phát hiện đoạn chữ liên quan trên English, Tamil và Marathi.

Bài tổng kết ghi nhận nhiều nhóm đăng ký nhưng chỉ hai nhóm nộp kết quả; bảng xếp hạng dùng Macro F1.[^12]

Bài học cho StereoQueerEval:

- thêm nhiều phần hoặc nhiều nhãn không tự động làm số nhóm tham gia tăng;
- phần thi cần chương trình chấm điểm, mô hình tham khảo, quyền nhận dữ liệu, hướng dẫn và hạn nộp rõ ràng;
- phần C có nhiều nhãn bản dạng và loại đối tượng, nên nếu cách nộp không đơn giản thì độ phức tạp có thể làm giảm số người tham gia.

### 9.4. So sánh nhanh

| Khía cạnh | HODI 2023 | LT-EDI 2024/2025 | StereoQueerEval 2027 |
|---|---|---|---|
| Nguồn | Bài đăng tiếng Ý | Chủ yếu là bình luận YouTube | Bình luận YouTube, tiêu đề và mô tả |
| Nội dung thù ghét | Hai nhóm và phần giải thích | Homophobia, transphobia, non-anti-LGBT+ | Thù ghét trực tiếp, thù ghét ẩn ý, không thù ghét hoặc ủng hộ |
| Định kiến riêng | Không phải trọng tâm chính | Không phải trọng tâm chính | Có Phần A riêng |
| Mức độ chi tiết về bản dạng | LGBTQIA+ ở mức khái quát | Chủ yếu homo/trans | L/G/B/T/Q/I/A/Q+/NB |
| Đoạn giải thích | Có ở HODI và LT-EDI 2025 | Có ở LT-EDI 2025 | Chưa thấy kết quả đoạn chữ trong quy định hiện tại |
| Ngôn ngữ | Italian | 10 hoặc 8 tùy năm và phần | English, Italian, Dutch, Farsi theo bản đề xuất |
| Cách xếp hạng | Macro F1; IOU/F1 cho phần giải thích | Macro F1 | Dự kiến Macro F1 |

## 10. Điều đã xác nhận và điều chưa được công bố

### 10.1. Đã xác nhận từ nguồn chính thức và các trang đã công bố

- Task 8 vẫn có tên trong danh sách SemEval-2027 ngày 17/09/2026; tên phần thi, sáu organizer và email liên hệ vẫn được hiển thị.[^27]
- Website Task 8 tại `stereoqueereval.github.io` đã được dùng và kiểm tra trong snapshot 09/09; **danh sách SemEval ngày 17/09 hiện không còn hiển thị link website bên cạnh Task 8**.[^27]
- Dữ liệu được mô tả gồm bình luận YouTube, với tiêu đề và mô tả làm ngữ cảnh.
- Có ba phần A, B và C, cùng mô tả khái quát về từng phần.
- C có liên quan đến B về mặt ý tưởng nhưng không bắt buộc phải dùng kết quả của B.
- Người tham gia có thể chọn phần và ngôn ngữ theo mô tả/bản đề xuất đã công bố.
- Bản đề xuất nêu bốn ngôn ngữ: English, Italian, Dutch và Farsi.
- Bản đề xuất nêu 12 bảng xếp hạng và dùng Macro F1.
- Snapshot website Task 8 ngày 09/09 ghi bắt đầu giai đoạn huấn luyện là 18/09/2026; **chưa coi đây là release đã xác nhận lại ngày 17/09**.[^7]
- Người tham gia được mô tả là cần nộp kết quả và báo cáo mô tả cách làm.

### 10.2. Chưa được công bố hoặc chưa đủ thông tin để kết luận

- Số lượng dữ liệu huấn luyện, phát triển và kiểm tra theo ngôn ngữ và phần thi.
- Phân bố nhãn và mức chênh lệch thực tế giữa các nhóm.
- Ý nghĩa chính xác của <code>L, G, B, T, Q, I, A, Q+, NB</code>.
- Hướng dẫn gán nhãn, thành phần người gán nhãn, mức thống nhất, cách giải quyết bất đồng và cách gộp nhãn.
- Giấy phép bộ dữ liệu, quy định chia sẻ lại, điều khoản YouTube, cách xử lý dữ liệu bị xóa và thông tin cá nhân.
- Cấu trúc dữ liệu chính xác, tên cột, mã định danh, cách mã hóa ký tự, các trường ngữ cảnh và cách xử lý giá trị bị thiếu.
- Cách chia dữ liệu để tránh trùng video, trùng bình luận hoặc để thông tin từ cùng một video lọt sang nhiều phần.
- Công thức chấm điểm chính xác, nhất là cách chấm nhiều nhãn và nhiều nhóm ở C.
- Định dạng tệp nộp, số lần nộp tối đa, giới hạn nộp và quy định về dữ liệu hoặc mô hình bên ngoài.
- Địa chỉ Codabench và đường dẫn hướng dẫn của phần thi.
- Mã nguồn, phiên bản mô hình tham khảo và chương trình đánh giá.
- Hạn nộp riêng của phần thi nếu khác lịch chung của SemEval.
- Việc có hay không có bảng xếp hạng sau giai đoạn đánh giá, bộ kiểm tra kín và bộ kiểm tra công khai.
- Bài mô tả chính thức của StereoQueerEval 2027 trong tài liệu hội thảo.

### 10.3. Không nên suy luận từ các nguồn liên quan

- Không suy ra số lượng dữ liệu StereoQueerEval từ HODI hoặc LT-EDI.
- Không suy ra định kiến đồng nghĩa với lời nói thù ghét.
- Không suy ra một bình luận chỉ được gán một nhóm bản dạng.
- Không suy ra C bắt buộc phải nhận kết quả của B.
- Không suy ra Farsi bao gồm mọi phương ngữ Ba Tư hoặc mọi cách viết nếu hướng dẫn chưa nói rõ.
- Không suy ra từ nhận định “cân bằng” rằng mọi ngôn ngữ và mọi nhãn đều có số lượng cân bằng.
- Không dùng điểm của HODI hoặc LT-EDI làm mô hình tham khảo chính thức cho Task 8.

## 11. Danh sách việc cần làm khi dữ liệu và hướng dẫn được công bố

### Giai đoạn 0 - trước khi có dữ liệu

- Chuẩn bị chương trình đọc được nhiều cách sắp xếp dữ liệu, gồm <code>id</code>, <code>language</code>, <code>comment</code>, <code>title</code>, <code>description</code> và các nhãn nếu có.
- Tách cách chuẩn bị theo từng cặp ngôn ngữ và phần thi; không viết cố định chỉ cho English và Italian.
- Chuẩn bị cách tính điểm tạm thời trên máy cá nhân cho A, B và C, đồng thời ghi rõ đây chỉ là cách tính tạm thời cho tới khi có chương trình chính thức.
- Chuẩn bị kết quả để có thể nộp từng phần hoặc nhiều phần cùng lúc.
- Ghi lại mô hình, lần khởi tạo, cách chia dữ liệu, bộ tách từ, độ dài tối đa, ngưỡng và phiên bản bảng nhãn.

### Giai đoạn 1 - ngay sau khi dữ liệu được phát hành

- Kiểm tra mã hóa ký tự, ô trống, bản ghi trùng, nhãn không khớp ngôn ngữ, độ dài bình luận và ngữ cảnh, cũng như việc thông tin từ cùng một video có bị lẫn giữa các phần hay không.
- Lập bảng ghi mã kiểm tra tệp, số dòng, tên trường, nhãn, phần dữ liệu và ghi chú về quyền sử dụng.
- Đối chiếu với hướng dẫn về bảng nhãn, cách C xử lý trường hợp không có nhãn, mối quan hệ giữa đối tượng và nhóm bản dạng, cùng chương trình chấm điểm chính thức.
- Đo số lượng theo ngôn ngữ, phần thi và nhãn; không gọi là “cân bằng” chỉ vì bản đề xuất dùng từ đó.
- So sánh cách chỉ dùng bình luận với cách dùng cả ngữ cảnh trên cùng một phần dữ liệu phát triển.

### Giai đoạn 2 - cách làm ban đầu tối thiểu

- A: một mô hình dùng chung cho nhiều ngôn ngữ để phân loại hai nhóm định kiến.
- B: một mô hình phân loại ba nhóm: thù ghét trực tiếp, thù ghét ẩn ý, không thù ghét hoặc ủng hộ.
- C: một phần dự đoán các nhóm bản dạng và một phần dự đoán loại đối tượng; thử cả cách dự đoán độc lập và cách dự đoán theo hai bước.
- So sánh khi thêm hoặc bớt ngữ cảnh: chỉ bình luận; bình luận và tiêu đề; bình luận và mô tả; bình luận, tiêu đề và mô tả.
- Báo cáo Macro F1 theo ngôn ngữ, theo nhóm hoặc bản dạng và bằng bảng nhầm lẫn; không chỉ báo cáo một điểm trung bình chung.

Các mô hình cụ thể chỉ là đề xuất nghiên cứu, không phải yêu cầu của phần thi. Có thể dùng một mô hình nhiều ngôn ngữ như XLM-R hoặc mDeBERTa để đối chiếu; với Italian có thể thử thêm mô hình riêng. Mọi so sánh phải dùng cùng cách chia dữ liệu và mức thử nghiệm.

### Giai đoạn 3 - kiểm tra độ ổn định và phân tích

- Điều chỉnh ngưỡng của C trên dữ liệu phát triển, không điều chỉnh trên bộ kiểm tra kín hoặc bộ kiểm tra cuối.
- Kiểm tra các nhóm bản dạng ít xuất hiện, ngôn ngữ trộn lẫn, nhiều cách viết, cách dùng lại từ từng mang nghĩa xúc phạm, cách nói giảm nói tránh, mỉa mai, thảo luận ủng hộ và các trường hợp cần ngữ cảnh mới hiểu được.
- Phân tích các trường hợp báo nhầm khi bình luận chỉ nhắc đến một nhóm bản dạng nhưng không mang tính thù ghét.
- Phân tích các trường hợp bỏ sót định kiến hoặc lời nói thù ghét ẩn ý không chứa từ ngữ miệt thị rõ ràng.
- Tách lỗi theo ngôn ngữ và nhóm bản dạng; Macro F1 có thể che khuất vấn đề của các nhãn hiếm.
- Chỉ giữ lại kết quả dự đoán và đoạn bằng chứng ở mức cần thiết; hạn chế đưa lại các bình luận gây hại nguyên văn vào báo cáo.

### Giai đoạn 4 - nộp bài và viết báo cáo

- Chạy công cụ kiểm tra định dạng chính thức khi ban tổ chức công bố.
- Chốt bảng ghi thông tin gồm mã thay đổi của chương trình, môi trường chạy, phiên bản mô hình, phiên bản dữ liệu, phiên bản chương trình chấm điểm, ngưỡng và mã kiểm tra của tệp nộp.
- Kiểm tra tệp nộp chỉ chứa dự đoán hợp lệ, không có dòng thay thế và không có nhãn ngoài danh sách cho phép.
- Báo cáo cần nêu rõ tài nguyên, mô hình đã học trước, dữ liệu bên ngoài, cách chuyển kiến thức giữa các ngôn ngữ, so sánh khi thêm bớt ngữ cảnh, điểm theo từng ngôn ngữ và phần thi, phân tích lỗi và các giới hạn.

## 12. Các câu hỏi nên gửi ban tổ chức

Nếu muốn tham gia nghiêm túc, đây là những câu hỏi quan trọng nhất:

1. Bốn ngôn ngữ trong bản đề xuất đã được chốt cho dữ liệu huấn luyện, phát triển và kiểm tra chưa? Có ngôn ngữ nào bị thay đổi không?
2. A, Q+ và NB trong bảng nhãn bản dạng chính thức có nghĩa là gì? A là asexual, aromantic hay một nhóm gộp?
3. C có điểm riêng cho loại đối tượng và nhóm bản dạng không, hay chấm trên từng cặp kết hợp của hai loại nhãn?
4. C được tính Macro F1 nhiều nhãn theo cách nào: trung bình theo nhãn, theo mẫu, hay dùng kết quả khớp hoàn toàn?
5. Nếu chỉ bình luận mang tính thù ghét mới có nhãn C, thì các bình luận còn lại cần nộp kết quả rỗng, nhãn “none” hay không nộp dòng đó?
6. Tiêu đề và mô tả có luôn đi kèm bình luận không? Có được dùng thêm thông tin YouTube hoặc dữ liệu trên web bên ngoài không?
7. Giấy phép dữ liệu, việc xóa hoặc ẩn thông tin trong bình luận, quy định về thông tin cá nhân và quyền lưu dữ liệu sau cuộc thi như thế nào?
8. Địa chỉ Codabench, số lần nộp tối đa, định dạng tệp nộp, thời gian dữ liệu phát triển, thời gian kiểm tra kín và hạn cuối là gì?
9. Khi nào chương trình chấm điểm, công cụ kiểm tra định dạng, mô hình tham khảo và dữ liệu mẫu được công bố?
10. Dữ liệu có được chia theo video hoặc người dùng để tránh thông tin bị lặp giữa các phần không?

## 13. Đánh giá hiện tại

### Điểm mạnh

- Vấn đề có ý nghĩa xã hội và thách thức thực tế: định kiến, lời nói thù ghét ẩn ý, ngữ cảnh, nhiều ngôn ngữ và nhiều mức độ chi tiết về nhóm bản dạng.
- Cấu trúc các phần khá rõ ở mức khái quát.
- Nhóm tổ chức có các nghiên cứu trước đây liên quan trực tiếp đến HODI và lời nói thù ghét nhắm đến LGBTQIA+.
- Có Farsi và Dutch, giúp phần thi không chỉ tập trung vào English và Italian.
- Macro F1 cùng bảng xếp hạng theo ngôn ngữ phù hợp với mục tiêu không để nhóm hoặc ngôn ngữ có nhiều dữ liệu lấn át nhóm hoặc ngôn ngữ ít dữ liệu.

### Rủi ro cần theo dõi

- Quy định của phần thi chưa đủ chi tiết để người khác có thể lặp lại chính xác cách làm.
- C có thể rất khó nếu vừa có nhiều nhãn bản dạng vừa có nhiều loại đối tượng mà chưa có bảng giải thích nhãn và chương trình chấm điểm rõ ràng.
- Nhận định “bộ dữ liệu cân bằng” chưa có số liệu chứng minh.
- Quyền nhận dữ liệu và các công cụ phục vụ cuộc thi **chưa thấy được công bố đầy đủ qua các nguồn công khai được kiểm tra ngày 17/09**; tài nguyên riêng qua email vẫn có thể tồn tại.
- Quyền sử dụng bình luận YouTube, thông tin riêng tư, nội dung đã bị xóa và cách xử lý nội dung gây hại cần được làm rõ.
- Lịch chung của SemEval và mốc Task 8 đã chụp ngày 09/09 đang lệch nhau; thậm chí trang lịch chung kiểm tra lại 17/09 hiển thị 01/09 thay vì 08/09 đã ghi trước đó.[^28]
- Snapshot website ngày 09/09 có placeholder ở form/Codabench/guideline; đến 17/09, danh sách SemEval không còn hiển thị website Task 8 và GitHub organization vẫn có hai cách đặt tên miền khác nhau.[^27][^29] Đây là rủi ro vận hành, không phải bằng chứng task bị hủy.

### Kết luận thực tế

StereoQueerEval đáng theo dõi và có thể bắt đầu chuẩn bị ngay ở mức **quy trình nghiên cứu và mô hình ban đầu bám sát các quy định đã biết**. Chưa nên điều chỉnh lớn hoặc tuyên bố đã sẵn sàng cho cuộc thi cho đến khi có:

1. bộ dữ liệu thật cùng giấy phép và quyền truy cập;
2. hướng dẫn chính thức và bảng nhãn;
3. chương trình chấm điểm và công cụ kiểm tra định dạng;
4. ngày và địa chỉ cuộc thi chính xác;
5. thống kê dữ liệu và cách chia dữ liệu.

## 14. Kiểm tra bổ sung về mức độ sẵn sàng ngày 09/09/2026

> **Snapshot lịch sử:** Mục 14 giữ nguyên các quan sát ngày 09/09 để theo dõi thay đổi. Trạng thái hiện thời ngày 17/09 nằm ở Mục 1 và Mục 23; nếu có xung đột, dùng phần cập nhật mới hơn.

### 14.1. Quy định đã khá rõ, nhưng các trang và công cụ phục vụ cuộc thi chưa hoàn thiện

Kiểm tra lại các tệp công khai trên GitHub và mã của website chính thức cho thấy:

- Trang mô tả phần thi đang hoạt động và xác nhận đúng ba phần A, B, C; dữ liệu đưa vào là bình luận YouTube cùng tiêu đề và mô tả làm ngữ cảnh.[^14]
- Trang **How to participate** vẫn ghi Codabench và hướng dẫn là “link to be appeared”, tức sẽ bổ sung đường dẫn sau; trang cũng xác nhận người tham gia phải nộp kết quả và báo cáo mô tả cách làm.[^15]
- Trang **Get the data** vẫn yêu cầu điền biểu mẫu, nhưng liên kết trong mã của trang là `href="."`, tức trỏ về chính trang hiện tại. Trang nói hướng dẫn nhận dữ liệu sẽ được gửi qua email và chương trình chấm điểm cùng mô hình tham khảo sẽ ở GitHub.[^16]
- Trang **Important Dates** hiện chỉ ghi `18 Sept 2026 - Dữ liệu huấn luyện available (giai đoạn huấn luyện starts)`; chưa có hạn đánh giá riêng được công bố trên trang này.[^17]
- Trang tổ chức trên GitHub hiện chỉ cho thấy kho chính là website; danh sách tệp công khai chưa cho thấy kho riêng chứa bộ dữ liệu, chương trình chấm điểm hoặc mô hình tham khảo.[^18]
- Trang chính còn dấu vết của mẫu cũ: phần mô tả vẫn là `Shared task Homotransphobia Detection in Italian`, một số phần tài liệu tham khảo bị ẩn trong mã, và trang mô tả có thông tin tạo hoặc sửa từ năm 2019 dù nội dung nói về SemEval-2027.[^14]

Các dấu hiệu trên cho thấy **website chưa hoàn thiện**, nhưng không phải bằng chứng rằng phần thi bị hủy hoặc quy định đã thay đổi. Khi các nguồn mâu thuẫn nhau, nên ưu tiên theo thứ tự:

1. Danh sách phần thi chính thức và lời mời hoặc lịch chung của SemEval-2027.
2. Hướng dẫn, chương trình chấm điểm và trang cuộc thi do ban tổ chức phát hành.
3. Trang mô tả phần thi và bản đề xuất PDF.
4. Website và các tệp gốc trên GitHub.
5. Bài tổng kết của các cuộc thi trước và nghiên cứu liên quan, chỉ dùng làm tiền lệ tham khảo; không dùng để thay thế nhãn hoặc quy định nộp bài của Task 8.

### 14.2. Cách ghi lại những thay đổi trong các lần cập nhật sau

Mỗi khi phần thi có thêm tài liệu hoặc tệp, cần ghi lại ít nhất:

| Nội dung | Cần ghi lại |
|---|---|
| Thời điểm kiểm tra | Giờ UTC và giờ Việt Nam. |
| Nguồn | Địa chỉ chính xác của trang hoặc tệp. |
| Loại tài liệu | Website, hướng dẫn, bộ dữ liệu, chương trình chấm điểm, mô hình tham khảo hoặc Codabench. |
| Phiên bản hoặc mã thay đổi | Mã thay đổi, thẻ phiên bản hoặc số phiên bản hiển thị; nếu không có thì ghi “chưa cung cấp”. |
| Mã kiểm tra tệp | Mã SHA-256 của tệp đã tải về, nếu được phép lưu tệp. |
| Trạng thái | Đã xác nhận, tạm thời, đã thay thế hoặc đang vướng. |
| Ảnh hưởng | Có thay đổi cấu trúc dữ liệu, bảng nhãn, chỉ số, hạn nộp, giấy phép hay chỉ thay đổi thông tin trên website. |

Không ghi đè lên bộ dữ liệu hoặc chương trình chấm điểm gốc. Mỗi tài liệu hoặc tệp mới nên được lưu kèm bảng ghi thông tin, đồng thời giữ bản cũ để sau này biết một kết quả trên máy cá nhân được tạo theo quy định nào.

## 15. Các nghiên cứu trước đây và việc có thể chuẩn bị

### 15.1. Bộ dữ liệu YouTube nhiều ngôn ngữ trước đây

Một bài nghiên cứu về bình luận YouTube bằng nhiều ngôn ngữ công bố bộ dữ liệu gồm 15.141 bình luận, xây dựng hệ thống phân loại nhiều tầng và dùng dữ liệu do chuyên gia gán nhãn để nghiên cứu homophobia và transphobia.[^19] Bài tổng kết LT-EDI 2022 tách số liệu thành English 4.946, Tamil 4.161 và Tamil-English 6.034; nhãn `Non-anti-LGBTQ+ content` chiếm tỷ lệ rất lớn, nên Macro F1 được dùng làm chỉ số chính.[^20]

Điều nên chuẩn bị cho Task 8:

- có báo cáo số lượng theo nhóm và theo ngôn ngữ; độ chính xác hoặc F1 tính chung có thể che khuất các nhãn hiếm;
- kiểm tra ngôn ngữ trộn lẫn, cách viết chuyển tự và khác biệt chữ viết, thay vì chỉ tin vào mã ngôn ngữ trong tệp;
- giữ lại thông tin video hoặc nguồn nhóm nếu ban tổ chức cung cấp, vì bình luận cùng video có thể làm kết quả bị lẫn thông tin;
- xem bộ dữ liệu cũ là **tài liệu tham khảo và dữ liệu bên ngoài có thể cân nhắc**, không mặc định rằng được phép dùng để huấn luyện.

Bộ dữ liệu cũ không thể thay thế StereoQueerEval vì khác ngôn ngữ, hệ thống nhãn, ngữ cảnh dữ liệu, mức độ chi tiết về bản dạng và cách lấy mẫu. Nếu được phép dùng dữ liệu bên ngoài, phải kiểm tra trùng lặp trước khi đưa vào.

### 15.2. Bằng chứng về thiên lệch trong cách lấy mẫu và gán nhãn

LT-EDI 2022 cho biết bình luận được thu từ các video giải thích về LGBTQ+, đồng thời tránh các video kể chuyện coming-out cá nhân vì lý do bảo mật. Đây là một lựa chọn có lý do đạo đức, nhưng cũng có thể tạo thiên lệch vì chủ đề video đã được chọn trước.[^20] Vì thế, khi Task 8 phát hành tiêu đề và mô tả, nên kiểm tra xem mô hình đang học ngôn ngữ gây hại hay chỉ học chủ đề hoặc dấu hiệu của video.

HODI 2023 là tiền lệ gần về nhóm tổ chức: dữ liệu được gán bởi ba thành viên cộng đồng LGBTQIA+ người Ý, qua ba bước. Phần A dùng biểu quyết đa số, còn phần giải thích được gộp lại. Bài viết báo cáo mức thống nhất ở từng bước và thừa nhận rằng việc lấy mẫu theo từ khóa hoặc sự kiện không phản ánh đầy đủ sự phân bố tự nhiên của lời nói thù ghét.[^22] Khi chuẩn bị, nên tách riêng:

- phân tích mức đồng thuận và những trường hợp khó phân biệt;
- phân tích các trường hợp báo nhầm do từ ngữ chỉ bản dạng;
- phân tích việc chủ đề hoặc sự kiện làm kết quả bị lẫn thông tin;
- phân biệt giữa “nói về một nhóm bản dạng” và “tấn công nhóm đó”.

Kho HODI hiện cũng giới thiệu HODIAT với thông tin về mức độ gay gắt và đối tượng, được gán bởi ba người gán nhãn; muốn truy cập cần liên hệ nhóm tổ chức.[^23] Đây là nguồn tham khảo cho việc mô tả đối tượng và nguồn gốc dữ liệu, nhưng không được đưa vào quy trình Task 8 nếu chưa có quyền sử dụng và chưa kiểm tra trùng lặp.

### 15.3. Một số bộ dữ liệu mở rộng khác

Một bài viết LREC-COLING 2024 mô tả khoảng 10.000 bình luận YouTube bằng Telugu, Kannada và Gujarati, có nhãn do chuyên gia gán và mô hình tham khảo dùng transformer.[^21] Bộ dữ liệu này không nằm trong bốn ngôn ngữ được đề xuất cho Task 8, nhưng hữu ích khi chuẩn bị cho ngôn ngữ ít dữ liệu, chuyển kiến thức giữa các ngôn ngữ và lập danh sách dữ liệu bên ngoài có thể xem xét. Không nên dùng điểm của các bộ dữ liệu đó để dự đoán kết quả Task 8.

## 16. Điều kiện quan trọng về quyền truy cập, riêng tư và giấy phép sử dụng

Trước khi huấn luyện hoặc tải thêm bình luận từ YouTube, cần làm rõ các điều kiện sau:

| Điều kiện | Câu hỏi bắt buộc | Bằng chứng cần lưu | Nếu chưa có |
|---|---|---|---|
| G1 - quyền truy cập | Nhận dữ liệu qua biểu mẫu hoặc email nào, ai được dùng và được dùng trong bao lâu? | Email, điều khoản hoặc thông tin đăng ký | Chỉ chuẩn bị chương trình, chưa huấn luyện. |
| G2 - giấy phép | Được lưu, xử lý, công bố văn bản gốc, kết quả dự đoán và ví dụ lỗi ở mức nào? | Thông báo giấy phép hoặc quy định sử dụng dữ liệu | Không đưa văn bản gốc vào GitHub hoặc báo cáo. |
| G3 - nguồn dữ liệu YouTube | Ban tổ chức gửi văn bản gốc hay gửi mã và thông tin để lấy lại dữ liệu? | Tài liệu mô tả dữ liệu và câu trả lời của ban tổ chức | Không tự thu thập thêm để “bù” dữ liệu. |
| G4 - thông tin cá nhân | Có tên người dùng, mã tác giả, đường dẫn, thời điểm, mã video hoặc cây trả lời không? | Kiểm tra cấu trúc dữ liệu và ghi chú về việc loại bỏ thông tin nhạy cảm | Bỏ hoặc mã hóa các trường không cần thiết trước khi dùng. |
| G5 - dữ liệu bị xóa | Có cách xử lý bình luận hoặc video đã bị xóa hay không còn truy cập được không? | Hướng dẫn xử lý dữ liệu bị xóa hoặc được cập nhật | Không tuyên bố rằng bộ dữ liệu có thể được tái tạo đầy đủ. |
| G6 - dữ liệu bên ngoài | Có được dùng HODI, LT-EDI, dữ liệu do mô hình tạo hoặc bản dịch không? | Câu trả lời rõ trong hướng dẫn | Cách làm chuẩn phải chỉ dùng dữ liệu được phép. |

Nếu phải lấy lại dữ liệu YouTube, chỉ dùng giao diện lập trình hợp lệ sau khi đọc điều khoản và hướng dẫn của ban tổ chức. Điều khoản của YouTube nói dữ liệu qua giao diện lập trình có thể bao gồm bình luận và phải tuân thủ các chính sách dành cho nhà phát triển cùng điều khoản YouTube; điểm `comments.list` cho phép lấy bình luận theo mã hoặc bình luận cha và có tùy chọn `plainText`.[^24][^25] Vì vậy, không nên mặc định rằng chương trình thu thập dữ liệu cũ hoặc đường dẫn bộ dữ liệu của bên thứ ba là an toàn để dùng.

SemEval cũng đặt ra các yêu cầu chung về giấy phép, riêng tư, mức thống nhất khi gán nhãn, công cụ kiểm tra định dạng, chương trình chấm điểm, mô hình tham khảo và khả năng lưu giữ dữ liệu sau cuộc thi. Việc thông tin đã được công bố không có nghĩa là được phép đưa toàn bộ văn bản lên mạng.[^26] Với Task 8, đây là danh sách điều cần xác nhận thực tế, không phải giả định rằng mọi việc đã hoàn tất chỉ vì phần thi đã được chọn.

## 17. Các tài liệu và tệp nên chuẩn bị trước khi dữ liệu huấn luyện được mở

### 17.1. Cách sắp xếp thư mục đề xuất

```text
task8_work/
  README.md
  data/
    incoming/             # file ban tổ chức gửi, chỉ đọc
    raw/                  # bản dữ liệu gốc đã ghi mã kiểm tra, không chỉnh sửa
    processed/            # parquet/JSONL sau khi chuẩn hóa trong nhóm
    manifests/            # cấu trúc dữ liệu, mã kiểm tra, nguồn gốc và ghi chú giấy phép
  configs/
    task8_languages.yaml
    label_map_provisional.json
    experiment_matrix.yaml
  src/
    io_contract.py
    normalize.py
    audit_schema.py
    audit_leakage.py
    metrics_provisional.py
    build_submission.py
    train.py
  reports/
    data_card.md
    data_audit.md
    baseline_report.md
    error_taxonomy.md
  runs/
  submissions/
  environment/
    requirements.lock.txt
```

Đây là cách sắp xếp thư mục trong nhóm, **không phải định dạng tệp nộp**. Chỉ những tệp và trường được hướng dẫn chính thức xác nhận mới được dùng trong `build_submission.py`.

### 17.2. Mẫu dữ liệu thống nhất trong nhóm

Khi nhận dữ liệu, đưa về cùng một mẫu để quy trình không phụ thuộc vào tên cột do ban tổ chức đặt:

```text
example_id              string, required, unique within split
language                enum candidate: en/it/nl/fa, verify from data
comment                 string, required
video_title             string or null
video_description       string or null
split                   train/dev/test, verify official meaning
label_a                 opaque until official glossary is confirmed
label_b                 opaque until official glossary is confirmed
target_type             opaque or null
identity_labels         list[string] or null
source_group_id         optional, never submit unless instructed
```

Không tự gán nghĩa cho <code>L/G/B/T/Q/I/A/Q+/NB</code> trước khi nhận bảng giải thích chính thức. Trong chương trình, tạm thời coi đây là các chuỗi nhãn; chỉ thêm tên dễ đọc trong một tệp riêng có ghi phiên bản.

### 17.3. Điều kiện kiểm tra ban đầu của dữ liệu

Chỉ bắt đầu chạy mô hình tham khảo khi các kiểm tra đơn giản sau đã đạt:

- mã hóa ký tự ổn định, không mất chữ Unicode, biểu tượng cảm xúc hoặc ký tự Ba Tư;
- <code>example_id</code> không trùng trong cùng một phần dữ liệu và không bị tự ý đánh số lại mà không ghi nhận;
- danh sách nhãn khớp hướng dẫn, không tự thay nhãn lạ bằng một nhãn khác;
- đếm riêng bình luận, tiêu đề và mô tả bị bỏ trống;
- đối chiếu mã ngôn ngữ với nội dung mẫu và ghi rõ cách xử lý khi trộn ngôn ngữ;
- báo cáo văn bản trùng hoàn toàn và văn bản gần giống nhau;
- nếu có <code>video_id</code>, <code>channel_id</code>, mã chủ đề hoặc mã trả lời thì kiểm tra cách chia dữ liệu để thông tin từ cùng nhóm không bị lẫn;
- không có dữ liệu huấn luyện và kiểm tra bị trùng do làm sạch, dịch hoặc lặp lại ngữ cảnh;
- lưu mã SHA-256 của bản dữ liệu gốc và bảo đảm dữ liệu đã xử lý có thể lần ngược về dòng gốc;
- chương trình chấm điểm và công cụ kiểm tra định dạng chạy được trên mẫu do ban tổ chức cung cấp;
- chốt lại `label_map.json`, `data_schema.json`, `data_card.md` và `split_audit.md` trước khi thử nghiệm.

## 18. Các thử nghiệm ban đầu có thể chạy sau khi có dữ liệu

### 18.1. Các bước thử từ đơn giản đến phức tạp

| Mã | Cách làm | Mục đích | Điều kiện để xem xét bước tiếp theo |
|---|---|---|---|
| E0 | Chọn nhóm xuất hiện nhiều nhất hoặc chọn ngẫu nhiên theo tỷ lệ | Kiểm tra chương trình chấm điểm và mức chênh lệch giữa các nhóm | Chỉ dùng để kiểm tra, không dùng để kết luận mô hình tốt. |
| E1 | Word + character TF-IDF + Logistic Regression | Cách làm đơn giản để đối chiếu, nhận ra cách viết khác nhau, từ miệt thị và ngôn ngữ trộn | Đọc đúng dữ liệu, chạy lại cho cùng kết quả và có báo cáo theo từng nhóm. |
| E2 | E1, tách riêng bình luận, tiêu đề và mô tả | Đo lợi ích của ngữ cảnh bằng một cách làm đơn giản | So sánh trên cùng cách chia dữ liệu và cùng lần khởi tạo. |
| E3 | Mô hình dùng chung cho nhiều ngôn ngữ, chỉ đọc bình luận | Có một cách làm mạnh hơn để đối chiếu khi chưa dùng ngữ cảnh | Không điều chỉnh dựa trên bộ kiểm tra kín hoặc bộ kiểm tra cuối. |
| E4 | Mô hình dùng chung cho nhiều ngôn ngữ, đọc đủ ngữ cảnh | Đo tác dụng của tiêu đề và mô tả | Báo cáo rõ kết quả khi thêm hoặc bớt ngữ cảnh. |
| E5 | Một mô hình chung với các phần dự đoán A, B và C | Khai thác mối liên hệ giữa các phần | Chỉ chạy khi nhãn C và chương trình chấm điểm đã rõ; so với E3 và E4. |
| E6 | C độc lập so với C dùng kết quả của B | Kiểm tra việc chạy theo hai bước có thật sự hữu ích không | Dùng cùng mức tài nguyên và không xem cách chạy hai bước là bắt buộc. |

Chỉ chốt phiên bản mô hình sau khi biết giấy phép, độ dài tối đa và ngôn ngữ hoặc chữ viết thực tế. Cách làm chuẩn đầu tiên nên chỉ dựa vào dữ liệu được phép của cuộc thi, ngoài bộ tách từ hoặc mô hình đã học trước; mọi kho dữ liệu bên ngoài, bản dịch hoặc nhãn do máy tạo phải được ghi riêng.

### 18.2. So sánh tác dụng của ngữ cảnh và kiểm tra thông tin bị lẫn

Nên chạy ít nhất các cách sau:

1. chỉ dùng bình luận;
2. bình luận và tiêu đề;
3. bình luận và mô tả;
4. bình luận, tiêu đề và mô tả;
5. chỉ dùng tiêu đề và mô tả;
6. đổi ngữ cảnh để đối chiếu trên dữ liệu phát triển: giữ nguyên bình luận, thay tiêu đề và mô tả bằng ngữ cảnh của một video khác cùng ngôn ngữ.

Nếu chỉ dùng ngữ cảnh cho điểm bất thường cao hoặc việc đổi ngữ cảnh làm kết quả thay đổi mạnh, cần kiểm tra lại cách chia theo video và khả năng mô hình đang học chủ đề của video trước khi điều chỉnh mô hình. Tiêu đề và mô tả được phép dùng, nhưng cũng có thể khiến bài toán trở nên quá dễ theo cách không mong muốn.

### 18.3. Cách tính điểm tạm thời khi chưa có chương trình chấm điểm cho C

Bản đề xuất nêu Macro F1 là chỉ số chính, nhưng chưa có công thức đủ chi tiết cho C. Cho tới khi có chương trình chấm điểm chính thức, có thể tính tạm thời:

- A: Macro F1 cho hai nhóm;
- B: Macro F1 cho ba nhóm;
- C - loại đối tượng: Macro F1 cho loại đối tượng;
- C - bản dạng: Macro F1 riêng cho từng nhãn bản dạng;
- C - kết hợp: chỉ tính để tham khảo khi có định nghĩa rõ, không gọi là điểm chính thức;
- báo cáo thêm theo ngôn ngữ, theo nhóm hoặc bản dạng, bảng nhầm lẫn, số lượng mẫu và ngưỡng.

Không dùng kết quả trống, nhãn tự thay thế hoặc nhãn đúng do ban tổ chức cung cấp để “lấp” phần C. Nếu bình luận không mang tính thù ghét không cần kết quả C, vẫn giữ chúng trong dữ liệu trên máy cá nhân nhưng để phần tạo tệp nộp làm theo hướng dẫn; không tự suy ra `none` hoặc một tập rỗng.

### 18.4. Thử nghiệm nhỏ và điều kiện chọn cách làm tốt hơn

Đợt đầu nên giới hạn ở E0-E4, với 1–3 lần khởi tạo ngẫu nhiên và không thử quá nhiều tổ hợp. Chỉ chuyển sang mô hình tiếp theo khi:

- chương trình chấm điểm chính thức hoặc tạm thời chạy đúng trên mẫu kiểm tra;
- không có thông tin bị lẫn và không vi phạm quy định dữ liệu;
- Macro F1 tăng trên phần phát triển đã chốt so với cách làm đối chiếu;
- nhóm hoặc bản dạng hiếm không giảm nghiêm trọng dù điểm chung tăng;
- lợi ích của việc thêm ngữ cảnh được giải thích bằng các so sánh trên, không chỉ do dấu hiệu quá dễ từ video hoặc chủ đề;
- bảng ghi thông tin của mỗi lần chạy lưu phiên bản chương trình, mã kiểm tra dữ liệu, mô hình, lần khởi tạo, bộ tách từ, độ dài tối đa, ngưỡng và môi trường chạy.

Nếu hướng dẫn hoặc chương trình chấm điểm chưa có sau khi dữ liệu được phát hành, dừng ở việc kiểm tra dữ liệu và chạy nhanh cách làm đối chiếu; không điều chỉnh lớn và không tuyên bố về điểm trên cuộc thi.

## 19. Kế hoạch kiểm tra độ ổn định, sự công bằng và lỗi

Nên chuẩn bị các nhóm kiểm tra riêng ngay từ đầu:

- bình luận nhắc đến một nhóm bản dạng nhưng không mang tính thù ghét hoặc có tính ủng hộ;
- từ ngữ miệt thị rõ ràng so với định kiến ẩn ý hoặc cách nói giảm nói tránh;
- cách dùng lại từ từng mang nghĩa xúc phạm và lời nói được trích dẫn hoặc thuật lại;
- mỉa mai, phủ định, câu hỏi tu từ và câu trả lời trong cuộc trò chuyện;
- bình luận cần tiêu đề hoặc mô tả mới hiểu được đang nói tới ai hay điều gì;
- cách viết khác nhau, biểu tượng cảm xúc, đường dẫn, thẻ đánh dấu, ngôn ngữ trộn lẫn và chuyển tự;
- các cách viết hoặc chữ viết khác nhau của Farsi/Persian nếu dữ liệu có;
- đối tượng là cá nhân so với nhóm;
- các nhãn bản dạng ít xuất hiện và bình luận nhắc đến nhiều nhóm bản dạng;
- bình luận, tiêu đề hoặc mô tả đã bị xóa, bị bỏ trống hoặc bị thiếu.

Các chỉ số sau giúp hiểu lỗi, nhưng không thay thế điểm xếp hạng chính thức:

- Macro F1 theo từng ngôn ngữ;
- độ chính xác, độ bao phủ, F1 và số lượng mẫu theo từng nhóm hoặc bản dạng;
- tỷ lệ báo nhầm trên bình luận chỉ nhắc đến bản dạng nhưng không thù ghét;
- đường biểu diễn kết quả khi thay đổi ngưỡng của phần C;
- bảng lỗi theo từng nhóm và khoảng tin cậy nếu số mẫu đủ lớn;
- mức độ kết quả thay đổi khi thay ngữ cảnh và tỷ lệ lỗi ở các mẫu trùng hoặc gần trùng.

Không nên đưa bình luận gây hại nguyên văn vào nhật ký công khai, mã chương trình, bảng theo dõi hoặc báo cáo quá mức cần thiết. Bảng tổng hợp lỗi nên ưu tiên <code>example_id</code>, nhãn đúng, nhãn dự đoán, nhóm kiểm tra và lý do; chỉ lưu văn bản gốc ở nơi hạn chế người được xem nếu giấy phép cho phép.

## 20. Kế hoạch nội bộ đề xuất

Các mốc dưới đây là **kế hoạch nội bộ**, không phải hạn chính thức mới:

| Khoảng thời gian | Việc | Kết quả cần có |
|---|---|---|
| 09–17/09/2026 | Tạo thư mục làm việc, chốt môi trường, đọc nghiên cứu trước đây và chuẩn bị email | Chương trình chưa phụ thuộc vào cấu trúc dữ liệu cụ thể; chưa có tuyên bố về kết quả huấn luyện. |
| 18–19/09/2026 | Theo dõi mốc dữ liệu huấn luyện; nhận dữ liệu qua kênh ban tổ chức | Có ghi nhận về quyền truy cập, giấy phép và cấu trúc dữ liệu, hoặc ghi rõ điểm còn vướng. |
| 20–22/09/2026 | Kiểm tra ban đầu về cấu trúc dữ liệu, nhãn, trùng lặp, thông tin bị lẫn và ngữ cảnh bị thiếu | `data_audit.md`, `data_schema.json`, mã SHA-256. |
| Tuần 1 sau khi phát hành | E0-E2 trên từng ngôn ngữ và trên dữ liệu gộp | Bảng kết quả đối chiếu và kết quả theo từng nhóm kiểm tra. |
| Tuần 2–3 | E3-E4 và so sánh khi thêm hoặc bớt ngữ cảnh | Ghi chú quyết định về việc dùng ngữ cảnh. |
| Sau khi chương trình chấm điểm C rõ ràng | E5-E6 và điều chỉnh ngưỡng | Ghi chú đánh giá phần C; không dùng điểm tạm thời để nộp. |
| Trước 01/12/2026 | Chốt bản chương trình và bảng ghi thông tin trước hạn dữ liệu nội bộ | Một bản có thể chạy lại với cùng dữ liệu và quy định. |
| 10–31/01/2027 | Giai đoạn đánh giá theo lịch SemEval | Chỉ nộp tệp đã kiểm tra định dạng; giữ mã kiểm tra và nhật ký tệp nộp. |
| 02/2027 | Viết báo cáo mô tả cách làm | Báo cáo có tài nguyên, thử nghiệm, so sánh, giới hạn và vấn đề đạo đức. |

## 21. Mẫu email nếu dữ liệu hoặc hướng dẫn vẫn chưa mở

Có thể dùng mẫu tiếng Anh dưới đây để hỏi ban tổ chức:

```text
Subject: StereoQueerEval Task 8 – data access and guidelines

Dear StereoQueerEval organizers,

I am preparing a research system for SemEval-2027 Task 8. The public task page
currently lists 18 September 2026 as the start of the training period, but the
Codabench and guidelines links still appear as placeholders.

Could you please share or confirm:

1. the participant registration/data-access link and expected delivery date;
2. the final train/dev/test schema and official label glossary;
3. the exact Subtask C output and Macro-F1 computation, including empty labels;
4. the split policy for comments from the same video/thread/user;
5. whether external datasets, translation, retrieved context, or generative models are allowed;
6. the submission filename/format, run limit, Codabench URL, and final deadline;
7. the data license, retention/deletion policy, and whether raw comments may be stored or quoted in a technical report.

I will follow the task terms and will not redistribute the data. Thank you for
your help.

Best regards,
[Name / affiliation / contact]
```

## 22. Việc cần làm tiếp theo

1. Giữ `task8_context.md` làm sổ theo dõi nghiên cứu; khi có tài liệu hoặc tệp mới, ghi thêm ngày kiểm tra và nguồn thay vì âm thầm sửa các kết luận cũ.
2. Chuẩn bị cấu trúc ban đầu trên máy cá nhân, bảng ghi thông tin về dữ liệu và cách tính điểm tạm thời; chưa cần tải dữ liệu bên ngoài.
3. Theo dõi mốc 18/09/2026. Nếu dữ liệu hoặc hướng dẫn vẫn chưa có địa chỉ thật, gửi email ở trên và ghi lại điểm còn vướng.
4. Khi nhận dữ liệu, kiểm tra dữ liệu trước khi huấn luyện, rồi mới chạy E0-E2; chỉ chạy C và E5-E6 sau khi chương trình chấm điểm và bảng nhãn đã rõ.
5. Trong mọi báo cáo, tách rõ ba loại thông tin: `thông tin chính thức của phần thi`, `kết quả kiểm tra trên máy cá nhân` và `giả thuyết nghiên cứu`.

## 23. Cập nhật deep research ngày 17/09/2026

Mục này **bổ sung và, khi có xung đột về trạng thái hiện thời, thay thế các ảnh chụp trạng thái ngày 09/09** ở các mục trước. Các mô tả lịch sử vẫn được giữ để biết một kết luận đã dựa trên phiên bản nguồn nào.

### 23.1. Điều mới về trạng thái Task 8

| Kiểm tra ngày 17/09/2026 | Kết quả | Ảnh hưởng thực tế |
|---|---|---|
| Danh sách SemEval-2027 | Task 8 vẫn được liệt kê với tên `StereoQueerEval`, sáu organizer và email liên hệ; mục Task 8 hiện không hiển thị website/Codabench, chỉ có `contact organizers`.[^27] | **Không có bằng chứng phần thi bị hủy**, nhưng không nên giả định website cũ vẫn là kênh công bố chính chỉ vì nó từng được link ngày 09/09. |
| Lịch chung SemEval | Trang chủ được truy cập ngày 17/09 hiển thị `Training data: September 1, 2026`, khác với ảnh chụp 09/09 trong tài liệu này từng ghi 08/09.[^28] | Lịch chung đang có dấu hiệu thay đổi/không đồng bộ. Với Task 8, phải ưu tiên thông báo riêng của organizer khi có. |
| GitHub organization | Hồ sơ hiện hiển thị 1 repository công khai là `stereoqueereval.github.io`; website profile lại là `stereoqueereval-semeval2027.github.io`.[^29] | Tiếp tục coi sự khác biệt tên miền là một điểm cần xác minh; không suy ra repository dữ liệu bị ẩn hay cuộc thi bị hủy. |
| Codabench / data / scorer / baseline | Tìm kiếm công khai ngày 17/09 không làm xuất hiện tài nguyên mới được lập chỉ mục cho StereoQueerEval. | Giữ trạng thái **chưa xác nhận công khai**; nếu organizer gửi qua email hoặc form riêng thì phải lưu bằng chứng truy cập và điều khoản. |
| Mốc 18/09 | Đây là mốc Task 8 riêng gần nhất đã ghi nhận từ website trong lần kiểm tra 09/09.[^7] | Vì 18/09 là ngày kế tiếp theo giờ Việt Nam, đây là **checkpoint ưu tiên số 1**, không phải bằng chứng rằng dữ liệu chắc chắn sẽ xuất hiện. |

Một tín hiệu phụ: một trang đề xuất dự án nghiên cứu của Heriot-Watt được lập chỉ mục ngày 17/09 vẫn mô tả sinh viên sẽ làm SemEval-2027 Task 8 với bình luận YouTube và title/description làm ngữ cảnh.[^30] Đây không phải nguồn quy định của cuộc thi, nhưng nó phù hợp với việc Task 8 vẫn đang được các nhóm bên ngoài chuẩn bị, thay vì cho thấy task đã biến mất.

### 23.2. HODIAT 2025: tiền lệ gần hơn cho phần C so với HODI 2023

HODIAT là phần mở rộng của HODI do Greta Damo, Alessandra Teresa Cignarella, Tommaso Caselli, Viviana Patti và Debora Nozza công bố năm 2025; ba người đầu/cuối trong nhóm tác giả cũng liên quan trực tiếp đến StereoQueerEval. HODIAT thêm hai lớp nhãn rất gần với những gì Task 8 cần suy nghĩ: **target = INDIVIDUAL/GROUP** và **aggressiveness = AGGRESSIVE/NON-AGGRESSIVE**.[^31]

Những điểm có giá trị trực tiếp cho kế hoạch Task 8:

- HODIAT có 6.000 tweet tiếng Ý; trong các mẫu hateful, nhãn target nghiêng nhiều về **individual**, còn aggressiveness rất hiếm. Đây là bằng chứng cụ thể rằng các lớp con “target/degree of harm” có thể mất cân bằng mạnh ngay cả khi lớp hate tổng thể không quá lệch.[^32]
- Thử nghiệm HODIAT cho thấy **target classification dao động nhiều giữa mô hình**, fine-tuning thường mạnh hơn zero/few-shot, và XLM-T là encoder mạnh trong thiết lập của bài đó. Quan trọng hơn, tác giả báo cáo rằng **huấn luyện ba nhãn chung có xu hướng kém hơn huấn luyện riêng**.[^33]
- Vì vậy, E5 của tài liệu này không nên chỉ thử một mô hình multi-task rồi mặc định chia sẻ biểu diễn sẽ tốt. Cần có ít nhất hai nhánh đối chiếu: `separate models/heads` và `joint multi-task`, cùng split/seed/tài nguyên. Nếu joint model tăng điểm A/B nhưng làm C giảm, không nên giữ chỉ vì kiến trúc đẹp hơn.
- Với C, phải báo cáo F1 từng nhãn/target và số mẫu, không chỉ một macro chung; nếu phân bố giống HODIAT, một mô hình có thể đạt điểm tổng tốt nhưng bỏ sót lớp hiếm.

### 23.3. MultiPRIDE 2026: reclaimed slurs là nhóm lỗi bắt buộc phải kiểm tra

MultiPRIDE tại EVALITA 2026 là shared task đa ngôn ngữ về **slur reclamation** trong ngữ cảnh LGBTQ+. Task A dùng nội dung bài đăng; Task B thêm ngữ cảnh từ biography của người dùng. Bài tổng kết ghi nhận 19 đội, 61 runs cho A và 27 runs cho B.[^34][^35]

Điểm quan trọng cho StereoQueerEval không phải là mang dữ liệu hay user profile của MultiPRIDE sang Task 8, mà là **định nghĩa nhóm lỗi**:

- cùng một từ từng được dùng như slur có thể là tấn công, trích dẫn, tự gọi trong nhóm, tái chiếm dụng hoặc dùng mang tính ủng hộ; do đó **lexicon slur không được phép trở thành shortcut cho hate**;
- Task B của StereoQueerEval đã có nhãn “không thù ghét hoặc ủng hộ”, nên cần tạo một slice riêng gồm `reclamation / self-reference / quotation / counterspeech / supportive mention` để đo false positive;
- MultiPRIDE cho thấy ngữ cảnh có thể quyết định nghĩa của từ, nhưng Task 8 hiện chỉ công bố **video title + description** làm context. Không được tự đưa biography, inferred identity hay profile data vào hệ thống Task 8 nếu hướng dẫn cuối cùng không cho phép;
- một system paper của chính Greta Damo tại MultiPRIDE 2026 tiếp tục xem các tín hiệu ngôn ngữ và sentiment là hữu ích cho reclaimed-slur detection, làm cho nhóm lỗi này càng đáng đưa vào error taxonomy của Task 8.[^36]

**Bổ sung vào error taxonomy:** ngoài “từ miệt thị rõ ràng”, phải tách ít nhất `offensive slur`, `reclaimed/in-group use`, `quoted slur`, `counterspeech`, `supportive discussion`, `identity mention without attack`. Đây là phân tích lỗi, không phải tự tạo nhãn mới cho cuộc thi.

### 23.4. Nghiên cứu 2026 về stereotype: cần coi định nghĩa nhãn là một phần của bài toán

Alessandra Teresa Cignarella, người phụ trách chính của StereoQueerEval, cùng Matteo Pellegrini công bố năm 2026 một tổng quan về cách làm việc với stereotype trong NLP. Bài viết nhấn mạnh rằng cách khái niệm hóa và vận hành hóa “stereotype” rất không đồng nhất giữa các nghiên cứu; nhóm tác giả còn thử nhiều hệ nhãn trước đây trên một phần **QUEEREOTYPES** để đánh giá độ rõ, mức chồng lấn và tính hữu dụng thực tế.[^37]

Song song, survey của Cignarella, Anastasia Giachanou và Els Lefever tổng hợp hơn 6.000 tài liệu ứng viên và nhấn mạnh nhu cầu nghiên cứu stereotype theo hướng **đa ngôn ngữ, intersectional và gắn với nền tảng lý thuyết rõ ràng**.[^38]

Hệ quả cho Subtask A:

- không nên xem A là “hate-light” hay tự ánh xạ A từ B; định nghĩa stereotype có thể có ranh giới riêng;
- ngay khi annotation guideline xuất hiện, cần lưu **nguyên văn định nghĩa, ví dụ biên, quy tắc phủ định/trích dẫn/mỉa mai**, rồi chuyển thành checklist lỗi;
- nếu dữ liệu cung cấp annotation rời theo annotator, không nên vứt bỏ chúng quá sớm để chỉ giữ majority label; disagreement có thể cho biết mẫu nào có ranh giới khái niệm khó.

### 23.5. Disagreement và few-shot: giữ thông tin bất đồng nếu organizer cung cấp

Caselli và Plaza-del-Arco (WOAH 2025) nghiên cứu toxic-language detection với annotation rời và entropy của bất đồng. Trong few-shot ICL, các ví dụ có mức ambiguity trung bình thường hữu ích hơn cách chọn đơn giản, nhưng **ICL vẫn thường kém fine-tuned encoders** trong các thí nghiệm của họ.[^39]

Ứng dụng cho Task 8:

- nếu chỉ có nhãn consensus: không thể tái tạo disagreement, chỉ đánh dấu các mẫu model không chắc chắn;
- nếu có nhãn từng annotator: lưu nguyên, tính entropy/disagreement và dùng làm **difficulty bucket**, không ép mọi mẫu thành “dễ/khó” dựa trên cảm giác;
- few-shot LLM nên là baseline hoặc error-analysis aid, không nên mặc định thay cho encoder fine-tuning chỉ vì task có nhiều ngôn ngữ.

### 23.6. Dịch sang English không nên là pipeline mặc định

Chan, Nguyen và Joshi (2024) đánh giá hate speech LGBTQIA+ trên English, Italian, Chinese và English-Tamil code-mixed. Kết quả của họ cho thấy fine-tuning cải thiện khá nhất quán, trong khi **machine translation cho kết quả lẫn lộn** và có thể làm mất sắc thái xã hội-văn hóa.[^40]

Vì StereoQueerEval dự kiến có Dutch và Farsi cùng English/Italian:

- baseline chính nên ưu tiên mô hình multilingual xử lý ngôn ngữ gốc;
- `translate-to-English` chỉ nên là **ablation** nếu quy định cho phép dịch/LLM/API bên ngoài;
- nếu thử dịch, phải báo cáo riêng theo ngôn ngữ và nhóm lỗi như code-switch, slang, reclaimed slur, implicit stereotype; không chỉ báo cáo điểm sau dịch;
- không dùng bản dịch để ghi đè văn bản gốc trong data pipeline.

### 23.7. Nghiên cứu mới 2026: context giúp phát hiện harm ẩn, nhưng phải chống shortcut

SpectrumNet tại ICWSM 2026 nghiên cứu LGBTQ+-targeted cyberbullying và dùng conversational context cùng cơ chế attention/fusion để phát hiện các trường hợp tinh vi khó thấy khi đọc từng comment độc lập.[^41] Dù dataset và bài toán khác Task 8, kết quả này củng cố lý do phải thử context một cách có kiểm soát.

Do đó, các ablation context ở Mục 18.2 nên được xem là **bắt buộc cho báo cáo**, đặc biệt:

1. comment only;
2. comment + title;
3. comment + description;
4. comment + title + description;
5. context only;
6. shuffled/mismatched context cùng ngôn ngữ.

Nếu `context only` quá mạnh hoặc shuffled context vẫn cho điểm cao bất thường, cần kiểm tra topic leakage/video leakage trước khi tăng độ phức tạp mô hình.

### 23.8. Anti-LGBTQ bias 2026: dữ liệu dịch và dữ liệu cộng đồng có thể cho kết luận khác nhau

Một preprint mới cuối tháng 08/2026 về anti-LGBTQ bias trong mô hình German/English so sánh benchmark dịch với stereotype do cộng đồng nói tiếng Đức cung cấp. Tác giả báo cáo khác biệt giữa hai nguồn và nhấn mạnh tầm quan trọng của **cultural adaptation**, đồng thời thấy fine-tuning giảm bias trung bình nhưng không đồng đều theo mô hình và identity.[^42]

Đây **không phải nghiên cứu trực tiếp về StereoQueerEval**, nhưng là cảnh báo phù hợp cho bốn ngôn ngữ dự kiến của Task 8: không nên giả định cùng một danh sách stereotype hoặc cùng cách dịch nhãn/trigger sẽ có nghĩa xã hội giống nhau giữa English, Italian, Dutch và Farsi.

### 23.9. Ma trận thử nghiệm cập nhật

Giữ E0–E6 ở Mục 18, nhưng bổ sung các điều kiện sau thay vì tăng mô hình một cách tràn lan:

| Mã bổ sung | Thử nghiệm | Lý do | Điều kiện |
|---|---|---|---|
| E5a | A/B/C dùng model hoặc head riêng | HODIAT cho thấy joint training có thể làm xấu nhiều nhãn | Chạy cùng backbone, split, seed với E5. |
| E5b | Joint multi-task A+B+C | Đo xem chia sẻ biểu diễn có thật sự giúp Task 8 | Chỉ sau khi schema C rõ; không lấy lợi ích A/B để che giảm C. |
| E7 | Reclamation/quotation/supportive stress slice | Giảm shortcut “slur = hate” theo bài học MultiPRIDE | Chỉ là diagnostic trên dev, không tự sửa gold label. |
| E8 | Native multilingual vs translate-to-English | Dịch có thể làm mất sắc thái | Chỉ chạy nếu external translation được phép; lưu cả text gốc và bản dịch. |
| E9 | Confidence/disagreement analysis | Hate/stereotype là nhiệm vụ chủ quan | Chạy nếu có probability; entropy annotator chỉ khi raw annotations được cung cấp. |
| E10 | Context robustness: context-only + shuffled context | Phát hiện leakage/shortcut từ video topic | Phải dùng cùng ngôn ngữ và split, không làm thay đổi comment. |

### 23.10. Ưu tiên hành động từ tối 17/09 sang 18/09

1. **Ngày 18/09:** kiểm tra lại danh sách SemEval, repository/website Task 8, email organizer và tìm Codabench; ghi giờ UTC + ICT và URL thực tế.
2. Nếu có data: **đọc license/guideline trước**, lưu bản gốc chỉ đọc + SHA-256, rồi chạy schema audit; chưa fine-tune ngay.
3. Tìm các trường mới ngoài `comment/title/description`, đặc biệt `video_id`, `thread/reply`, `annotator`, `timestamp`, `channel/user`; không đưa vào model trước khi biết có được phép và có nguy cơ leakage hay không.
4. Chốt glossary A/B/C và scorer trước khi tối ưu threshold của C.
5. Thêm hai bộ error slice ngay từ đầu: **reclaimed/quoted/supportive language** và **context-dependent implicit harm**.
6. Với Italian, ưu tiên baseline riêng `XLM-T hoặc encoder tương đương` bên cạnh multilingual baseline; nhưng không coi kết quả HODIAT là performance estimate cho Task 8.
7. Nếu đến sau mốc 18/09 vẫn chưa có đường dẫn thật, dùng email ở Mục 21 và bổ sung câu hỏi về `reclaimed slurs / quotation / counterspeech` cùng quy tắc external translation/LLM.

### 23.11. Câu hỏi mới nên thêm vào email organizer

Ngoài 10 câu ở Mục 12, nên hỏi thêm:

11. Với B, **reclaimed slurs, self-reference, quoted slurs và counterspeech** được gán vào `non-hate/supportive` theo quy tắc nào? Có ví dụ biên chính thức không?
12. Có cung cấp **annotator-level labels, confidence hoặc adjudication notes** không, hay chỉ gold consensus?
13. External machine translation, LLM-based translation, pseudo-labeling và synthetic augmentation có được phép không? Nếu có, có phải khai báo từng provider/model/version không?
14. Title/description là **toàn bộ context được phép**, hay participant được phép truy hồi thêm metadata/video/transcript? Nếu không được, cần ghi rõ để tránh hệ thống vô tình dùng thông tin ngoài benchmark.
15. Nếu cùng một video có nhiều comment, split có được group theo `video_id/thread/user` không? Nếu không có ID công khai, organizer có đảm bảo split đã tránh leakage ở mức nguồn không?

## 24. Nguồn tham khảo

[^1]: SemEval, “[SemEval-2027 Tasks](https://semeval.github.io/SemEval2027/tasks.html)”, danh sách phần thi chính thức; kiểm tra ngày 09/09/2026.

[^2]: StereoQueerEval, “[StereoQueerEval - Identification of Stereotypes and Hate Speech Targeting the LGBTQIA+ Community](https://stereoqueereval.github.io/)”, trang chính thức của phần thi; kiểm tra ngày 09/09/2026.

[^3]: StereoQueerEval, “[Task Description](https://stereoqueereval.github.io/stereoqueereval/)”, trang mô tả phần thi; kiểm tra ngày 09/09/2026.

[^4]: StereoQueerEval, “[StereoQueerEval Pitch.pdf](https://github.com/StereoQueerEval/stereoqueereval.github.io/blob/main/StereoQueerEval%20Pitch.pdf)”, bản đề xuất đã công bố, trang 1–2; kiểm tra ngày 09/09/2026. Đây là bản đề xuất, không thay thế cho hướng dẫn chính thức cuối cùng.

[^5]: StereoQueerEval, “[How to participate](https://stereoqueereval.github.io/how_to_participate/)”, trang hướng dẫn tham gia; kiểm tra ngày 09/09/2026.

[^6]: StereoQueerEval, “[Get the data](https://stereoqueereval.github.io/get_the_data/)”, trang hướng dẫn nhận dữ liệu; kiểm tra ngày 09/09/2026.

[^7]: StereoQueerEval, “[Important Dates](https://stereoqueereval.github.io/important_dates/)”, các mốc thời gian riêng của phần thi; kiểm tra ngày 09/09/2026.

[^8]: StereoQueerEval, “[Organizers and Contacts](https://stereoqueereval.github.io/contact/)”, trang ban tổ chức và liên hệ; kiểm tra ngày 09/09/2026.

[^9]: SemEval, “[SemEval-2027](https://semeval.github.io/SemEval2027/)” và “[Call for SemEval Task Proposals 2027](https://semeval.github.io/SemEval2027/cft.html)”, lịch chung, yêu cầu đối với phần thi và yêu cầu về đạo đức, giấy phép; kiểm tra ngày 09/09/2026.

[^10]: Debora Nozza và cộng sự, “[HODI at EVALITA 2023: Overview of the first Shared Task on Homotransphobia Detection in Italian](https://ceur-ws.org/Vol-3473/paper26.pdf)”, CEUR Workshop Proceedings, 2023.

[^11]: Bharathi Raja Chakravarthi và cộng sự, “[Overview of Third Shared Task on Homophobia and Transphobia Detection in Social Media Comments](https://aclanthology.org/2024.ltedi-1.11/)”, LT-EDI@EACL 2024, ACL Anthology.

[^12]: Prasanna Kumar Kumaresan và cộng sự, “[Overview of Homophobia and Transphobia Span Detection in Social Media Comments](https://aclanthology.org/2025.ltedi-1.36/)”, LT-EDI 2025, ACL Anthology.

[^13]: “[StereoQueerEval GitHub organization](https://github.com/StereoQueerEval/)” và “[website repository](https://github.com/StereoQueerEval/stereoqueereval.github.io)”, các kho mã nguồn công khai được kiểm tra ngày 09/09/2026. Khi kiểm tra chưa thấy kho riêng công khai cho bộ dữ liệu, chương trình chấm điểm hoặc mô hình tham khảo.

[^14]: StereoQueerEval, “[trang chính trong mã website](https://github.com/StereoQueerEval/stereoqueereval.github.io/blob/main/index.html)” và “[trang mô tả phần thi trong mã website](https://github.com/StereoQueerEval/stereoqueereval.github.io/blob/main/stereoqueereval/index.html)”, kiểm tra ngày 09/09/2026. Dùng để đối chiếu nội dung và các dấu vết cũ trên website; mã website không thay thế hướng dẫn chính thức.

[^15]: StereoQueerEval, “[How to participate](https://github.com/StereoQueerEval/stereoqueereval.github.io/blob/main/how_to_participate/index.html)”, tệp nguồn của trang hướng dẫn tham gia, kiểm tra ngày 09/09/2026. Trang vẫn ghi Codabench và hướng dẫn sẽ được bổ sung sau.

[^16]: StereoQueerEval, “[Get the data](https://github.com/StereoQueerEval/stereoqueereval.github.io/blob/main/get_the_data/index.html)”, tệp nguồn của trang nhận dữ liệu, kiểm tra ngày 09/09/2026. Liên kết biểu mẫu hiện là `.`; trang hướng dẫn nhận thông tin qua email và trỏ tới GitHub để lấy chương trình chấm điểm, mô hình tham khảo.

[^17]: StereoQueerEval, “[Important Dates](https://github.com/StereoQueerEval/stereoqueereval.github.io/blob/main/important_dates/index.html)”, tệp nguồn của trang mốc thời gian, kiểm tra ngày 09/09/2026; hiện chỉ ghi dữ liệu huấn luyện có từ ngày 18/09/2026.

[^18]: “[StereoQueerEval GitHub organization](https://github.com/StereoQueerEval/)” và “[website repository](https://github.com/StereoQueerEval/stereoqueereval.github.io)”, danh sách tệp công khai được kiểm tra ngày 09/09/2026.

[^19]: Bharathi Raja Chakravarthi và cộng sự, “[Dataset for Identification of Homophobia and Transophobia in Multilingual YouTube Comments](https://arxiv.org/abs/2109.00227)”, arXiv:2109.00227, 2021. Tóm tắt bài viết mô tả 15.141 bình luận nhiều ngôn ngữ được chuyên gia gán nhãn và một hệ thống nhãn nhiều tầng.

[^20]: Bharathi Raja Chakravarthi và cộng sự, “[Overview of The Shared Task on Homophobia and Transphobia Detection](https://aclanthology.org/2022.ltedi-1.57/)”, LT-EDI@ACL-IJCNLP 2022; dùng để đối chiếu số lượng theo ngôn ngữ, chênh lệch giữa các nhóm, cách thu thập dữ liệu và cách dùng Macro F1.

[^21]: Prasanna Kumar Kumaresan và cộng sự, “[Dataset for Identification of Homophobia and Transphobia for Telugu, Kannada, and Gujarati](https://aclanthology.org/2024.lrec-main.393/)”, LREC-COLING 2024, trang 4404–4411.

[^22]: Debora Nozza và cộng sự, “[HODI at EVALITA 2023: Overview of the first Shared Task on Homotransphobia Detection in Italian](https://aclanthology.org/2023.evalita-1.30/)”, EVALITA 2023; dùng để đối chiếu việc gán nhãn, mức thống nhất, cách lấy mẫu và đánh giá.

[^23]: “[HODI-EVALITA/HODI_2023](https://github.com/HODI-EVALITA/HODI_2023)”, kho chính thức được kiểm tra ngày 09/09/2026; dùng làm nguồn tham khảo cho HODIAT, quyền truy cập và chương trình đánh giá của HODI.

[^24]: Google for Developers, “[YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)”, điều khoản hiện hành; kiểm tra ngày 09/09/2026.

[^25]: Google for Developers, “[Comments: list - YouTube Data API](https://developers.google.com/youtube/v3/docs/comments/list)”, tài liệu được cập nhật lần cuối ngày 04/09/2026 UTC; kiểm tra ngày 09/09/2026.

[^26]: SemEval-2027, “[Call for SemEval Task Proposals 2027](https://semeval.github.io/SemEval2027/cft.html)”, yêu cầu về giấy phép, riêng tư, mức thống nhất khi gán nhãn, công cụ kiểm tra định dạng, chương trình chấm điểm, mô hình tham khảo và các công cụ phục vụ cuộc thi; kiểm tra ngày 09/09/2026.

[^27]: SemEval, “[SemEval-2027 Tasks](https://semeval.github.io/SemEval2027/tasks.html)” và tệp nguồn “[tasks.md](https://github.com/SemEval/SemEval2027/blob/main/tasks.md)”; kiểm tra lại ngày 17/09/2026. Task 8 vẫn được liệt kê với email `stereoqueereval@gmail.com`, nhưng mục Task 8 hiện chỉ có liên kết liên hệ organizer, không hiển thị website/Codabench.

[^28]: SemEval, “[SemEval-2027](https://semeval.github.io/SemEval2027/)”; kiểm tra lại ngày 17/09/2026. Trang được truy cập ở lần kiểm tra này hiển thị `Training data: September 1, 2026`, khác với trạng thái lịch chung được ghi trong lần kiểm tra 09/09 của tài liệu này; do đó cần lưu cả ngày kiểm tra khi viện dẫn lịch.

[^29]: “[StereoQueerEval GitHub organization](https://github.com/StereoQueerEval/)”; kết quả được lập chỉ mục kiểm tra ngày 17/09/2026 cho thấy 1 repository công khai là `stereoqueereval.github.io`, mô tả “Official website of StereoQueerEval”, trong khi website profile là `https://stereoqueereval-semeval2027.github.io/`.

[^30]: Heriot-Watt University, “[All Proposals – Identification of Stereotypes and Hate Speech Targeting the LGBTQIA+ Community](https://projects.hw.ac.uk/Proposals/Index)”; kết quả được lập chỉ mục ngày 17/09/2026. Chỉ dùng như tín hiệu độc lập rằng các nhóm bên ngoài vẫn chuẩn bị cho Task 8; không dùng làm nguồn quy định cuộc thi.

[^31]: Greta Damo, Alessandra Teresa Cignarella, Tommaso Caselli, Viviana Patti và Debora Nozza, “[HODIAT: A Dataset for Detecting Homotransphobic Hate Speech in Italian with Aggressiveness and Target Annotation](https://aclanthology.org/2025.woah-1.11/)”, WOAH 2025, ACL Anthology.

[^32]: Damo và cộng sự, HODIAT, bảng phân bố nhãn: 6.000 tweet (5.000 train + 1.000 test); target và aggressiveness được tính trên các mẫu hateful. Bản PDF trên ACL Anthology: <https://aclanthology.org/2025.woah-1.11.pdf>.

[^33]: Damo và cộng sự, HODIAT, phần Results/Conclusion: fine-tuning nhìn chung mạnh hơn zero/few-shot trong thiết lập của bài; target classification dao động; tác giả báo cáo huấn luyện ba nhãn jointly có xu hướng kém hơn huấn luyện riêng. <https://aclanthology.org/2025.woah-1.11.pdf>.

[^34]: Chiara Ferrando và cộng sự, “[MultiPRIDE at EVALITA 2026: Overview of the Multilingual Automatic Detection of Slur Reclamation in the LGBTQ+ Context Task](https://aclanthology.org/2026.evalita-1.18/)”, EVALITA 2026.

[^35]: CWI, “[MultiPRIDE at EVALITA 2026](https://ir.cwi.nl/pub/36453/)”; trang dự án/bài viết ghi 19 teams, 61 runs cho Task A, 27 runs cho Task B và liên kết code chính thức.

[^36]: Greta Damo và Nicolás Benjamín Ocampo, “[HateItOff at MultiPRIDE: Linguistic and Sentiment Cues in Reclaimed LGBTQ+ Slur Detection](https://aclanthology.org/2026.evalita-1.25/)”, EVALITA 2026.

[^37]: Alessandra Teresa Cignarella và Matteo Pellegrini, “[An Overview of Current Practices and Recommendations for Working with Stereotypes in NLP](https://aclanthology.org/2026.nlperspectives-1.11/)”, NLPerspectives @ LREC 2026; có case study trên một phần dữ liệu QUEEREOTYPES.

[^38]: Alessandra Teresa Cignarella, Anastasia Giachanou và Els Lefever, “[A Survey on Stereotype Detection in Natural Language Processing](https://doi.org/10.1145/3770754)”, ACM Computing Surveys, vol. 58(5), 2026; bản ghi mở tại Ghent University: <https://biblio.ugent.be/publication/01K82YFKHV1422CC0CZDB6663P>.

[^39]: Tommaso Caselli và Flor Miriam Plaza-del-Arco, “[Learning from Disagreement: Entropy-Guided Few-Shot Selection for Toxic Language Detection](https://aclanthology.org/2025.woah-1.5/)”, WOAH 2025.

[^40]: Fai Leui Chan, Duke Nguyen và Aditya Joshi, “[‘Is Hate Lost in Translation?’: Evaluation of Multilingual LGBTQIA+ Hate Speech Detection](https://aclanthology.org/2024.alta-1.11/)”, ALTA 2024.

[^41]: Arslan Bisharat và cộng sự, “[SpectrumNet: Detecting LGBTQ+ Cyberbullying with Dynamic Context-Aware Attention](https://ojs.aaai.org/index.php/ICWSM/article/view/42638)”, ICWSM 2026.

[^42]: Melina Morch và Daniel Braun, “[Evaluating and Mitigating Anti-LGBTQ Biases in German and Multilingual Language Models](https://arxiv.org/abs/2608.30884)”, arXiv:2608.30884, đăng 31/08/2026. Đây là preprint và không phải tài liệu của StereoQueerEval; dùng làm bằng chứng tham khảo về khác biệt giữa stereotype dịch và stereotype thu thập từ cộng đồng.
