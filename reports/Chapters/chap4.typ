= CHƯƠNG 4. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

== 4.1 Tóm tắt kết quả

Đồ án đã hoàn thành mục tiêu nghiên cứu đặt ra: xây dựng và đánh giá so sánh các phương pháp Thị giác Máy tính cho bài toán phân loại trái cây, làm cơ sở cho bước triển khai thực tế tiếp theo.

Thông qua việc xây dựng và so sánh bốn phương pháp từ Machine Learning truyền thống đến Deep Learning, nhóm không chỉ tìm được mô hình hoạt động hiệu quả mà còn hiểu rõ tại sao từng phương pháp hoạt động tốt hay kém trong ngữ cảnh cụ thể này, từ vai trò của màu sắc, giới hạn của đặc trưng thủ công, đến tác động của bọc nilon lên chất lượng phân loại.

Giải pháp đề xuất là mô hình ResNet18 Transfer Learning kết hợp với cơ chế hiển thị danh sách gợi ý Top-K có tính khả thi cao để triển khai thực tế vì:
- Không yêu cầu thay đổi lớn về hạ tầng
- Tích hợp tự nhiên vào quy trình hiện tại của nhân viên thu ngân
- Đạt độ chính xác đủ để thay thế việc tra cứu thủ công trong điều kiện thực tế tại quầy.

== 4.2 Kinh nghiệm rút ra

Qua quá trình thực hiện đồ án, nhóm rút ra một số bài học:

- *Hiểu bài toán trước khi chọn mô hình:*
  Một mô hình tốt không chỉ đến từ kiến trúc phức tạp mà từ việc hiểu đúng đặc thù của bài toán. Khi nắm rõ những thách thức thực tế, điều kiện ánh sáng, cách trái cây được đóng gói, môi trường quầy thu ngân, các quyết định kỹ thuật sẽ có định hướng rõ ràng và hiệu quả hơn.


- *Dữ liệu và cách biểu diễn dữ liệu quan trọng không kém mô hình:*
  Cùng một thuật toán nhưng với cách biểu diễn dữ liệu phù hợp có thể tạo ra sự khác biệt lớn về kết quả. Chiến lược augmentation hợp lý, thiết kế dataset có chủ đích, và lựa chọn đặc trưng phù hợp với bài toán đều là những yếu tố quyết định không kém gì việc chọn kiến trúc mô hình.

- *Thiết kế thực nghiệm trung thực phản ánh đúng thực tế:*
  Đánh giá mô hình trong điều kiện gần với thực tế triển khai thay vì điều kiện lý tưởng có thể giúp kết quả có giá trị ứng dụng thực sự. Một mô hình chỉ hoạt động tốt trong phòng thí nghiệm nhưng thất bại ngoài thực tế không có ý nghĩa.

- *So sánh nhiều mô hình:*

  Thay vì chỉ chọn một mô hình và tối ưu nó, việc so sánh nhiều phương pháp trên cùng một bộ dữ liệu giúp hiểu được tại sao mỗi phương pháp hoạt động tốt hay kém. Kiến thức này có giá trị hơn nhiều so với chỉ biết mô hình nào cho số liệu đẹp nhất.
== 4.3 Hạn chế
Mặc dù đồ án đã rút ra được những kết quả thuận lợi, nhưng nhóm không thể không nhắc đến những hạn chế còn tồn tại:

- *Phạm vi phân loại hẹp:*

  Hệ thống chỉ đang hỗ trợ 5 loại trái cây phổ biến, trong khi thực tế tại các siêu thị và cửa hàng tiện lợi có hàng chục loại trái cây khác nhau. Đây là giới hạn lớn nhất nếu muốn triển khai thực tế diện rộng.
  
- *Điều kiện thu thập dữ liệu còn kiểm soát:*

  Phần ảnh tự thu thập (bagged) chỉ do một người chụp, trong khoảng một tuần, dưới một nguồn sáng đèn phòng duy nhất. Mặc dù điều này đảm bảo tính nhất quán trong quá trình thực nghiệm, nó cũng giới hạn tính đa dạng của dataset, chưa bao quát được các biến thể về thiết bị chụp, phong cách thao tác của nhiều người khác nhau, hoặc sự thay đổi ánh sáng theo thời điểm trong ngày. Hệ quả là hiệu năng thực tế khi triển khai có thể thấp hơn kết quả thực nghiệm.
  
- *Chưa kiểm chứng thực tế:*

  Kết quả đạt được trên tập test trong môi trường kiểm soát. Hệ thống chưa được thử nghiệm với hệ thống camera thực tế gắn tại quầy thu ngân. Sự khác biệt về thiết bị, góc chụp, điều kiện ánh sáng thực tế, hoặc thậm chí nếu nhân viên không mở bọc ra để cho camera thấy được phần lớn trái cây cũng có thể ảnh hưởng nghiêm trọng đến hiệu năng.
== 4.4 Hướng phát triển

Dựa trên những hạn chế đã nhận ra, nhóm đề xuất các hướng phát triển tiếp theo để hoàn thiện hệ thống và tiến gần hơn đến khả năng triển khai thực tế:

- *Mở rộng phạm vi phân loại:*

  Bổ sung thêm nhiều loại trái cây và nông sản phổ biến tại siêu thị Việt Nam, đồng thời thu thập dữ liệu trong nhiều điều kiện thực tế hơn (nhiều loại camera, góc chụp, điều kiện ánh sáng khác nhau) để tăng khả năng tổng quát hóa của mô hình.

- *Tối ưu hóa cho thiết bị thực tế:*

  Thử nghiệm các kiến trúc nhẹ hơn như MobileNet hay EfficientNet để phù hợp với thiết bị tại quầy thu ngân với tài nguyên tính toán hạn chế, giảm chi phí triển khai mà vẫn đảm bảo độ chính xác.

- *Giảm thiểu thao tác thủ công:*

  Tích hợp thêm module phát hiện vật thể (_object detection_) để hệ thống tự động nhận diện khi có trái cây đặt lên cân, giúp giảm bớt sự phụ thuộc vào thao tác ấn nút chụp của nhân viên, từ đó rút ngắn thêm thời gian xử lý tại quầy thu ngân.


#pagebreak()