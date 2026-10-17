= CHƯƠNG 1. TỔNG QUAN
== 1.1 Mô tả bài toán

Ngành bán lẻ hiện đại đang chứng kiến sự tăng trưởng mạnh mẽ của các chuỗi siêu thị và cửa hàng tiện lợi. Tại các quầy thanh toán, phần lớn sản phẩm đóng gói đã có mã vạch để quét nhanh. Tuy nhiên, các mặt hàng trái cây tươi thường được bán theo khối lượng và không có mã vạch riêng.

Với sự phát triển mạnh mẽ của lĩnh vực Thị giác Máy tính (Computer Vision), việc xây dựng một hệ thống tự động phân loại trái cây từ hình ảnh camera trở nên hoàn toàn khả thi. Hệ thống như vậy có thể tích hợp trực tiếp vào bàn cân, giúp tăng tốc quy trình thanh toán, giảm thiểu sai sót, và nâng cao trải nghiệm khách hàng.
== 1.2 Lý do thực hiện

Quy trình hiện tại tại quầy thu ngân còn nhiều hạn chế: khi khách hàng mang trái cây đến quầy, nhân viên phải tự nhìn bằng mắt thường, sau đó tìm kiếm tên hoặc mã sản phẩm trong hệ thống. Quy trình thủ công này gây ra nhiều vấn đề:

- Tốn thời gian: Nhân viên phải tra cứu thủ công, đặc biệt với nhân viên mới chưa quen sản phẩm.
- Ùn tắc: Vào giờ cao điểm, quầy thanh toán dễ bị quá tải do tốc độ xử lý chậm.
- Sai sót: Nhập sai mã sản phẩm dẫn đến tính sai giá, gây thất thoát doanh thu hoặc khiếu nại từ khách hàng.

Những hạn chế trên đặt ra nhu cầu cho một hệ thống phân loại trái cây tự động có thể phân loại chính xác và nhanh chóng ngay tại quầy thu ngân.

Hệ thống được thiết kế để tích hợp tự nhiên vào quy trình hiện tại: 
- Một camera gắn cố định phía trên bàn cân
- Nhân viên đặt trái cây lên cân và ấn nút chụp 
- Hệ thống tự động trả về danh sách loại trái cây có xác suất phân loại cao nhất để nhân viên xác nhận bằng một thao tác. 

So với quy trình thủ công hiện tại (nhìn bằng mắt → tra cứu tên → tìm mã → nhập tay), cách tiếp cận này rút ngắn đáng kể thời gian xử lý và giảm thiểu sai sót nhập liệu.
#pagebreak()
== 1.3 Phát biểu bài toán
=== 1.3.1 Định nghĩa bài toán
- *Input:*
  + Tập dữ liệu gồm có $N$ phần tử đã được gán nhãn: $cal(D) = {(bold(x)_i, y_i)}_(i=1)^N$.
  
    - Tập các nhãn của $cal(D)$ là: $cal(L) = {y_i | i = 1, dots, C}$, (C là số lớp)
    - Mỗi phần tử của $cal(D)$:
      - $bold(x)_i$ là ảnh số, $x_i in bb(R)^(H times W times 3)$
      - $bold(y_i)$ là giá trị nhãn (label) được gán cho $bold(x)_i$

- *Output:*  $f: bb(R)^(H times W times 3) -> cal(L)$\ Sao cho với ảnh đầu vào $bold(x)$, mô hình dự đoán đúng $bold(hat(y) = f(x))$

 #figure(
 grid(
   columns: 5,
     gutter: 8pt,
     image("../Figures/710.jpg", width: 100%),
     image("../Figures/7.jpg", width: 100%),
     image("../Figures/3.jpg", width: 100%),
     image("../Figures/19.jpg", width: 100%),
     image("../Figures/215.jpg", width: 100%),
   ),
   caption: [Ảnh mẫu 5 loại trái cây trong dataset: táo, chuối, xoài, cam, cà chua],
 )

=== 1.3.2 Phạm vi và giả định
==== 1.3.2.1 Phạm vi phân loại

Hệ thống chỉ phân loại *5 loại trái cây* phổ biến tại siêu thị Việt Nam, với định nghĩa cụ thể:

#table(
  columns: (auto, auto, 1fr),
  inset: 8pt,
  stroke: 0.5pt,
  [*Tên lớp*], [*Loại cụ thể*], [*Đặc điểm phân loại*],
  [Táo (apple)], [Táo đỏ], [Hình tròn, vỏ đỏ hoặc đỏ-vàng xen kẽ, bề mặt bóng, có cuống ngắn],
  [Chuối (banana)], [Chuối vàng], [Hình dài cong, vỏ vàng ],
  [Xoài (mango)], [Xoài xanh], [Hình bầu dục, vỏ xanh, bề mặt mịn],
  [Cam (orange)], [Cam ngọt], [Hình tròn, vỏ cam sần nhẹ],
  [Cà chua (tomato)], [Cà chua đỏ], [Hình tròn hoặc hơi dẹt, vỏ đỏ mịn bóng, có cuống/lá xanh],
)

#pagebreak()


==== 1.3.2.2 Điều kiện chụp ảnh

- *Góc chụp:* Camera gắn cố định, chụp từ trên xuống vuông góc với mặt bàn cân. Góc chụp không thay đổi trong suốt quá trình sử dụng.
- *Ánh sáng:* Khu vực cân có ánh sáng đủ và tương đối ổn định (đèn LED cố định). Không xét các trường hợp ánh sáng cực đoan (quá tối, ngược sáng mạnh).
- *Nền:* Có thể là nền trắng (điều kiện lý tưởng) hoặc nền có nhiều vật thể xung quanh (bàn, kệ — mô phỏng quầy thực tế).
- *Khoảng cách:* Camera cách mặt bàn 30–50cm, đảm bảo trái cây chiếm phần lớn khung hình.

==== 1.3.2.3 Giả định
- Mỗi lần cân chỉ có 1 loại mặt hàng trên cân (có thể nhiều quả cùng loại).
- Trái cây ở trạng thái còn tươi, không bị dập nát, thối rữa hoặc biến dạng nghiêm trọng.
- Bề mặt trái cây không bị che khuất hoàn toàn. Nếu có bọc nilon, phần lớn bề mặt vẫn nhìn thấy được qua bọc hoặc sau khi kéo bọc xuống. Bọc nilon có thể trong suốt hoặc đục, nhưng phải là dạng plain (không có hình ảnh, hoạ tiết màu mè).


#pagebreak()