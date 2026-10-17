= CHƯƠNG 3. THỰC NGHIỆM VÀ KẾT QUẢ
== 3.1 Bộ dữ liệu
=== 3.1.1 Tổng quan về bộ dữ liệu
Tập dữ liệu được sử dụng trong đồ án gồm hình ảnh của 5 loại trái cây phổ biến tại các siêu thị Việt Nam:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  stroke: 0.5pt,
  [*STT*], [*Tên lớp*], [*Tên tiếng Việt*],
  [0], [apple], [Táo],
  [1], [banana], [Chuối],
  [2], [mango], [Xoài],
  [3], [orange], [Cam],
  [4], [tomato], [Cà chua],
)

Dữ liệu được thu thập từ hai nguồn:
- Ảnh có sẵn (3.850 ảnh): Lấy từ các dataset công khai trên Kaggle, bao gồm *2.000 ảnh nền trắng (clear)* và *1.850 ảnh có nhiễu nền (noisy background)*.
- Ảnh tự thu thập (*2.150 ảnh*): Nhóm tự chụp trong điều kiện mô phỏng quầy thu ngân thực tế - trái cây được đặt trong bọc nilon (*bagged*), chụp từ góc từ trên xuống với ánh sáng cố định.

Cụ thể, quy trình thu thập ảnh bagged được thực hiện như sau:
- Người chụp: Một thành viên trong nhóm phụ trách toàn bộ quá trình thu thập, đảm bảo tính nhất quán về góc chụp và thao tác đặt trái cây.
- Thời gian: Dữ liệu được thu thập trong khoảng một tuần, mỗi ngày chụp một số lượng ảnh nhất định cho từng loại trái cây.
- Bao bì: Trái cây được đặt trong hai loại bọc nilon phổ biến tại siêu thị Việt Nam, bọc trong suốt và bọc đục mờ, nhằm mô phỏng đa dạng các tình huống tại quầy thu ngân thực tế.
- Điều kiện ánh sáng: Toàn bộ ảnh được chụp dưới ánh sáng đèn phòng cố định, không sử dụng ánh sáng tự nhiên để hạn chế biến động ánh sáng giữa các ngày chụp.
Việc phân chia ba điều kiện ảnh *(_clear, noisy background, bagged_)* - nhằm đánh giá khả năng tổng quát hóa của mô hình từ điều kiện lý tưởng (nền trắng, đối tượng rõ ràng) đến điều kiện thực tế tại quầy thu ngân (nền phức tạp, trái cây bọc nilon).

*Lưu ý: * Ảnh bagged được nhóm tự chụp bằng máy ảnh với độ phân giải cao, kích thước gốc lên đến $approx 3000×3000$ pixel, làm tăng tổng dung lượng của dataset lên đến $approx 1.95 "Gb"$

Để giảm chi phí lưu trữ và tăng tốc độ load dữ liệu, toàn bộ ảnh *_bagged_* được resize về kích thước $512×512$ pixel theo tỉ lệ cố định (fixed proportion) trước khi đưa vào pipeline, đảm bảo không làm biến dạng đối tượng và vẫn đủ độ phân giải để trích xuất đặc trưng và augment xa hơn.
=== 3.1.2 Cấu trúc bộ dữ liệu
==== 3.1.2.1 Phân chia các tập dữ liệu (Train/Val/Test)
Dữ liệu được phân chia thành 3 tập theo tỉ lệ 70/15/15 cho mỗi lớp:

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 8pt,
  stroke: 0.5pt,
  [*Tập*], [*Số ảnh/lớp*], [*Tổng ảnh*], [*Tỉ lệ*], [*Mục đích*],
  [Train], [840], [4.200], [70%], [Huấn luyện mô hình],
  [Validation], [180], [900], [15%], [Tinh chỉnh siêu tham số, theo dõi overfitting],
  [Test], [180], [900], [15%], [Đánh giá hiệu năng cuối cùng],
)

Việc phân chia dữ liệu được cố định trước khi huấn luyện, đảm bảo cả 4 mô hình đều sử dụng cùng một bộ dữ liệu để so sánh công bằng.

==== 3.1.2.2 Đặc điểm của dữ liệu
Tập dữ liệu bao gồm ảnh trong nhiều điều kiện khác nhau:

- *Ảnh nền trắng (clear):* Trái cây trên nền đồng nhất, rõ ràng - chiếm phần lớn tập train, giúp mô hình học đặc trưng cơ bản.
- *Ảnh nhiễu nền (noisy background):* Trái cây trên nền không đồng nhất (bàn, kệ, nhiều vật thể) - mô phỏng quầy thu ngân thực tế. Các vật thể trong ảnh noisy background không phải là trái cây thuộc các lớp khác trong dataset, đảm bảo tuân theo giả định ban đầu
- *Ảnh bọc nilon (bagged):* Trái cây trong bọc nilon trong suốt/bóng - điều kiện khó nhất vì bọc gây phản chiếu ánh sáng, làm mờ đặc trưng hình dạng và màu sắc.

Tổng cộng dataset gồm 6.000 ảnh, được phân bố như :

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, left, left),
  inset: 8pt,
  stroke: 0.5pt,
  [*Tập*], [*Clear*], [*Noisy background*], [*Bagged*],
  [Train (840/lớp)], [400], [220], [220],
  [Validation (180/lớp)], [-], [90], [90],
  [Test (180/lớp)], [-], [60], [120],
)

Ảnh clear chỉ xuất hiện trong tập train vì mục đích của chúng là: 
- Giúp mô hình học các đặc trưng cơ bản của từng loại trái cây trong điều kiện lý tưởng - nền trắng, đối tượng rõ ràng, không bị che khuất. 

Tập validation và test:
- Được thiết kế chỉ gồm ảnh noisy background và bagged, vì đây mới là điều kiện thực tế tại quầy thu ngân. Việc đánh giá mô hình trên hai điều kiện khó này đảm bảo kết quả phản ánh đúng hiệu năng trong môi trường triển khai thực tế.

==== 3.1.2.1 Phân bố nhãn

Dữ liệu được phân bố *cân bằng hoàn toàn* giữa các lớp (mỗi lớp 1.200 ảnh). 

Sự cân bằng này được duy trì trong cả ba tập:
- 840 ảnh/lớp trong train
- 180 ảnh/lớp trong validation
- 180 ảnh/lớp trong test

Ngoài ra, mỗi lớp cũng có phân bố tương đồng về số lượng ảnh (clear, noisy background, bagged), đảm bảo mô hình không bị thiên lệch theo bất kỳ lớp nào.
== 3.2 Độ đo
Để đánh giá hiệu năng của các mô hình một cách toàn diện, nhóm sử dụng các độ đo sau. Với dataset cân bằng hoàn toàn (180 ảnh/lớp), Macro Average và Weighted Average cho kết quả tương đương nhau.

=== 3.2.1 Accuracy

Tỉ lệ dự đoán đúng trên tổng số mẫu:

$ "Accuracy" = frac("Số dự đoán đúng", "Tổng số mẫu") = frac(sum_(c=1)^C "TP"_c, N) $

Accuracy là độ đo tổng quan nhất — phù hợp khi dataset cân bằng như bộ dữ liệu của nhóm (mỗi lớp 180 ảnh test).
=== 3.2.2 Precision

Trong các mẫu được dự đoán là lớp $c$, bao nhiêu phần trăm thực sự thuộc lớp $c$:

$ "Precision"_c = frac("TP"_c, "TP"_c + "FP"_c) $

Ý nghĩa thực tế: khi hệ thống nói "đây là táo", đúng bao nhiêu phần trăm, nếu sai thì nhân viên tính sai giá cho khách hàng.
=== 3.2.3 Recall

Trong tất cả mẫu thực sự thuộc lớp $c$, mô hình nhận diện được bao nhiêu phần trăm:

$ "Recall"_c = frac("TP"_c, "TP"_c + "FN"_c) $

*Ý nghĩa thực tế:* Trong tất cả ảnh táo thật, model nhận ra được bao nhiêu %, nếu bỏ sót thì nhân viên phải nhập tay.
=== 3.2.4 F1-score
Trung bình điều hòa của Precision và Recall — cân bằng cả hai:

$ "F1"_c = 2 dot frac("Precision"_c dot "Recall"_c, "Precision"_c + "Recall"_c) $

F1-score là metric tổng hợp nhất để so sánh các mô hình. Giá trị 1 là tốt nhất, 0 là tệ nhất.

=== 3.2.5 Macro Average và Weighted Average

Với bài toán nhiều lớp, các metric trên được tổng hợp theo hai cách:

- *Macro Average:* Tính trung bình cộng của metric từng lớp — coi tất cả lớp quan trọng như nhau.
- *Weighted Average:* Tính trung bình có trọng số theo số mẫu mỗi lớp — phù hợp khi dữ liệu không cân bằng.


Vì dataset của nhóm cân bằng hoàn toàn (mỗi lớp 180 ảnh test):
- Macro Average $approx$ Weighted Average, hai cách tổng hợp này cho kết quả gần như giống nhau. 
Nhóm sử sẽ dụng Macro Average làm độ đo chính để so sánh các mô hình.
=== 3.2.6 Confusion Matrix

Confusion Matrix là bảng kích thước $C times C$ (với $C$ là số lớp) tổng hợp kết quả dự đoán so với nhãn thực tế. 

Với bài toán 5 lớp, ma trận có kích thước $5 times 5$, trong đó:
- Phần tử $M_(i,j)$ là số mẫu thuộc lớp $i$ nhưng bị dự đoán thành lớp $j$.
- Đường chéo chính thể hiện số dự đoán đúng.

Từ Confusion Matrix, ta định nghĩa cho mỗi lớp $c$:
- *True Positive (TP):* Số mẫu thuộc lớp $c$ và được dự đoán đúng là lớp $c$.
- *False Positive (FP):* Số mẫu không thuộc lớp $c$ nhưng bị dự đoán nhầm là lớp $c$.
- *False Negative (FN):* Số mẫu thuộc lớp $c$ nhưng bị dự đoán nhầm thành lớp khác.
- *True Negative (TN):* Số mẫu không thuộc lớp $c$ và được dự đoán đúng là không thuộc lớp $c$.

*Confusion Matrix* giúp phát hiện các cặp nhầm lẫn cụ thể:
- Apple bị nhầm thành Tomato bao nhiêu lần: thông tin này không thể thấy được từ `Accuracy` hay `F1-score` đơn thuần.

== 3.3 Kết quả thực nghiệm
=== 3.3.1 Tổng quan về các kết quả

Tất cả 4 mô hình được đánh giá trên cùng một tập test (900 ảnh, 180 ảnh/lớp). 

Bảng dưới tổng hợp kết quả:


#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    [*Mô hình*], [*Accuracy*], [*Precision (macro)*], [*Recall (macro)*], [*F1 - score (Macro)*],

    [KNN (hog_hsv)], [56.44%], [57.31%], [56.44%], [56.46%],
    [SVM (hog-only)], [60.00%], [60.00%], [60.00%], [59.00%],
    [SVM (hog-color)], [83.00%], [83.00%], [83.00%], [83.00%],
    [CNN], [87.00%], [88.20%], [87.00%], [86.58%],
    [ResNet18], [99.00%], [99.00%], [99.00%], [99.00%],
  ),
  caption: [Bảng so sánh kết quả các mô hình huấn luyện],
)

Có sự tiến bộ rõ rệt theo trục tăng dần độ phức tạp:
- ML truyền thống *(KNN ~56%, SVM ~60-83%)*
- Deep Learning *(CNN ~87%, ResNet18 ~99%)*.

==== 3.3.1.1 KNN
Kết quả chi tiết của 5 lớp khi sử dụng mô hình KNN:

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header (Tiêu đề cột)
    [*Lớp*], [*Precision*], [*Recall*], [*F1-score*],

    // Nội dung bảng dựa trên image_50daba.png
    [apple],  [0.69], [0.52], [0.59],
    [banana], [0.63], [0.66], [0.65],
    [mango],  [0.49], [0.59], [0.53],
    [orange], [0.53], [0.47], [0.49],
    [tomato], [0.53], [0.59], [0.56],
  ),
  caption: [Báo cáo chi tiết kết quả phân loại theo từng lớp của mô hình KNN],
)

- Banana đạt kết quả tốt nhất (`F1=0.65`) nhờ hình dạng dài cong đặc trưng dễ nhận diện. 
- Orange là lớp yếu nhất (`F1=0.49`): chỉ 84/180 ảnh đúng, bị nhầm nhiều sang mango (38 ảnh) do tông màu tương tự khi bọc nilon.
- Apple có precision cao (`0.69`) nhưng recall thấp (`0.52`), khi KNN nói "đây là táo" thì khá chính xác, nhưng bỏ sót nhiều ảnh táo thật.
==== 3.3.1.2 SVM

Nhóm thực hiện hai thử nghiệm để đánh giá vai trò của màu sắc:


#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header
    [*Lớp*], [*Precision*], [*Recall*], [*F1-score*],

    // Nội dung từ image_50cbd3.png
    [apple],  [0.61], [0.54], [0.57],
    [banana], [0.66], [0.70], [0.68],
    [mango],  [0.54], [0.58], [0.56],
    [orange], [0.55], [0.51], [0.53],
    [tomato], [0.62], [0.64], [0.63],
  ),
  caption: [Báo cáo chi tiết của mô hình SVM chỉ sử dụng đặc trưng HOG],
)

Chỉ dùng HOG:
- Banana vẫn là lớp tốt nhất (`F1=0.68`) nhờ hình dạng đặc trưng.
- Orange là lớp yếu nhất (`F1=0.53`): thông tin hình dạng không đủ để phân biệt các loại trái cây tròn
- Tomato bị nhầm nhiều: precision thấp hơn recall cho thấy nhiều ảnh lớp khác bị dự đoán nhầm thành tomato.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header
    [*Lớp*], [*Precision*], [*Recall*], [*F1-score*],

    // Nội dung từ image_50c432.png
    [apple],  [0.91], [0.75], [0.82],
    [banana], [0.80], [0.91], [0.85],
    [mango],  [0.83], [0.84], [0.84],
    [orange], [0.84], [0.84], [0.84],
    [tomato], [0.81], [0.81], [0.81],
  ),
  caption: [Báo cáo chi tiết của mô hình SVM sử dụng đặc trưng HOG + Color],
)

Thêm Color Histogram giúp accuracy tăng 23 điểm (60%→83%): 
- Cải thiện đồng đều trên tất cả các lớp. 
- Orange tăng mạnh nhất từ` F1=0.53` lên `0.84`: Color Histogram giúp phân biệt tông màu cam với các lớp khác hiệu quả hơn nhiều so với chỉ dùng hình dạng. 
- Apple có precision cao nhất (`0.91`) nhưng recall thấp nhất (`0.75`) 

Khi SVM dự đoán "táo" thì hầu như đúng, nhưng vẫn bỏ sót một phần đáng kể ảnh táo thật. Đây là lớp duy nhất còn precision và recall chênh lệch rõ rệt sau khi thêm Color Histogram.
==== 3.3.1.3 CNN
Kết quả chi tiết của 5 lớp khi sử dụng mô hình CNN:



#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header
    [*Lớp*], [*Precision*], [*Recall*], [*F1-score*],

    // Nội dung từ image_50717d.png
    [apple],  [0.92], [0.59], [0.72],
    [banana], [0.91], [0.97], [0.94],
    [mango],  [0.93], [0.91], [0.92],
    [orange], [0.94], [0.96], [0.95],
    [tomato], [0.72], [0.92], [0.81],
  ),
  caption: [Báo cáo chi tiết của mô hình CNN],
)

CNN đạt `87%` accuracy:
- Vượt SVM (hog-color) dù chỉ dùng kiến trúc đơn giản, chứng tỏ tự học đặc trưng hiệu quả hơn đặc trưng thủ công. 
- Tuy nhiên apple vẫn là lớp yếu nhất — recall chỉ `0.59`, bị nhầm sang tomato 64 ảnh (cặp nhầm lớn nhất toàn bộ thực nghiệm). 
- Tomato có precision thấp (`0.72`) vì nhận nhiều ảnh apple nhầm vào. Banana, orange, mango đều đạt` F1 > 0.90`.
==== 3.3.1.4 ResNet18
Kết quả chi tiết của 5 lớp khi sử dụng mô hình ResNet18:

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header
    [*Lớp*], [*Precision*], [*Recall*], [*F1-score*],

    // Nội dung từ image_506297.png
    [apple],  [0.98], [0.98], [0.98],
    [banana], [0.99], [0.98], [0.99],
    [mango],  [0.96], [0.99], [0.97],
    [orange], [1.00], [0.99], [1.00],
    [tomato], [1.00], [0.99], [0.99],
  ),
  caption: [Báo cáo chi tiết của mô hình ResNet18],
)

ResNet18 đạt `99%` accuracy:
- Gần như đúng trên tất cả các lớp
- Orange và tomato đạt precision = `1.00`, không có bất kỳ dự đoán sai nào
- Cặp nhầm apple $<->$ tomato: vốn là điểm yếu nghiêm trọng của CNN đã được giải quyết phần lớn.
- Mango có F1 thấp nhất (`0.97`) nhưng vẫn cao hơn CNN (`0.92`)
=== 3.3.2 Nhận xét về kết quả
==== 3.3.2.1 KNN

#figure(
  image("/Figures/cm_knn.png", width: 70%),
  caption: [
    Ma trận nhầm lẫn của mô hình KNN
  ]
)

Nhìn vào confusion matrix:
- Orange là lớp yếu nhất — chỉ `84/180` ảnh được phân loại đúng. Đây cũng là lớp bị phân tán sang nhiều lớp khác nhất, đặc biệt sang mango (`38 ảnh`) và tomato (`30 ảnh`).
- Mango nhận nhiều ảnh nhầm từ tất cả các lớp khác, do màu xanh của xoài dễ bị nhầm với nhiều màu khác trong điều kiện bọc nilon.
- Apple có precision cao `(0.69)` nhưng recall thấp `(0.52)`: khi KNN dự đoán "táo" thì khá chính xác, nhưng bỏ sót nhiều ảnh táo thật do nhầm sang mango `(27 ảnh)` và tomato `(23 ảnh)`. 
- Banana là lớp tốt nhất (119/180) nhờ hình dạng dài cong đặc trưng dễ phân biệt với các loại trái cây tròn.
Nhìn chung, KNN phân tán lỗi khá đều giữa các lớp — không có cặp nhầm lẫn nào quá nổi bật, phản ánh đúng bản chất của thuật toán: khoảng cách trong không gian đặc trưng nhiều chiều bị loãng, mọi lớp đều bị ảnh hưởng tương đương.
==== 3.3.2.2 SVM

#figure(
  image("/Figures/hog-only.png", width: 80%),
  caption: [
    Ma trận nhầm lẫn của mô hình SVM chỉ sử dụng HOG
  ]
)

Với SVM chỉ sử dụng đặc trưng HOG:
- Orange tiếp tục là lớp yếu nhất `(92/180)`: bị nhầm nhiều sang mango `(42 ảnh)`, đây là cặp nhầm lớn nhất của cấu hình này. 
- Tomato bị nhầm sang apple `(28 ảnh`): cả hai đều có hình tròn, HOG không phân biệt được khi bọc nilon che khuất cuống/lá xanh của cà chua. 
- Banana vẫn là lớp tốt nhất `(126/180)`.
#figure(
  image("/Figures/hog-color.png", width: 80%),
  caption: [Ma trận nhầm lẫn của mô hình SVM sử dụng HOG + Color]
)

Thêm Color Histogram tạo ra sự cải thiện rõ rệt:
- Đường chéo chính đậm hơn, các ô ngoài đường chéo trở nên rất nhạt. 
- Orange tăng từ `92` lên `152/180`: Color Histogram phân biệt được tông màu cam với xanh lá của mango hiệu quả.
- Banana có độ chính xác cao `(164/180)`, với `0` ảnh nhầm sang apple.

*Điểm đáng chú ý:* 
- Apple vẫn bị nhầm sang tomato `(20 ảnh)`
- Mango bị nhầm sang banana `(25 ảnh)`
Hai cặp nhầm dai dẳng qua cả hai cấu hình, cho thấy đây là giới hạn của đặc trưng thủ công khi xử lý ảnh bọc nilon.

Nguyên nhân là một số ảnh xoài chín có vỏ ngả vàng, khiến Color Histogram của xoài trùng với vùng vàng đặc trưng của chuối; trong khi HOG hình bầu dục của xoài và HOG hình dài cong của chuối lại không đủ khác biệt sau khi bị bọc nilon làm mờ đường viền.
==== 3.3.2.3 CNN

#figure(
  image("/Figures/cnn_learningcur.png"),
  caption: [
     Learning Curve CNN
  ]
)

Learning curve cho thấy quá trình huấn luyện CNN diễn ra theo hai giai đoạn rõ rệt:

- Giai đoạn đầu (epoch 0-4):
  - `val_loss` giảm nhanh từ `1.5` xuống `0.49`
  - `val_acc` tăng vọt từ `0.31` lên `0.80`, mô hình học nhanh các đặc trưng cơ bản. 
  - `train_acc` cũng tăng nhanh lên khoảng `0.87`, khoảng cách giữa train_acc và val_acc còn khá lớn (~0.07) cho thấy mô hình đang trong giai đoạn học tích cực.
  
- Giai đoạn giữa (epoch 4-6): 
  - `val_loss` tăng trở lại lên `0.62`
  - `val_acc` dao động quanh `0.79-0.80`
  - Trong khi `train_acc` vẫn tiếp tục tăng,khoảng cách giữa train và val ngày càng rõ hơn, dấu hiệu overfitting nhẹ. 
  - ReduceLROnPlateau kích hoạt, giảm learning rate từ $1e-3$ xuống $5e-4$.
- Giai đoạn cuối (epoch 6-9): 
  - learning rate nhỏ hơn giúp `val_loss` giảm mạnh xuống `0.31` và `val_acc` tăng lên `0.89`, mô hình hội tụ tốt hơn. 
  - train_acc tiếp tục tăng lên `~0.93`, khoảng cách với `val_acc` thu hẹp lại còn khoảng `0.04`. 
  - EarlyStopping không kích hoạt, mô hình chạy đủ 10 epoch và lưu checkpoint tại epoch cuối với val_acc cao nhất `(0.887)`.

#figure(
  image("/Figures/cm_cnn.png"),
  caption: [Ma trận nhầm lẫn của mô hình CNN]
)

Confusion matrix CNN cho thấy một điểm bất thường rõ rệt: apple chỉ đạt `106/180` (`recall=0.59`) trong khi các lớp khác đều trên `90%`. 

Nguyên nhân là `64` ảnh apple bị nhầm sang tomato, đây là cặp nhầm lớn nhất trong toàn bộ thực nghiệm. Cả hai đều có hình tròn và tông màu đỏ, khi bọc nilon che khuất cuống và texture vỏ thì CNN không đủ đặc trưng để phân biệt. 

Tomato cũng bị ảnh hưởng theo: precision chỉ đạt `0.72` vì nhận nhiều ảnh apple nhầm vào. 

Các lớp còn lại (banana, orange, mango) đều đạt `F1 > 0.90`, cho thấy CNN học tốt với các loại trái cây có màu sắc và hình dạng đặc trưng rõ ràng.
==== 3.3.2.4 ResNet18
#figure(
  image("/Figures/learningcur_resnet.png"),
  caption: [Learning Curve ResNet18]
)
Learning curve ResNet18 có đặc điểm rất khác biệt so với CNN, phản ánh rõ lợi thế của Transfer Learning:

- Giai đoạn hội tụ nhanh (epoch 1-2): 
  - Train loss giảm mạnh từ `0.22` xuống `0.03`
  - Val loss giảm từ `0.07` xuống `0.03`. 
  - Train accuracy tăng từ `0.94` lên `0.99`
  - Val accuracy từ `0.983` lên `0.991` 
  - Chỉ sau 2 epoch, mô hình đã đạt hiệu năng rất cao nhờ pretrained weights.
  Đây là điểm khác biệt lớn nhất so với CNN, CNN cần 4-5 epoch mới đạt `val_acc = 0.80`, trong khi ResNet18 đạt `0.99` chỉ sau 2 epoch.
  
- Giai đoạn dao động (epoch 2-12): 
  - Train loss tiếp tục giảm về gần `0`, 
  - Val loss dao động nhẹ quanh `0.01-0.09`, không có xu hướng tăng hay giảm rõ ràng. 
  - Val acc dao động trong khoảng `0.967-0.997`, đạt đỉnh `0.9978` ở `epoch 10`. 
  - Khoảng cách giữa train acc `(~0.999)` và val acc `(~0.992)` rất nhỏ
  Transfer Learning giúp mô hình tổng quát hóa tốt, không bị overfitting nghiêm trọng.
  
- Giai đoạn overfitting nhẹ (epoch 12-17): 
  - Val loss bắt đầu tăng dần từ `0.014` lên `0.06` 
  - Trong khi train loss vẫn tiếp tục giảm về gần `0`, khoảng cách giữa train và val loss ngày càng rõ hơn.
  - Val acc cũng giảm nhẹ từ `0.994` xuống `0.985`. 
  - EarlyStopping kích hoạt ở epoch 17, dừng huấn luyện sớm. 
  Model được load lại từ checkpoint epoch 12: epoch có val_loss thấp nhất `(0.0137)`, không phải epoch 10 có val_acc cao nhất` (0.9978)`.


#figure(
  image("/Figures/resnet_cm.png", width: 70%),
  caption: [Ma trận nhầm lẫn của mô hình ResNet18]
)

Confusion matrix ResNet18 gần như là ma trận đơn vị, tức là mô hình dự đoán trên tập `test` hầu như đúng hoàn toàn. 

Tổng cộng chỉ có `12` ảnh bị phân loại sai trên 900 ảnh test. 

Các lỗi còn lại rất nhỏ và tập trung chủ yếu vào cặp apple $<->$ mango (5 ảnh, chủ yếu là apple bị nhầm sang mango) và tomato $->$ apple (2 ảnh), cũng là cặp có đặc trưng màu sắc gần nhau nhất trong dataset.

Orange và tomato đạt `precision = 1.00`, không có bất kỳ dự đoán sai nào. 

Đặc biệt, cặp apple $<->$ tomato vốn gây ra 70 lỗi cho CNN được giải quyết gần như hoàn toàn, chỉ còn 2 ảnh tomato bị nhầm sang apple.


== 3.4 Phân tích tổng hợp

=== 3.4.1 ML truyền thống vs Deep Learning

Kết quả thực nghiệm cho thấy sự khác biệt rõ rệt giữa hai nhóm phương pháp:

#figure(
  table(
    columns: (auto, 1fr, auto),
    inset: 10pt,
    align: (left, left, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header
    [*Nhóm*], [*Mô hình*], [*Accuracy*],

    // Nội dung từ image_505a98.png
    [ML truyền thống], [KNN], [56.44%],
    [ML truyền thống], [SVM (hog-only)], [60.00%],
    [ML truyền thống], [SVM (hog-color)], [83.00%],
    [Deep Learning],   [CNN], [87.00%],
    [Deep Learning],   [ResNet18], [*99.00%*],
  ),
  caption: [So sánh Accuracy giữa các nhóm mô hình],
)

*Đặc trưng thủ công vs tự học:*

- CNN (87%) vượt SVM hog-color (83%) dù chỉ dùng kiến trúc đơn giản. Chứng tỏ việc tự học đặc trưng từ pixel hiệu quả hơn đặc trưng do con người thiết kế. 
- Các lớp tích chập của CNN có thể học đồng thời cả thông tin hình dạng lẫn màu sắc, trong khi SVM phải trích xuất HOG và Color Histogram riêng lẻ rồi ghép lại.

*Vai trò của màu sắc:*

- Thí nghiệm SVM hog-only vs hog-color cho thấy rõ nhất vai trò của màu sắc: chỉ thêm Color Histogram đã tăng `23` điểm accuracy `(60%→83%)`. 
- Điều này khẳng định màu sắc là đặc trưng phân biệt quan trọng nhất trong bài toán phân loại trái cây, phù hợp với thực tế: mỗi loại trái cây có tông màu đặc trưng riêng.


=== 3.4.2 Phân tích lỗi

*Cặp nhầm lẫn dai dẳng: Apple $<->$ Tomato*

Đây là cặp nhầm lẫn xuất hiện xuyên suốt tất cả các mô hình, với mức độ giảm dần theo độ phức tạp:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    inset: 10pt,
    align: (left, center, center),
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black) 
    } else {
      (bottom: 0.3pt + gray)
    },
    
    // Header
    [*Mô hình*], [*Apple $arrow$ Tomato*], [*Tomato $arrow$ Apple*],

    // Nội dung từ image_4ea000.png
    [KNN],             [23 ảnh], [12 ảnh],
    [SVM (hog-only)],  [23 ảnh], [*28 ảnh*],
    [SVM (hog-color)], [20 ảnh], [12 ảnh],
    [CNN],             [*64 ảnh*], [6 ảnh],
    [ResNet18],        [0 ảnh],  [2 ảnh],
  ),
  caption: [Thống kê số lượng ảnh bị phân loại nhầm giữa Apple và Tomato],
)

Nguyên nhân gốc rễ là do táo và cà chua có ba đặc điểm giống nhau:
- Hình tròn
- Tông màu đỏ
- Kích thước tương tự. 
Đặc trưng phân biệt duy nhất là cuống ngắn của táo và cuống/lá xanh của cà chua. Nhưng khi bọc nilon che khuất hoặc ánh sáng phản chiếu làm mờ, cả hai đặc trưng này đều biến mất. 

KNN và SVM không đủ khả năng học các đặc trưng tinh vi này từ HOG và Color Histogram. CNN học được một phần nhưng vẫn nhầm nhiều (64 ảnh), cho thấy kiến trúc đơn giản với 10 epoch chưa đủ để nắm bắt đặc trưng phân biệt tinh vi. 

Chỉ ResNet18 với pretrained weights từ ImageNet mới giải quyết được gần như hoàn toàn vì pretrained weights đã học các đặc trưng texture và hình dạng rất chi tiết từ 1.4 triệu ảnh, giúp phân biệt được ngay cả khi bọc nilon.

*Cặp nhầm lẫn thứ hai: Mango $<->$ Banana:*

Cặp này xuất hiện chủ yếu ở:
- CNN (`14` ảnh mango $->$ banana)
- SVM hog-color (`25` ảnh mango$->$banana). 

Nguyên nhân: xoài trong dataset chủ yếu có vỏ xanh, nhưng một số ảnh xoài chín có tông vàng dễ nhầm với chuối vàng. Màu vàng là đặc trưng chủ đạo của chuối, khi xoài chín chuyển vàng thì Color Histogram của hai loại trở nên giống nhau hơn. 

ResNet18 giải quyết tốt cặp này (chỉ `3` ảnh bị nhầm banana $->$ mango) nhờ học được sự khác biệt về texture và hình dạng giữa hai loại.

*Orange (Cam) — lớp khó với ML truyền thống:*

Orange là lớp yếu nhất:
- KNN (84/180 dự đoán đúng)
- SVM hog-only (92/180 dự đoán đúng)
Và bị nhầm nhiều sang mango. 

Nguyên nhân: HOG chỉ mô tả hình dạng, orange và mango đều có hình tròn/bầu dục tương tự, không đủ đặc trưng để phân biệt khi chỉ dựa vào gradient. 

Thực ra hai loại này có màu sắc rất khác nhau: cam tập trung ở vùng đỏ-cam trong khi mango tập trung ở vùng xanh lá. 

Chính vì vậy khi thêm Color Histogram, SVM cải thiện mạnh nhất ở lớp này ($92 -> 152 \/ 180$ dự đoán đúng), nên nhận thấy rằng màu sắc là đặc trưng phân biệt hiệu quả nhất cho các trái cây có màu sắc đặc trưng khác nhau rõ rệt.
=== 3.4.3 Các phát hiện quan trọng
Qua quá trình thực nghiệm, nhóm rút ra 3 phát hiện quan trọng:

1. *Hệ thống cho thấy tiềm năng triển khai thực tế:* 

  ResNet18 đạt 99% accuracy trên tập test với 67% ảnh bọc nilon, điều kiện mô phỏng sát nhất với thực tế quầy thu ngân. Điều này cho thấy tiềm năng có thể hỗ trợ nhân viên thu ngân; tuy nhiên, cần kiểm chứng thêm trong môi trường triển khai thực tế.

2. *Màu sắc là yếu tố phân loại quan trọng nhất, camera màu là yêu cầu bắt buộc:* 

  Kết quả cho thấy thông tin màu sắc đóng vai trò quyết định trong phân loại trái cây. Khi triển khai thực tế, camera gắn trên bàn cân phải là camera màu chất lượng tốt, camera đen trắng hoặc chất lượng thấp sẽ làm giảm hiệu năng đáng kể.

3. *Bọc nilon là điều kiện khó nhất, cần hướng dẫn nhân viên sử dụng đúng cách:*  

  Khi bọc nilon che khuất hoàn toàn cuống và lá xanh, ngay cả ResNet18 vẫn có thể nhầm. Hệ thống Top-K (hiển thị danh sách gợi ý thay vì một kết quả duy nhất) giúp nhân viên dễ dàng xác nhận đúng loại trái cây trong các trường hợp khó này.

#pagebreak()
