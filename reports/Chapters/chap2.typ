
= CHƯƠNG 2. PHƯƠNG PHÁP THỰC NGHIỆM

Để đánh giá hiệu quả của các hướng tiếp cận khác nhau cho bài toán phân loại trái cây, nhóm xây dựng và so sánh bốn mô hình theo trục tăng dần độ phức tạp: 

#table(
  columns: (0.5fr, 0.8fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt,
  [*Mô hình*], [*Vai trò*],[*Đặc điểm*],
  [KNN], [Baseline ML cơ bản nhất],[Phân loại dựa trên khoảng cách; không huấn luyện, lưu toàn bộ tập train để tra cứu khi dự đoán],
  [SVM], [Đại diện ML truyền thống],[Tìm siêu phẳng phân tách tối ưu; yêu cầu trích đặc trưng thủ công (HOG, Color Histogram)],
  [CNN], [Baseline Deep Learning],[Tự học đặc trưng từ pixel qua các lớp tích chập; huấn luyện từ đầu trên tập dữ liệu của nhóm],
  [ResNet18], [Mô hình chính],[Tự động học đặc trưng với kiến trúc sâu; tận dụng trọng số pretrained từ ImageNet],
)
Cách tiếp cận bốn mô hình theo tăng dần độ phức tạp cho phép kiểm tra từng giả thuyết độc lập: 
- Liệu đặc trưng thủ công có đủ để phân loại trái cây? 
- Deep Learning có thực sự vượt trội?
- Transfer Learning mang lại lợi ích bao nhiêu khi dataset nhỏ?
- Từ đó rút ra kết luận có cơ sở về mô hình phù hợp nhất để triển khai.
== 2.1 Tiền xử lí dữ liệu
Trước khi đưa dữ liệu vào huấn luyện, ảnh cần được tiền xử lý để đảm bảo tính nhất quán và tối ưu hiệu năng. Pipeline tiền xử lý bao gồm 4 bước chính:

=== 2.1.1 Resize ảnh <resize>
Ảnh gốc trong dataset có thể có kích thước khác nhau. Để đảm bảo đầu vào đồng nhất cho mô hình, tất cả ảnh được resize về kích thước cố định:
#v(2cm)
#table(
  columns: (auto, auto, auto), 
  inset: 8pt,
  stroke: 0.5pt,
  [*Mô hình*], [*Kích thước*], [*Lý do*],
  [CNN], [224 × 224], [Theo chuẩn ImageNet - đảm bảo nhất quán với ResNet18 để so sánh công bằng],
  [ResNet18], [224 × 224], [Theo chuẩn ImageNet - kích thước pretrained weights được huấn luyện trên],
  [KNN], [224 × 224], [Nhất quán với CNN và ResNet18 - đảm bảo vector đặc trưng HOG được trích xuất từ cùng độ phân giải],
  [SVM], [128 × 128], [Giảm chiều vector HOG để tiết kiệm bộ nhớ và thời gian tính toán — 128×128 vẫn đủ độ phân giải để trích xuất đặc trưng hình dạng và màu sắc],
)
=== 2.1.2 Chuẩn hóa giá trị pixel (Normalization)
Giá trị pixel gốc nằm trong khoảng \[0, 255\]. Chuẩn hóa giúp mô hình hội tụ nhanh hơn và gradient ổn định hơn. Nếu không chuẩn hóa, các giá trị pixel lớn (0–255) sẽ khiến gradient có độ lớn không đồng đều giữa các lớp, dẫn đến quá trình huấn luyện không ổn định.

*Đối với CNN, KNN, SVM:* <normalization>
- Scale giá trị pixel từ \[0–255\] về \[0, 1\] bằng phép chia cho 255.
$ x' = x/255 $

*Đối với ResNet18 (Transfer Learning):*
- Sử dụng chuẩn hóa theo mean/std của ImageNet để tận dụng trọng số pretrained:
  - $mu$ = \[0.485, 0.456, 0.406\]
  - $sigma$ = \[0.229, 0.224, 0.225\]

  $ x' = (x \/ 255 -mu)/sigma $
=== 2.1.3 Data Augmentation <augmentation>

Data Augmentation được áp dụng *chỉ trên tập train* nhằm tăng tính đa dạng và chống overfitting. Mỗi epoch, cùng một ảnh sẽ được biến đổi ngẫu nhiên khác nhau.

Các kỹ thuật augmentation được sử dụng:

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  stroke: 0.5pt,
  [*Kỹ thuật*], [*Mô tả*], [*Mô hình áp dụng*],
  [RandomHorizontalFlip], [Lật ảnh ngang - trái cây lật ngang vẫn giữ đặc trưng], [CNN, ResNet18],
  [RandomRotation(15$degree$/30$degree$)], [Xoay ảnh - mô phỏng trái cây đặt nghiêng trên cân], [CNN, ResNet18\ (CNN: ±30$degree$,\ ResNet18: ±15$degree$ )],
  [ColorJitter], [Điều chỉnh brightness/contrast - mô phỏng điều kiện ánh sáng khác nhau], [ResNet18],
  [Dropout (0.25, 0.5)], [Tắt ngẫu nhiên neuron khi train, buộc model học tổng quát hơn], [CNN],
)
=== 2.1.4 Label Encoding <augmentation>
Tên lớp được ánh xạ thành số nguyên, thống nhất giữa cả 4 mô hình theo thứ tự alphabetical:

#table(
  columns: (auto, auto),
  inset: 8pt,
  stroke: 0.5pt,
  [*Tên lớp*], [*Mã số*],
  [apple], [0],
  [banana], [1],
  [mango], [2],
  [orange], [3],
  [tomato], [4],
)

Việc thống nhất label encoding giữa các mô hình là bắt buộc để đảm bảo so sánh kết quả chính xác, đặc biệt khi phân tích Confusion Matrix.


*Lưu ý*: Tiền xử lý là các phép biến đổi trong bộ nhớ RAM, không thay đổi file ảnh gốc trên ổ đĩa. Mỗi epoch, ảnh được biến đổi ngẫu nhiên khác nhau (đối với augmentation).

== 2.2 Trích xuất đặc trưng

Trích xuất đặc trưng thủ công là bước bắt buộc đối với các mô hình Machine Learning truyền thống (KNN và SVM) vì các thuật toán này không thể xử lý trực tiếp dữ liệu ảnh thô mà cần một vector đặc trưng số học làm đầu vào. Nhóm sử dụng hai loại đặc trưng: Color Histogram mô tả phân bố màu sắc và HOG mô tả hình dạng của trái cây.

Ngược lại, CNN và ResNet18 tự động học đặc trưng trực tiếp từ pixel ảnh thông qua các lớp tích chập, không cần bước trích xuất thủ công. Đây là một trong những ưu điểm cốt lõi của Deep Learning so với Machine Learning truyền thống, và sẽ được phân tích rõ hơn trong phần so sánh kết quả.
=== 2.2.1 Đặc trưng Color Histogram
==== 2.2.1.1 Định nghĩa

*Color Histogram* là phương pháp trích xuất đặc trưng màu sắc của ảnh bằng cách thống kê phân bố các giá trị màu trong không gian màu. 

Với mỗi ảnh, Color Histogram tạo ra một vector số thống kê mô tả tần suất xuất hiện của từng dải màu, cho phép so sánh và phân loại ảnh dựa trên đặc trưng màu sắc mà không cần quan tâm đến vị trí hay hình dạng của vật thể.
==== 2.2.1.2 Nguyên lý hoạt động
===== 2.2.1.2.1 Chọn không gian màu
Thay vì dùng không gian *RGB* (Red-Green-Blue), nhóm sử dụng không gian *HSV* (Hue-Saturation-Value) vì HSV tách biệt thông tin màu sắc (Hue) khỏi thông tin ánh sáng (Value), giúp đặc trưng màu ổn định hơn khi điều kiện chiếu sáng thay đổi. 

Điều này đặc biệt quan trọng với ảnh bọc nilon vì bọc gây phản chiếu ánh sáng làm thay đổi độ sáng bề mặt trái cây.

Ba kênh của HSV:

- Hue (H): Màu sắc thực sự (đỏ, vàng, xanh lá...)
- Saturation (S): Độ bão hòa màu (màu nhạt hay đậm)
- Value (V): Độ sáng (tối hay sáng)
===== 2.2.1.2.2 Phân chia không gian màu

Mỗi kênh HSV được chia thành các bin (khoảng giá trị). Số bin càng nhiều thì vector đặc trưng càng chi tiết nhưng cũng càng dài và dễ bị nhiễu. Ví dụ với kênh Hue chia thành 16 bin, mỗi bin đại diện cho một dải màu rộng 22.5° trong vòng tròn màu sắc.
===== 2.2.1.2.3 Đếm tần suất màu sắc

Với mỗi pixel trong ảnh, giá trị HSV của pixel đó được ánh xạ vào bin tương ứng và bộ đếm của bin đó tăng lên 1. Kết quả là một vector histogram thể hiện phân bố màu sắc toàn cục của ảnh.


#figure(
  grid(
    rows: 2,
    row-gutter: 5pt,
    image("../Figures/cam.png", width: 90%),
    image("../Figures/xoai.png"),
  ),
  caption: [
    Hue Histogram của Cam và Xoài
  ]
)

- Cam có phần lớn pixel tập trung ở vùng Đỏ - Cam
- Trong khi xoài tập trung ở vùng Vàng - Xanh lá
- Điều này cho thấy Color Histogram nắm bắt được sự khác biệt màu sắc đặc trưng giữa hai loại trái cây.

==== 2.2.1.3 Ưu và nhược điểm
*Ưu điểm:*

- Đơn giản, tính toán nhanh
- Không cần lo lắng ảnh bị xoay và dịch chuyển vì trái cây đặt ở vị trí hay góc nào trong ảnh cũng cho histogram tương tự
- Hiệu quả cho bài toán phân loại trái cây vì mỗi loại có tông màu đặc trưng rõ ràng
- Sử dụng HSV giúp ổn định hơn khi ánh sáng thay đổi so với RGB.

*Nhược điểm:*

- Mất thông tin không gian: không phân biệt được vị trí màu trong ảnh, dẫn đến hai ảnh có phân bố màu giống nhau nhưng nội dung khác nhau có thể cho histogram tương tự.
- Nhạy cảm với bọc nilon: phản chiếu ánh sáng làm thay đổi giá trị Hue và Saturation, khiến màu trái cây bị lệch.
- Không nắm bắt được thông tin hình dạng: hai loại trái cây cùng màu (táo đỏ và cà chua đỏ) dễ bị nhầm lẫn nếu chỉ dùng Color Histogram đơn độc.
=== 2.2.2 Đặc trưng HOG
==== 2.2.2.1 Định nghĩa
HOG (Histogram of Oriented Gradients) là phương pháp trích xuất đặc trưng hình dạng của ảnh bằng cách thống kê phân bố hướng gradient (cạnh, đường viền) trong các vùng cục bộ. 

Thay vì mô tả màu sắc như Color Histogram, HOG mô tả cấu trúc hình học của vật thể, giúp phân biệt các loại trái cây dựa trên hình dạng đặc trưng.
==== 2.2.2.2 Nguyên lý hoạt động
===== 2.2.2.2.1 Tính Gradient
Với mỗi pixel, HOG tính gradient theo hai chiều ngang và dọc bằng bộ lọc Sobel:

#set math.mat(delim: "[")

$
G_x = mat(
  -1, 0, 1;
  -2, 0, 2;
  -1, 0, 1
) * I, quad
G_y = mat(
  -1, -2, -1;
   0,  0,  0;
   1,  2,  1
) * I
$

Từ đó tính độ lớn và hướng gradient:

$
abs(G) = sqrt(G_x^2 + G_y^2) -> "Độ lớn (magnitude) của vector gradient"
\
theta = arctan( G_y / G_x ) -> "Hướng (orientation) của vector gradient"
$
===== 2.2.2.2.2 Tạo ô (cell) và khối (block)
Ảnh được chia thành các cell nhỏ (ví dụ 8×8 pixel). Mỗi cell là một vùng cục bộ mà HOG tính histogram hướng gradient riêng. Các cell liền kề được nhóm lại thành block (ví dụ 2×2 cell) để chuẩn hóa.

Ví dụ với ảnh 128×128 pixel, pixels_per_cell = (8,8):
- Số cell theo chiều ngang: 128/8 = 16 cell
- Số cell theo chiều dọc: 128/8 = 16 cell
- Tổng: 16×16 = 256 cell
===== 2.2.2.2.3 Tạo histogram hướng trong mỗi ô
Trong mỗi cell, hướng gradient (0°–180°) được chia thành 9 bin, mỗi bin rộng 20°. Mỗi pixel trong cell bỏ phiếu vào bin tương ứng với hướng gradient của nó, với trọng số là độ lớn gradient.

Ví dụ: 
- Tại vùng cạnh trái cây, gradient có hướng rõ ràng → bin tương ứng cao. 
- Tại vùng nền đồng nhất, gradient nhỏ → histogram phẳng.

===== 2.2.2.2.4 Chuẩn hóa khối
Mỗi block được chuẩn hóa L2 để giảm ảnh hưởng của thay đổi ánh sáng:
$ v' = v / sqrt(||v||_2^2 + ϵ^2) $

Trong đó $v$ là vector histogram của block, $ϵ$ là hằng số nhỏ tránh chia cho 0.
====== 2.2.2.2.5 Kết hợp đặc trưng
Các histogram của tất cả block được nối lại thành một vector đặc trưng 1 chiều, cũng chính là đầu vào cho KNN và SVM.
#figure(
  grid(
    columns: 2,
    image("../Figures/hog_banana.png", width: 80%),
    image("../Figures/hog_tomato.png", width: 80%)
  ),
  caption: [
    Minh hoạ đặc trưng HOG của Chuối (trái) và Cà chua (phải)
  ]
)
==== 2.2.2.3 Ưu nhược điểm
*Ưu điểm:*
- Nắm bắt tốt thông tin hình dạng và đường viền, chuối (hình dài cong), xoài (hình bầu dục) có HOG rất khác nhau
- Không thay đổi khi thay đổi ánh sáng nhờ chuẩn hóa khối
- Hiệu quả với các loại trái cây có hình dạng đặc trưng rõ ràng

*Nhược điểm:*
- Nhạy cảm với bọc nilon: bọc làm mờ đường viền trái cây, gradient tại cạnh trái cây giảm mạnh, HOG mất đi đặc trưng hình dạng quan trọng
- Không nắm bắt được thông tin màu sắc: cần kết hợp với Color Histogram để phân biệt các loại trái cây cùng hình dạng (táo và cà chua đều tròn)
- Chi phí tính toán tăng theo kích thước ảnh

Vì hai loại đặc trưng HOG và Color bổ sung lẫn nhau (HOG mô tả hình dạng nhưng thiếu màu, Color Histogram mô tả màu nhưng thiếu hình dạng), nhóm kết hợp cả hai trong các thí nghiệm KNN và SVM.
#pagebreak()
== 2.3 Các mô hình phân loại
=== 2.3.1 KNN
==== 2.3.1.1 Nguồn gốc

KNN (K-Nearest Neighbors) được giới thiệu vào năm 1951, là một trong những thuật toán Machine Learning đơn giản và lâu đời nhất. KNN thuộc nhóm instance-based learning. Mặc dù KNN không có pha học theo nghĩa truyền thống, GridSearchCV vẫn cần thiết để chọn siêu tham số tối ưu.

==== 2.3.1.2 Nguyên lý hoạt động

*Bước 1 - Tính khoảng cách:* Tính khoảng cách giữa $bold(x)$ và tất cả điểm trong tập train. Có nhiều công thức đo khoảng cách khác nhau:
- Euclidean: $ d(x,x') = sqrt(sum_i (x_i-x_i ')^2) $

- Manhattan: $ d(x,x') = sum_i |(x_i-x_i ')| $

- Cosine:  $ d(x,x') = 1 - (x.x')/(||x||_2 dot ||x'||_2) $

Việc lựa chọn công thức phù hợp phụ thuộc vào đặc trưng của bài toán, nhóm sử dụng _GridSearchCV_ để tự động tìm metric tối ưu, kết quả được trình bày ở phần Cài đặt.

*Bước 2 - Tìm K láng giềng gần nhất:* Chọn K điểm trong tập train có khoảng cách nhỏ nhất đến $bold(x)$

*Bước 3 : Bỏ phiếu đa số:* Nhãn dự đoán là nhãn xuất hiện nhiều nhất trong K láng giềng. Nếu dùng `weights=distance`, các láng giềng gần hơn có trọng số cao hơn trong quá trình bỏ phiếu:
$ hat(y) = "arg max"_(c in cal(L)) sum_(i=1)^K w_i dot 1{y_i = c}, quad
w_i = 1 / d(x, x_i) $

$ 1[dot] "là hàm chỉ thị, bằng 1 nếu điều kiện đúng, 0 nếu sai" $


#figure(
  image("/Figures/knn_illustration.png", width: 60%),
  caption: [
    Minh họa nguyên lý KNN trong không gian đặc trưng 2 chiều.
  ]
)

Trong hình trên, điểm ngôi sao là ảnh cần phân loại, vòng tròn bao gồm K=5 láng giềng gần nhất, đa số thuộc lớp Táo nên ảnh được phân loại là Táo. (Hình minh họa, không phải dữ liệu thực nghiệm)

==== 2.3.1.3 Cài đặt trong bài toán
*Trích xuất đặc trưng*:
KNN không thể xử lý trực tiếp ảnh pixel nên cần chuyển ảnh thành vector đặc trưng 1 chiều. Hai đặc trưng chính được sử dụng là HOG và Color Histogram HSV (đã trình bày ở mục 2.2), với tham số cụ thể:

- HOG: `orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)`
- Color Histogram HSV: `bins=[16, 8, 4]` cho kênh H, S, V

Ngoài ra, nhóm thử nghiệm thêm các đặc trưng bổ sung nhằm đánh giá xem việc tăng thêm thông tin có cải thiện KNN không:
- LBP (Local Binary Pattern): Mô tả texture bề mặt( radius=3, n_points=24)
- Color Moments: Thống kê mean, std, skewness của từng kênh HSV
- Hu Moments: Mô tả hình dạng bất biến với phép xoay và tỉ lệ

Từ đó nhóm xây dựng ba cấu hình đặc trưng với độ phức tạp tăng dần:

#table(
  columns: 2,
  [*Cấu hình*],[*Đặc trưng*],
  [`hog_hsv`],[HOG + Color Histogram HSV],
  [`full`],[HOG + Color Histogram HSV + LBP],
  [`full_v2`],[HOG + Color Histogram HSV + LBP + Color Moments + Hu Moments]
)

#import "@preview/showybox:2.0.4": showybox

#showybox(breakable: true,
  [
    ```py
    def extract_features(images, config='full'):
    vecs = []
    for img_bgr in images:
        img_u8  = (img_bgr * 255).astype(np.uint8) \
                  if img_bgr.dtype != np.uint8 else img_bgr
        gray    = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

        hog_f   = extract_hog(gray)
        color_f = extract_color_histogram(img_u8)

        if config == 'full_v2':
            lbp_f    = extract_lbp(gray)
            moment_f = extract_color_moments(img_u8)
            hu_f     = extract_hu_moments(gray)
            vec = np.concatenate([hog_f, color_f, lbp_f, moment_f, hu_f])
        elif config == 'full':
            lbp_f = extract_lbp(gray)
            vec = np.concatenate([hog_f, color_f, lbp_f])
        else:  # hog_hsv
            vec = np.concatenate([hog_f, color_f])

        vecs.append(vec)
    return np.array(vecs)
    ```
  ]
)

*Pipeline xử lý:*
Sau khi trích xuất, vector đặc trưng được xử lý qua hai bước trước khi đưa vào KNN:
- *StandardScaler*: Chuẩn hóa về mean=0, std=1, đảm bảo các đặc trưng có thang đo khác nhau không ảnh hưởng đến tính toán khoảng cách
- *PCA: * Giảm chiều giữ lại 98% variance `(n_components=0.98)`, giảm số chiều của vector đặc trưng, tăng tốc tính toán khoảng cách và giảm _curse of dimensionality_ 

Sau PCA, vector đặc trưng giảm từ $26756$  chiều xuống còn $3199$ chiều, giữ lại 98% variance

#showybox(
  [
    ```py
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)
    
    pca = PCA(n_components=0.98, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    print(f'Sau PCA: {X_train_pca.shape[1]} chiều')
    ```
  ]
)

*GridSearchCV: * Nhóm dùng GridSearchCV với 5-fold cross-validation để tìm bộ tham số tối ưu, thực hiện độc lập cho cả ba cấu hình đặc trưng:

#table(
  columns: 2,
  [*Tham số*],[*Giá trị thử nghiệm*],
  [`n_neighbors`],[`[3, 5, 7, 9, 11, 13, 15]`],
  [`weights`],[`[uniform, distance]`],
  [`metric`],[`[euclidean, manhattan, cosine]`]
)

#showybox(breakable: true,
  [
    ```py
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_pca, y_train)
print(f"Tham số tốt nhất: {grid_search.best_params_}")
print(f"CV Accuracy tốt nhất: {grid_search.best_score_:.4f}")
    ```
  ]
)

Sau khi chạy GridSearchCV trên cả ba cấu hình, hog_hsv cho kết quả tốt nhất với bộ tham số tối ưu:
#table(
  columns: 2,
  [*Tham số*],[*Giá trị tối ưu*],
  [Cấu hình đặc trưng],[`hog_hsv`],
  [`n_neighbors (K)`],[3],
  [`weights`],[`distance`],
  [`metric`],[`cosine`]
)



*Đáng chú ý:* 
- Cả ba cấu hình đều được thử nghiệm độc lập qua GridSearchCV. 
- Nhóm chọn hog_hsv làm cấu hình chính vì đây là cấu hình đơn giản nhất, khi các cấu hình có hiệu năng tương đương, ưu tiên mô hình ít phức tạp hơn 

Kết quả thực nghiệm cụ thể được trình bày ở Chương 3.

=== 2.3.2 SVM
==== 2.3.2.1 Nguồn gốc
SVM (Support Vector Machine) được giới thiệu vào năm 1995, là một trong những thuật toán Machine Learning truyền thống mạnh nhất cho bài toán phân loại. Khác với KNN dựa trên khoảng cách, SVM tìm một siêu phẳng phân tách tối ưu (optimal hyperplane) trong không gian đặc trưng, tức siêu phẳng có margin lớn nhất giữa các lớp.


==== 2.3.2.2 Nguyên lý hoạt động
*Tìm siêu phẳng tối ưu:*

Với bài toán phân loại nhị phân, SVM tìm siêu phẳng $w^T x + b = 0$ sao cho margin giữa 2 lớp là lớn nhất:
$
min_(w, b) 1/2 norm(w)^2
quad "subject to" quad
y_i (w^T x_i + b) >= 1, quad forall i
$
Các điểm dữ liệu nằm trên biên margin được gọi là _support vectors_ , đây là những điểm duy nhất ảnh hưởng đến vị trí siêu phẳng.

*Hàm mục tiêu với soft margin: *

Trong thực tế, dữ liệu thường không phân tách tuyến tính hoàn toàn. SVM cho phép một số điểm vi phạm margin thông qua tham số C:
- C lớn: phạt vi phạm nặng → margin hẹp hơn, dễ overfit
- C nhỏ: chấp nhận vi phạm nhiều hơn → margin rộng hơn, tổng quát hơn

*Kernel Trick: *

Với dữ liệu không phân tách tuyến tính trong không gian gốc, SVM dùng kernel function để ánh xạ dữ liệu lên không gian chiều cao hơn — nơi dữ liệu có thể phân tách tuyến tính. Nhóm sử dụng RBF kernel (Radial Basis Function):
$
K(x, x') = exp(-gamma norm(x - x')^2)
$

Tham số $gamma$ kiểm soát phạm vi ảnh hưởng của từng điểm dữ liệu:
  - $gamma$ lớn: mỗi điểm chỉ ảnh hưởng vùng nhỏ xung quanh → dễ overfit
  - $gamma$ nhỏ: mỗi điểm ảnh hưởng vùng rộng hơn → tổng quát hơn
#v(3pt)
Trong bài toán phân loại 5 loại trái cây, `scikit-learn` tự động xử lý phân loại đa lớp thông qua cơ chế tích hợp sẵn của SVC, nhóm không cần cấu hình thêm mà tập trung vào việc tìm bộ tham số C và gamma tối ưu thông qua GridSearchCV.

#figure(
  image("/Figures/svm_illustration.png", width: 70%),
  caption: [
    Minh hoạ cách nguyên lí SVM hoạt động
  ]
)

Trong hình trên:
- Đường liền nét là siêu phẳng phân tách tối ưu
- Hai đường nét đứt là biên margin
- Các điểm khoanh tròn xanh là support vectors. 

SVM tìm siêu phẳng có margin lớn nhất giữa hai lớp, chỉ các support vectors ảnh hưởng đến vị trí siêu phẳng. (Hình minh họa, không phải dữ liệu thực nghiệm)"
==== 2.3.2.3 Cài đặt trong bài toán
*Trích xuất đặc trưng:*

Tương tự KNN, SVM yêu cầu vector đặc trưng 1 chiều. Nhóm thực hiện hai thử nghiệm để đánh giá vai trò của từng loại đặc trưng:
#table(
  columns: 2,
  [*Cấu hình*],[*Đặc trưng*],
  [`hog-only`],[Chỉ HOG],
  [`hog-color`],[HOG + Color Histogram HSV],
)

Tham số HOG:

- `orientations=8`: chia 180° thành 8 bin, mỗi bin 22.5°. Số bin này nhỏ hơn KNN (9 bin) vì trái cây có hình dạng đơn giản, không cần phân biệt hướng gradient quá chi tiết
- `pixels_per_cell=(16,16)`: cell lớn hơn KNN (8×8) vì ảnh đầu vào nhỏ hơn (128×128) và hình dạng trái cây không cần mô tả quá chi tiết. Cell lớn hơn cũng giúp vector HOG ngắn hơn, giảm thời gian tính toán GridSearchCV
- cells_per_block=(2,2)

Tham số Color Histogram HSV:
- `bins=[8, 8, 8]` cho kênh H, S, V — ít bin hơn KNN `([16, 8, 4])` để giảm chiều vector, phù hợp với ảnh `128×128`

#showybox(breakable: true,
[
  ```py
  def features_extraction(img_path, config='hog-only', img_size=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    features = []

    if 'hog' in config:
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog(
            gray_scale,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        features.extend(hog_features)

    if 'color' in config:
        hsv_scale = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_scale], [0, 1, 2], None,
                            [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())

    return np.array(features)
  ```
]
)

*Pipeline xử lý:*

Khác với KNN, SVM không dùng PCA vì vector đặc trưng đã nhỏ (ảnh 128×128, pixels_per_cell=(16,16)) nên không cần giảm chiều thêm. Chỉ dùng `StandardScaler` để chuẩn hóa, bước này đặc biệt quan trọng với RBF kernel vì kernel tính khoảng cách trong không gian đặc trưng, nếu các đặc trưng có thang đo khác nhau sẽ làm lệch kết quả:

#showybox(breakable: true,
[
  ```py
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
  ```
]
)

*GridSearchCV:*

Nhóm dùng GridSearchCV với `5-fold` cross-validation, thực hiện độc lập cho cả hai cấu hình `hog-only` và `hog-color`:

#table(
  columns: 3,
  [*Tham số*],[*Giá trị thử nghiệm*],[*Ý nghĩa*],
  [$C$],[`[0.1, 1, 10, 100]`],[Mức độ phạt vi phạm margin, C lớn thì margin hẹp hơn],
  [Gamma],[`[scale, auto, 0.1, 0.01]`],[Phạm vi ảnh hưởng của RBF kernel],
  [Kernel],[`[rbf]`],[Cố định RBF, phù hợp với dữ liệu không phân tách tuyến tính],
)

#showybox(breakable: true,
[
  ```py
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best Params: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
  ```
]
)

Sau khi chạy GridSearchCV trên cả hai cấu hình, bộ tham số tối ưu tìm được:
#table(columns: 4,
[*Cấu hình*],[*$C$*],[*Gamma*],[*Kernel*],
[`hog-only`],[`100`],[`scale`],[`rbf`],
[`hog-color`],[`10`],[`scale`],[`rbf`],
)
Một điểm đáng chú ý: 

- `hog-only` cần `C=100` (phạt vi phạm rất nặng) để đạt kết quả tốt nhất, cho thấy chỉ dùng HOG thì các lớp khó phân tách, SVM phải tìm boundary rất chặt.
- Trong khi đó `hog-color` chỉ cần `C=10`, thêm màu sắc giúp các lớp phân tách dễ hơn, không cần margin quá chặt.
=== 2.3.3 CNN 
==== 2.3.3.1 Nguồn gốc
CNN (Convolutional Neural Network) được giới thiệu vào năm 1998 với kiến trúc LeNet - mạng nơ-ron đầu tiên ứng dụng thành công phép tích chập cho bài toán nhận dạng chữ viết tay. Khác với KNN và SVM yêu cầu trích xuất đặc trưng thủ công, CNN tự động học đặc trưng trực tiếp từ pixel ảnh thông qua các lớp tích chập, đây cũng là ưu điểm cốt lõi của Deep Learning so với Machine Learning truyền thống.
==== 2.3.3.2 Nguyên lý hoạt động
*Lớp tích chập (Convolutional Layer):*

Mỗi lớp tích chập áp dụng một bộ filter (kernel) lên ảnh đầu vào bằng phép tích chập:

$ (I ast K)(i, j) = sum_m sum_n I(i + m, j + n) dot K(m, n) $

Trong đó:
- I là ảnh đầu vào
- K là filter
Mỗi filter học một đặc trưng cụ thể:
- Filter ở lớp đầu học các đặc trưng đơn giản như cạnh và góc
- Filter ở lớp sâu hơn học các đặc trưng phức tạp hơn như hình dạng và texture. 

Với bài toán phân loại trái cây, các lớp đầu có thể học đường viền trái cây, các lớp sau học hình dạng tổng thể và màu sắc đặc trưng.

*Lớp pooling (MaxPooling): *

- Sau mỗi lớp tích chập, MaxPooling giảm kích thước feature map bằng cách lấy giá trị lớn nhất trong mỗi vùng:

- Giảm số tham số → tránh overfitting

Tăng tính bất biến với dịch chuyển, trái cây đặt ở vị trí khác nhau trên cân vẫn cho feature map tương tự

*Hàm kích hoạt ReLU: *

Sau mỗi lớp tích chập, hàm ReLU được áp dụng để tạo tính phi tuyến:

$ "ReLU"(x) = max(0, x) $

*BatchNormalization: *

Chuẩn hóa output của mỗi lớp về `mean=0`, `std=1` trong mỗi mini-batch, điều này giúp gradient ổn định hơn và mô hình hội tụ nhanh hơn.

*Dropout: *

Tắt ngẫu nhiên một tỉ lệ neuron trong quá trình huấn luyện — buộc mô hình học đặc trưng tổng quát thay vì ghi nhớ dữ liệu, giảm overfitting hiệu quả.

*GlobalAveragePooling2D: *

Thay thế Flatten truyền thống : tính trung bình toàn bộ feature map thay vì trải phẳng. Giảm mạnh số tham số ở phần classifier, hạn chế overfitting.
==== 2.3.3.3 Cài đặt trong bài toán
*Pipeline dữ liệu: *

- Dữ liệu được load trực tiếp từ thư mục bằng `image_dataset_from_directory` của `Keras`, tự động resize về `224×224`, chia `batch size=32` và shuffle tập train.
- Augmentation (RandomFlip, RandomRotation) được tích hợp trực tiếp vào kiến trúc model như các lớp đầu tiên. Mỗi epoch ảnh được biến đổi ngẫu nhiên khác nhau trong bộ nhớ RAM, không thay đổi file gốc.
- Chuẩn hóa pixel `[0, 255] → [0, 1]` được thực hiện bởi lớp Rescaling(1./255) bên trong model. 
Khác với KNN và SVM thực hiện chuẩn hóa bên ngoài bằng StandardScaler. Cách này đảm bảo khi deploy model, bước chuẩn hóa luôn được thực hiện tự động mà không cần xử lý thêm bên ngoài.

*Kiến trúc: *

Nhóm xây dựng CNN theo kiến trúc Sequential gồm 3 khối tích chập và một bộ phân loại:

#table(columns: 2,
[*Lớp*],[*Chi tiết*],
[`Rescaling`],[Chuẩn hóa pixel $[0, 255] -> [0, 1]$],
[`Augmentation`],[`RandomFlip(horizontal), RandomRotation(±30°)`],
[`Block 1`],[$"Conv2D"(32, 3 times 3, "relu") -> "BatchNorm" -> "MaxPool"(2 times 2) -> "Dropout"(0.25)$],
[`Block 2`],[$"Conv2D"(64, 3 times 3, "relu") -> "BatchNorm" -> "MaxPool"(2 times 2) -> "Dropout"(0.25)$],
[`Block 3`],[$"Conv2D"(128, 3 times 3, "relu") -> "BatchNorm" -> "MaxPool"(2 times 2) -> "Dropout"(0.25)$],
[`Classifier`],[$"GlobalAveragePooling" -> "Dense"(128,"relu") -> "Dropout"(0.5) -> "Dense"(5, "softmax")$],
)

#showybox(breakable: true,
[
  ```py
model = keras.Sequential([
    keras.layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.083),

    # Block 1
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    # Block 2
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    # Block 3
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    # Classifier
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation='softmax')
], name='CNN')
  ```
]
)

*Huấn luyện: *

- Optimizer: Adam với learning rate  $=1e-3$
- Loss function: Sparse Categorical Crossentropy ,phù hợp với nhãn dạng số nguyên
- Batch size: $32$
- Epochs tối đa: $10$

#showybox(breakable: true,
[
  ```py
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
  ```
]
)

*Callbacks: *

- `EarlyStopping` (`patience=5`, `monitor=val_loss`): dừng huấn luyện sớm nếu `val_loss` không cải thiện sau `5 epoch`, tự động khôi phục trọng số tốt nhất
- `ReduceLROnPlateau` (`patience=3`, `factor=0.5`, `monitor=val_loss`): giảm learning rate xuống một nửa khi `val_loss` bão hòa
- `ModelCheckpoint` (`monitor=val_accuracy`): lưu checkpoint tại epoch có `val_accuracy` cao nhất

#showybox(breakable: true,
[
  ```py
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=f'{MODEL_DIR}/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)
  ```
]
)
#figure(
  image("/Figures/CNN_struc.png", width: 110%),
  caption: [
    Minh hoạ kiến trúc CNN được nhóm xây dựng
  ]
)


=== 2.3.4 ResNet18
==== 2.3.4.1 Nguồn gốc

ResNet (Residual Network) được giới thiệu vào 2015 trong bài báo "_Deep Residual Learning for Image Recognition_" , giành chiến thắng tại cuộc thi ImageNet 2015. ResNet18 là phiên bản nhỏ nhất của họ ResNet với 18 lớp, phù hợp cho các bài toán có dataset vừa và nhỏ.

==== 2.3.4.2 Nguyên lý hoạt động
*Vấn đề của mạng sâu:*

Khi tăng số lớp của CNN, hiệu năng không nhất thiết tăng theo, thậm chí có thể giảm do gradient bị suy giảm dần khi lan truyền ngược qua nhiều lớp, khiến các lớp đầu gần như không được cập nhật.

*Skip Connection - Giải pháp của ResNet*

ResNet giải quyết vấn đề này bằng cơ chế Skip Connection (kết nối tắt): cho phép gradient nhảy qua một số lớp khi lan truyền ngược:

$ y = cal(F)(x, {W_i}) + x $

Trong đó:
- $x$ là input
- $cal(F)(x)$ là phần học thêm (residual)
- $y$ là output

Gọi $H(x)$ là mapping đích mà khối mạng cần học (tương ứng với $y$ khi không có skip connection). Thay vì học trực tiếp $H(x)$, mạng chỉ cần học phần dư:

$ cal(F)(x) = H(x) - x $

Cách tiếp cận này hiệu quả vì khi $H(x) approx x$ (mapping gần với phép đồng nhất), việc học $cal(F)(x) approx 0$ dễ hơn nhiều so với học $H(x)$ trực tiếp.

*Transfer Learning:*

Thay vì huấn luyện từ đầu, nhóm sử dụng ResNet18 đã được pretrained trên ImageNet: tập dữ liệu gồm 1.4 triệu ảnh và 1000 lớp.

Các lớp tích chập đã học được các đặc trưng tổng quát (cạnh, hình dạng, texture) có thể tái sử dụng cho bài toán phân loại trái cây.

Quá trình Transfer Learning gồm hai bước:

- *Bước 1:* Load ResNet18 pretrained, thay lớp Fully Connected cuối (1000 lớp) bằng lớp mới (5 lớp)
- *Bước 2:* Fine-tune toàn bộ mạng với learning rate nhỏ — vì ảnh trái cây trong bọc nilon có đặc thù khác ImageNet, cần train toàn bộ để mạng thích nghi
==== 2.3.4.3 Cài đặt trong bài toán

*Pipeline dữ liệu: *

Ảnh được resize về `224×224` và chuẩn hóa theo mean/std của ImageNet (đã trình bày ở #ref(<resize>, form: "page") và #ref(<normalization>, form: "page")) — bước bắt buộc để tương thích với pretrained weights. Augmentation chỉ áp dụng trên tập train

- Train: 
  - RandomHorizontalFlip(p=0.5)
  - RandomRotation(±15°)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Val/Test: Chỉ Normalize — không augmentation


#showybox(breakable: true,
[
  ```py
# 1. Pipeline cho tập TRAIN (Có Augmentation để chống overfit)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Pipeline cho tập VAL/TEST
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
  ```
]
)

*Load pretrained và thay lớp cuối:*

#showybox(breakable: true,
[
  ```py
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)

# Fine-tune toàn bộ
for param in model.parameters():
    param.requires_grad = True
  ```
]
)

Nhóm quyết định fine-tune toàn bộ các lớp thay vì freeze phần backbone vì:
- Ảnh trái cây trong bọc nilon có đặc thù rất khác ImageNet, các lớp tích chập cần được điều chỉnh để thích nghi.
- Việc dùng learning rate nhỏ đảm bảo các pretrained weights được cập nhật từ từ, không bị phá hủy đột ngột.

*Huấn luyện: *

- Optimizer: `Adam`
- Loss function: `CrossEntropyLoss`
- EarlyStopping: `patience=5`, theo dõi `val_loss`, lưu checkpoint tại val_loss thấp nhất


*Grid Search thủ công:*

Vì ResNet18 dùng `PyTorch` thay vì `scikit-learn`, nhóm tự cài Grid Search theo hai giai đoạn:

- Giai đoạn 1, Quick Search (3 epochs/tổ hợp): Thử tất cả tổ hợp tham số để lọc nhanh tín hiệu
- Giai đoạn 2, Full Training (20 epochs): Huấn luyện chính thức với bộ tham số tốt nhất từ giai đoạn 1

Không gian tham số:

#table(
  columns: 2,
  [*Tham số*],[*Giá trị thử nghiệm*],
  [`batch_size`],[`[32, 64, 128]`],
  [`learning rate`],[`[0.001, 0.0001, 0.00001]`],
  [`gamma` (LR Scheduler)],[`[0.1, 0.5]`]
)

#showybox(breakable: true,
[
  ```py
param_grid = {
    'batch_size': [32, 64, 128],
    'lr': [0.001, 0.0001, 0.00001],
    'gamma': [0.1, 0.5]
}

# Giai đoạn 1: Quick Search
for params in combinations:
    _, history = train_model(
        model_trial, train_loader, val_loader,
        criterion, optimizer_trial, num_epochs=3
    )
    current_acc = history['val_acc'][-1]
    if current_acc > best_val_acc:
        best_val_acc = current_acc
        best_params = params

# Giai đoạn 2: Full Training
final_model, final_history = train_model(
    final_model, train_loader, val_loader,
    criterion, optimizer_final,
    scheduler=scheduler_final, num_epochs=20
)
  ```
]
)


*Tham số tối ưu: *

Sau khi chạy GridSearch trên cấu hình, bộ tham số tối ưu tìm được:

#table(
  columns: 2,
  [*Tham số*],[*Giá trị tối ưu*],
  [`batch_size`],[`64`],
  [`learning rate`],[`0.0001` ($1e-4$)],
  [`gamma` (LR Scheduler)],[`0.1`]
)

#pagebreak()