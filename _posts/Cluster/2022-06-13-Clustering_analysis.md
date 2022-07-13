
# Giới Thiệu

Mình nghĩ rằng **Clustering** không phải là một cái gì quá xa lạ nữa trong mảng machine learning. Clustering (aka phân cụm) là một trong những thuật toán cơ bản của lớp bài toán unsupervised learning. Trong thực tế, clustering có nhiều ứng dụng trong nhiều mảng khác nhau như nén dữ liệu, tin sinh học, truy xuất thông tin ..v.v, và trong phạm vi của blog này chúng ta sẽ chỉ bàn luận về giá trị của clustering trong mảng liên quan tới data science mà thôi.

# Khái Niệm

Nếu như ta có một đống bùi nhùi các điểm dữ liệu. Thì clustering có thể hiểu một cách đơn giản là việc phân các dữ liệu đó thành các cụm khác nhau sao cho mỗi điểm dữ liệu thuộc cùng một cụm sẽ có các đặc điểm, thuộc tính tương tự (hoặc kiểu kiểu giống nhau), và sẽ khác so với các điểm dữ liệu thuộc cụm khác.

Với những trải nghiệm mà mình có, thì trong thực tế tùy từng bài toán, clustering có thể được dùng để review và có cái nhìn tổng quát hơn về dữ liệu, hay một trong những ứng dụng mình thường thấy là việc sử dụng clustering để có thể phân cụm và gán nhãn một lượng lớn dữ liệu giúp giảm thiểu lỗ lực phải gán nhãn bằng sức người.

Ví dụ, nếu ta có một lượng lớn dữ liệu chưa được gán nhãn, và giờ ta muốn thực hiện gán nhãn đống dữ liệu đó. Thì thay vì ta sẽ làm bằng tay từng chút một, ta sẽ thực hiện phân dữ liệu thành các cụm khác nhau và ta sẽ chỉ cần sample một số mẫu của từng cụm ra và gán nhãn cho các sample đó. Khi đó, nhãn của tất cả các điểm dữ liệu trong cùm một cụm sẽ giống như nhãn của sample được lấy ra từ cụm đó.

# Các Loại Phân Tích Cụm

Như tên của nó, Clustering là việc phân dữ liệu ra thành các cụm khác nhau. Do vậy, cũng như bao thuật toán khác, sẽ có nhiều cách tiếp cận khác nhau để có thể phân dữ liệu thành các cụm khác nhau.

Trong bài viết này, ta sẽ đi qua một số các phương pháp tiếp cận cho việc phân cụm và đi qua về ý tưởng của chúng. Còn các lý thuyết sâu hơn cho từng phương pháp mình xin phép để ở các bài sau.

## Centroid-based Clustering

Phương pháp này, sẽ giả sử mỗi một cụm sẽ được đại diện bởi một thứ gọi là tâm cụm (centroid). Các điểm dữ liệu thuộc một cụm bất kỳ thì sẽ nằm xung quanh tâm của cụm đó, và điểm dữ liệu sẽ thuộc vào cụm có khoảng cách tới tâm của cụm nào gần với nó nhất. Trong đó K-means là một phương pháp tiêu biểu trong việc tiếp cận theo hướng Centroid-based Clustering này.

## Hierachical Clustering

Phương pháp này đưa ra cho chúng ta một cây phân cụm theo dạng cụm này có thể là tập con hoặc tập cha của cụm khác. Nếu như ta có một tập dữ liệu mà có độ nhập nhằng giữa các nhãn (ví dụ trong dữ liệu ngành nghề ta có cả nhãn ngành IT, nhãn IT - phần cứng và nhãn IT - phần mềm), thì với cách tiếp cận này sẽ cho ta một cái nhìn tổng quát vê dữ liệu khi ta có cây cụm với IT - phần mềm và IT - phần cứng là 2 nhánh nhỏ của IT. Trong cách tiếp cận này thì có Agglomerative và Divisive là 2 phương pháp thường được sử dụng.

## Distribution-base Clustering

Đây là một phương pháp tiếp cận phân cụm mà ở đó các điểm dữ liệu thuộc cùng một cụm sẽ cùng tuân theo một phân phối nào đó. Điều này sẽ giúp ta có thể nắm bắt được một số thuộc tính cũng như mối tương quan giữa các thuộc tinh với nhau. (cần đọc thêm)

## Density-base Clustering

Đây là phương pháp phận cụm, mà ở đó ta sẽ coi một cụm là nơi có các điểm dữ liệu với mật độ cao và nơi các điểm dữ liệu có mật độ thấp sẽ được coi là nhiễu. Phương pháp phổ biến trong cách tiếp cận này có thể kể đến là DBSCAN.

# Tài Liệu Tham Khảo

