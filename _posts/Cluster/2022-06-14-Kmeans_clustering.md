---
title: K-means Clustering
layout: default
excerpt: Trước đó, chúng ta đã đi qua và có một cái nhìn tổng quát về bài toán clustering cũng như các cách tiếp cận của bài toán này. Hôm nay, ngày này đây, chúng ta sẽ đi tìm hiểu một trong những thuật toán phân cụm cơ bản và đầu tiên, thuật toán K-means ...
tags: ["clustering"]
---

# Giới Thiệu
Trước đó, chúng ta đã đi qua và có một cái nhìn tổng quát về bài toán clustering cũng như các cách tiếp cận của bài toán này. Hôm nay, ngày này đây, chúng ta sẽ đi tìm hiểu một trong những thuật toán phân cụm cơ bản và đầu tiên, thuật toán K-means.
# Ý Tưởng Của Thuật Toán K-means
Khi nhìn một đống bùi nhùi điểm dữ liệu, với thông tin đã biết số lượng cụm (cluster) cần phải chia. K-means sẽ giả sử mỗi một cụm sẽ được đại diện bởi các điểm gọi là tâm cụm và các điểm xung quanh tâm cụm này thường sẽ thuộc cụm mà tâm đó đại diện.
<img src="">
<p>Hình 1:</p>
Khi đó, sẽ có một hay nhiều cụm và tâm cụm khác nhau, với mỗi điểm dữ liệu khi xem xét, nó sẽ thuộc vào cụm có khoảng cách từ nó tới tâm cụm nào mà có giá trị nhỏ nhất. Do vậy, mục tiêu của bái toán ở đây là việc tối ưu hóa giải quyết 2 vấn đề.

* Tìm tâm cụm cho các cụm dữ liệu (vì lúc đầu K-means sẽ khởi tạo các cụm ngẫu nhiên, và chúng ta cần tối ưu tìm tâm cụm sao cho các điểm xung quanh tâm cụm đó là thuộc cụm đó chứ không phải cụm khác)
* Xét xem với 1 điểm dữ liệu thì nó sẽ thuộc cụm nào

Sao cho tổng bình phương khoảng cách giữa các đối tượng đến tâm nhóm (centroid) là nhỏ nhất. Để có thể phân cụm được:
Bước 1: K-means sẽ thực hiện khởi tạo random K tâm cụm khác nhau, đại diện cho K cụm dữ liệu.
Bước 2: Thực hiện tính khoảng cách (euclidean, cosine, manhattan, ...) từ tất cả các tâm cụm tới tất cả các điểm dữ liệu. Điểm dữ liệu có khoảng cách tới tâm cụm nào ngắn nhất thì sẽ thuộc cụm đó. Khi đó, mỗi một cụm sẽ có lượng dữ liệu nhất định.
Bước 3: Thực hiện tính toán lại tâm cụm bằng cách, tâm cụm sẽ là trung bình của tất cả các điểm thuộc cụm đó.

Cứ lặp lại bước 2 và 3 liên tục cho tới khi thuật toán hội tụ. Tức, khi mà tại các vòng lặp có ít các điểm dữ liệu bị phân cụm nhầm sang cụm khác hay vị trí của điểm tâm cụm không thay đổi nhiều.

# Lựa Chọn Số Cluster (K)
Ngoài việc giải quyết 2 vấn đề được nêu ở trên, thì số lượng cluster (K) mà thuật toán K-means giả định là thông tin đã biết trước cũng là một vấn đề cần lưu ý. Khi nhìn vào đống điểm dữ liệu đầu vào, câu hỏi đặt ra là bao nhiêu cluster là đủ, là hợp lý ?.

Có rất nhiều phương pháp hỗ trợ giúp ta có thể lựa chọn số lượng K hiệu quả hơn trong đó có thể kể đến là:

* **Elbow** method
* Average **silhouette** method

**Elbow** method là một phương pháp lựa chọn K bằng cách thử nghiệm thuật toán với 1 dải giá trị K khác nhau và tại giá trị của K mà từ đó trở đi hàm loss cho thuật toán không thay đổi nhiều. Thì ta nên chọn giá trị tại đó làm K nơi mà sẽ đạt được tính chất phân cụm một cách tổng quát nhất.

<img src="">
<p>Hình 2:</p>

**Silhouette** cũng là một trong những phương pháp lựa chọn số cluster phổ biến. Ở đó, Silhouette dùng để đo lường chất lượng của một cụm. Tương tự như **Elbow** ta thử nghiệm với một dải các giá trị K, khi đó, average silhouette sẽ là chất lượng của các cụm được phân theo số K đó. Average silhouette càng cao chất lượng của các cụm càng tốt và giá trị K mà ta sẽ chọn là giá trị mà tại đó Average silhouette đạt kết quả cao nhất.

<img src="">
<p>Hình 3:</p>

# Hạn Chế

* Mặc dù có các phương pháp giúp xác định số lượng K. Tuy nhiên, trong thực tế nếu không ước lượng được K nằm ở khoảng giá trị nào để thử nghiệm dải giá trị K xung quanh giá trị đó thì cũng sẽ tốn một lượng thời gian nhất định để tìm ra K
* Tốc độ hội tụ của thuật toán phụ thuộc nhiều vào giá trị ban đầu của các tâm cụm


# Tài Liệu Tham Khảo


