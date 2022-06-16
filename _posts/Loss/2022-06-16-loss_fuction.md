---
layout: default
title: Loss Function
tags: ["loss function", "objective function"]
excerpt: Bạn là người mới bắt đầu hay người đã tiếp xúc lâu với machine learning dù ít hay nhiều chắc hẳn cũng đã nghe loáng thoáng qua thứ gọi là Loss Function (aka Hàm Mất Mát) ...
---

<style>
.tablelines table, .tablelines td, .tablelines th {

        border: 1px solid black;
        }
</style>

# Khái niệm
Bạn là người mới bắt đầu hay người đã tiếp xúc lâu với machine learning dù ít hay nhiều chắc hẳn cũng đã nghe loáng thoáng qua thứ gọi là Loss Function (aka Hàm Mất Mát).

Loss Function hay còn gọi là hàm mất mát là một trong những thứ thiết yếu trong machine learning.

Về cơ bản Loss Function là một hàm số cho phép bạn đo lường mức độ sai khác giữa kết quả mà mô hình dự đoán so với giá trị thực cần dự đoán. Nếu mô hình dự đoán sai nhiều tức độ sai khác giữa kết quả mô hình dự đoán và giá trị thực lớn thì giá trị của loss function sẽ lớn và ngược lại.

Mức độ sai khác giữa kết quả dự đoán và giá trị thực có thể được định nghĩa theo nhiều cách khác nhau phụ thuộc từng bài toán. Ví dụ, nôm na là nếu một bài toán dự đoán giá đất mà với mỗi dự đoán mà mô hình đưa ra dù đúng hay sai, chúng ta đều có một giá trị sai khác (mất mát) tính được từ loss function mà chúng ta định nghĩa, giả sử loss function sẽ là giá trị tuyệt đối của hiệu giữa giá trị dự đoán và giá trị thực.
$ y = \| y - y_{pre} \| $
* y: là giá trị thực
* $y_{pre}$: là giá trị mô hinh dự đoán
Khi đó với một số điểm dữ liệu làm đầu vào ta sẽ thu được các dự đoán cho từng điểm dữ liệu đó và tính được giá trị mất mát tại các điểm đó như sau:

| Y | $Y_{pre}$| Loss value|
| --| --| --|
| 50| 40| 10|
| 80| 80|  0|
| 70| 80| 10|
{: .tablelines}
<p>Bảng 1: Bảng giá trị thực và giá trị của mô hình dự đoán cùng với giá trị mất mát</p>

Do vậy, ta có thể tính được sự sai khác (giá trị mất mát) do mô hình dự đoán tại các điểm dữ liệu. Và để đánh giá được mức độ tốt hay tệ của mô hình ta cần tính trung bình giá trị sai khác do mô hình dự đoán tại tất cả các điểm dữ liệu quan sát.

# Objective Function
Như đã đề cập ở trên, với mỗi một điểm dữ liệu mô hình sẽ dự đoán cho điểm dữ liệu đó, độ sai khác càng lớn khi sự chênh lệch giữa giá trị mô hình dự đoán và giá trị thực càng lớn.

Do vậy, mục tiêu của quá trình huấn luyện mô hình học máy là việc tối ưu sao cho giá trị hàm mất mát càng nhỏ càng tốt. Tuy nhiên, đôi khi các hàm mất mát được định nghĩa như các hàm thưởng với mục tiêu sao cho mức độ thưởng càng lớn càng tốt thì khi đó việc tối ưu sẽ là việc làm sao cho giá trị thưởng đạt cực đại. Tất cả các hàm như vậy được gọi chung là objective function (aka hàm mục tiêu), và mục đích của một quá trình huấn luyện mô hình là để tối ưu một hoặc một số hàm mục tiêu sao cho mô hình đạt kết quả tốt nhất.

# Loss Function tham gia quá trình cập nhật trọng số weight như thế nào ?
Về cơ bản loss function là một hàm số, như chúng ta đã biết để tìm cực tiểu hay cực đại cho một hàm số thì sẽ sử dụng đạo hàm của chúng. Nơi mà tại $x_{t}$ là điểm cực tiểu (đạo hàm tại $x_{t}$ bằng 0) của hàm mất mát và với đạo hàm của mọi điểm bên phải xt sẽ luôn dương và bên trái $x_{t}$ sẽ luôn âm.

Có thể nói nếu có thể tìm được một hoặc nhiều điểm cực tiểu $x_{t}$ thì ta chỉ việc thay lần lượt vào hàm số và tìm ra điểm có hàm số nhỏ nhất là xong. Tuy nhiên, trên thực tế hầu hết các trường hợp không thể tìm được điểm cực tiểu sao cho đạo hàm của hàm mất mát bằng 0. Do vậy để có thể tìm được điểm có thể cho ra hàm mất mát nhỏ nhất có thể mà chấp nhận được thì ta sẽ thực hiện cập nhật dần dần, đi dần dần để tìm điểm cực tiểu đó.

Với mỗi điểm xi nếu đạo hàm tại đó làm cho hàm số dương tức xi sẽ nằm bên phải điểm cực tiểu xt do vậy để tiến tới $x_{t}$ ta cần đi lùi lại 1 đoạn  lần đạo hàm tại $x_{i}$. Tương tự nếu đạo hàm tại xi cho giá trị âm thì xi sẽ nằm bên trái xt và cũng cần đi ngược lại để có thể tới gần hơn với $x_{i}$ (chi tiết bạn có thể đọc thêm bài về Gradient Descent)

Để có thể đạt được điểm cực tiểu có thể chấp nhận được (1 điểm nào đó gần xt nhất có thể) thì ta sẽ thực hiện lặp đi lặp lại việc đi lùi với  lần đạo hàm và cập nhật trọng số để cho mô hình biết được rằng ta đã đi lùi 1 khoảng bao nhiêu và vị trí hiện tại ở đâu để tiếp tục đi lùi đạo hàm, cứ lặp đi lặp lại việc tính và cập nhật trọng số (hàm loss sau mỗi lần cập nhật vì xi ngày càng gần xt nên hàm loss ngày càng đạt được tới cực tiểu) cho tới khi hàm loss đạt cực tiểu ở mức chấp nhận được.

# Kết Luận
Có thể nói Loss Function là một trong những thành phần quan trọng khi xây dựng mô hình học máy, việc thực hiện tối ưu hàm loss function thường sử dụng đạo hàm nên việc loss function nên có đạo hàm dễ xác định thường được cân nhắc khi xây dựng.
