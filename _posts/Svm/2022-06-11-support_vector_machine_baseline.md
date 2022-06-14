---
title: Support Vector Machine
layout: default
excerpt: SVMs (aka support vector machines) là một thuật toán quan trọng thường đường nhắc đến trong machine learning. Nhưng đôi khi ta lại hoang mang khi tìm kiếm tài liệu về SVMs, có khi ta thấy ghi là SVMs có khi lại thấy SVM hay SVC và SVR, chúng là gì và có phải tất cả đều là support vector machine không ? Bài viết này ...
tags: ["SVM"]
---

<style>
    .singleImg {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .textSingleImg {
        text-align: center;
    }
    .twoImgBlock {
        display: flex;
        just-content: center;
    }
    .twoImg {
        width: 400px;
        height: 400px;
        margin-left: 100px;
    }
</style>

<!-- code_chunk_output -->
- [Giới thiệu](#giới-thiệu)
- [SVMs là gì ?](#svms-là-gì-)
- [Ý tưởng chính của SVM](#ý-tưởng-chính-của-svm)
- [Soft margin](#soft-margin)
- [Kernel](#kernel)
- [Tài Liệu Tham Khảo<br/>](#tài-liệu-tham-khảo)
<!-- /code_chunk_output -->



# Giới thiệu

SVMs (aka support vector machines) là một thuật toán quan trọng thường đường nhắc đến trong machine learning. Nhưng đôi khi ta lại hoang mang khi tìm kiếm tài liệu về SVMs, có khi ta thấy ghi là SVMs có khi lại thấy SVM hay SVC và SVR, chúng là gì và có phải tất cả đều là support vector machine không ?

Bài viết này chúng ta sẽ đi qua các ý tưởng cũng như lớp lý thuyết tầng cao nhất để hiểu tổng quan về SVMs. Và các lý thuyết sâu hơn mình xin để dành ở các bài viết sau.

# SVMs là gì ?
Support vector machines (SVMs) là một thuật toán học máy được sử dụng rộng rãi và thường được dùng chính trong các bài toán regression và classification.

Trong đó, khi SVM áp dụng cho bài toán regression được gọi là SVR (support vector regression) và là SVC (support vector classification) khi sử dụng cho bài toán classification.

Trong phạm vi bài viết này, ta sẽ tập trung đi nhiều hơn vào SVC (loại bài toán mà mình cảm thấy là dùng SVM nhiều hơn cả).

# Ý tưởng chính của SVM

<img src="Assets/Pictures/Svm/1.png" class="singleImg">

<p class="textSingleImg">Hình 1: Các điểm dữ liệu của bài toán<p>

Về cơ bản trong một không gian N chiều chứa các điẻm dữ liệu (Hình 1) mục tiêu của SVM sẽ tìm 1 siêu phẳng (hyper plane) để có thể phân tách dữ liệu thành 2 phần tốt nhất có thể tương ứng với dữ liệu đó.

Nhắc lại một chút, thì trong 1 không gian N chiều, 1 siêu phẳng là 1 không gian con có kích thước N-1 chiều. Do vậy để dễ hình dung thì trong không gian 2 chiều siêu phẳng sẽ là 1 đường thẳng, trong không gian 3 chiều thì siêu phẳng sẽ là 1 mặt phẳng.

Có thể thấy nếu ta coi hình trên là một không gian 2 chiều thì để phân tách được dữ liệu thành 2 phần có rất nhiều đường thẳng có thể làm được điều đó như hình 2. Khi đó ta sẽ chọn đường thẳng nào, đường thẳng nào sẽ là tốt nhất để phân tách các điểm dữ liệu.

<img src="Assets/Pictures/Svm/2.png" class="singleImg">

<p class="textSingleImg">Hình  2: Vô số siêu phẳng (trong trường hợp này là đường thẳng) có thể phân chia dữ liệu thành 2 phần<p>

Trong SVMs, một siêu phẳng (hyperplane) được coi là tốt khi có thể phân tách 2 miền của các điểm dữ liệu sao cho margin là lớn nhất.

<img src="Assets/Pictures/Svm/3.png" class="singleImg">

<p class="textSingleImg">Hình  3: Siêu phẳng tốt nhất để phân tách dữ liệu<p>

Trong đó, margin là khoảng cách giữa siêu phẳng đến 2 điểm dữ liệu gần nhất tương ứng với các phân lớp sao cho khoảng cách từ mỗi điểm dữ liệu của các phân lớp đó tới siêu phẳng là bằng nhau. Điều đó nhằm đảm bảo tính công bằng trong việc phân lớp dữ liệu tránh việc hyperplane bị ở gần 1 trong 2 lớp. Khi đó phân lớp mà có hyperplane ở gần sẽ dễ bị phân loại sai khi có thêm dữ liệu vào. Việc tìm kiếm và tối ưu hyperplane để có được max margin sẽ tạo ra 1 hyperplane có lề công bằng cho 2 phân lớp dữ liệu.

Do vậy, một cách trực quan ý tưởng của thuật toán SVMs là tìm một siêu phẳng(hyperplane) phân tách dữ liệu thành 2 phần tương ứng sao cho đạt được margin lớn nhất.

<h1> Soft margin </h1>

Như ở trên đã đề cập, ta đã đi qua ý tưởng chính của thuật toán SVMs, tuy nhiên trong một số trường hợp với ý tưởng như trên SVMs lại hoạt động không thực sự tốt và thậm chí không hoạt động được.

Có thể thấy trường hợp mà chúng ta đã đi qua ở trên là dữ liệu có thể phân tách tuyến tính và điểm dữ liệu thuộc các ljkkớp khác nhau nằm khá xa nhau. Vậy thì khi dữ liệu không thể phân tách tuyến tính hay có các điểm dữ liệu nhiễu nằm gần nhau như Hình 4 và Hình 5 thì sao ?

<img src="Assets/Pictures/Svm/5.png" class="singleImg">

<p class="textSingleImg">Hình  4: Có các điểm dữ liệu nhiễu nằm gần class khác hơn bình thường<p>

<img src="Assets/Pictures/Svm/9.png" class="singleImg">

<p class="textSingleImg">Hình  5: Dữ liệu gần tuyến tính, có một số các điểm dữ liệu nhiễu nằm ở nhầm class<p>

Về cơ bản để có thể giải quyết được các bài toán trên, svm chấp nhận việc hy sinh một vài điểm dữ liệu để có thể tạo ra một hyperplane nơi có margin lớn hơn và tổng quát hơn. Và khi đó các đường nét đứt tạo thành margin cùng với hyperplane gọi là soft margin và trong phạm vi margin của mỗi class được gọi là vùng không an toàn của class đó.

<div class="twoImgBlock">
    <img class="twoImg" src="Assets/Pictures/Svm/6.png" />
    <img class="twoImg" src="Assets/Pictures/Svm/8.png" />
</div>

<!-- <div class="textTwoImg"> -->
<!-- <p>Hình 6<p><p>Hình 7<p> -->
<!-- </div> -->

Khi hy sinh một số điểm dữ liệu ta sẽ tạo được hyperplane và các soft margin tổng quát hơn (Hình 6 và Hình 7). Tuy nhiên để có thể tìm ra được đường hyperplane hợp lý thì việc hy sinh các điểm dữ liệu phải được kiểm soát, nếu không sẽ tạo ra 1 margin cực lớn bằng cách hy sinh hết các điểm dữ liệu.

Khi đó, mục tiêu bài toàn sẽ là việc tối ưu tìm đường hyperplane (siêu phẳng) sao cho margin là lớn nhất mà số lượng điểm hy sinh là nhỏ nhất.<br/>

<h1> Kernel </h1>

Nhìn lại các lý thuyết cũng như những vấn đề mà chúng ta đã đi qua ở trên, có thể thấy, dữ liệu là loại dữ liệu tuyến tính có thể được phân loại bằng 1 đường thẳng trong không gian 2 chiều hay 1 siêu phẳng trong không gian nhiều chiều hơn. Câu hỏi đặt ra là nếu dữ liệu không phân biệt tuyến tính thì SVMs có sử dụng được KHÔNG ?

<br/><br/>Câu trả lời là CÓ. Thực thế là năm 1963 SVMs được giới thiệu bởi Vapnik và khi đó SVMs mặc định chỉ dùng cho việc phân loại tuyến tính, và tới 30 năm sau thì các công bố khác được đưa ra và có thể áp dụng SVMs cho dữ liệu không phân biệt tuyến tính. Có thể kể đến như là RBF (aka Radial Basic Function) hay Polynomial và một số các phương pháp khác.

<br/><br/>Các phương pháp trên được gọi là các kernel trong SVMs, về bản chất chúng là các hàm số biến đổi dữ liệu từ không gian ban đầu thành dữ liệu ở một không gian mới sao cho dữ liệu phi tuyến ở không gian cũ được chuyển thành dữ liệu tuyến tính hoặc gần như tuyến tính ở không gian mới. Ví dụ ở hình d bên dưới dữ liệu ở không gian 2 chiều đang là bộ dữ liệu phi tuyến. Nhưng khi chuyển chúng sang không gian 3 chiều với hàm $\mathbf z=|x^2 + y^2|$ thì đã trở thành bộ dữ liệu có thể xem là tuyến tính và có thể phân loại bằng 1 siêu phẳng.

<br/>
<br/>
<img src="Assets/Pictures/Svm/10.png" class="twoImg">
<img src="Assets/Pictures/Svm/11.png" class="twoImg">
<!-- <p class="textSingleImg">Hình  8<p> -->
<!-- <p class="textSingleImg">Hình  9<p> -->
<br/>
<img src="Assets/Pictures/Svm/12.png" class="singleImg">
<p class="textSingleImg">Hình 10: Cách thức SVM xử lý khi dữ liệu không tuyến tính<p><br/>

<h1> Tài Liệu Tham Khảo</h1>

<p>[1] Chang and Lin, "LIBSVM: A Library for Support Vector Machines", 2001<br/></p>
<p>[2] [series support vector machine - machinelearningcoban - Tiệp Vũ](https://machinelearningcoban.com/2017/04/09/smv/#-gioi-thieu)<br/></p>
<p>[3] [sklearn.svm.SVC](scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)<br/></p>

