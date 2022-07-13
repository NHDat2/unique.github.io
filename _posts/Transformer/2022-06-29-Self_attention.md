---
title: Self-Attention
layout: default
excerpt: Nói không ngoa thì Self-Attention là linh hồn của kiến trúc transformer. Mọi thứ mới mẻ trong kiến trúc transformer được công bố tại thời điểm bấy giờ được sinh ra và xoay quanh Self-Attention ...
tags: ["NLP", "Transformer", "Self-Attention", "Attention"]
---

<style>
  .img {
    width: 400px;
    height: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;

  }
  #scaleImg {
    width: 700px;
    height: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
</style>

# Giới Thiệu

Ta đã đi qua tổng quan về ý tưởng của transformer, ta cũng đã bắt đầu đi sâu vào trong kiến trúc để hiểu cơ chế hoạt động với việc tìm hiểu Positional Encoding. Trong bài viết này ta sẽ đi tiếp tới cơ chế tiếp theo, có thể nói là cơ chế quan trọng nhất trong kiến trúc Transformer là Self-Attention.

Nói không ngoa thì Self-Attention là linh hồn của kiến trúc transformer. Mọi thứ mới mẻ trong kiến trúc transformer được công bố tại thời điểm bấy giờ được sinh ra và xoay quanh Self-Attention.

<img class="img" src="Assets/Pictures/Transformer/SelfAttention/sa_pos_in_transformer.png"/>
<p>Hình 1</p>

# Self-Attention
## Ý Tưởng

Ví dụ, khi ta có một câu đầu vào với n token $t_{1},\ t_{2},\ t_{3},...,\ t_{n}$. Khi đó, một cách dễ hiểu self-attention là cơ chế giúp token $t_{i}$ chú ý tới các token còn lại trong câu, để có thể giúp token $t_{i}$ nắm được mối quan hệ của nó với các token còn lại về mặt cấu trúc câu, mặt ngữ nghĩa, ..v.v. là như thế nào.

## Cơ Chế Hoạt Động

Trong bài báo gốc, self-attention được nhóm tác giả giới thiệu với một loạt các khái niệm khác liên quan như **Scaled dot-product attention** hay **Multi-head attention**. Trong đó **Scaled dot-product attention** là một cơ chế self-attention và **Multi-head attention** là việc nối nhiều **scaled dot-product attention** lại với nhau và đưa qua một lớp **fully connected**.

<img id="scaleImg" src="Assets/Pictures/Transformer/SelfAttention/sa_architecture.png"/>
<p>Hình</p>

### Self-Attention
#### Ma Trận Query, Key, Value (Q, K, V Matrix)

Trên Hình , có thể thấy Q, K, V cũng là 3 tham số được giới thiệu trong self-attention. Vậy Q, K, V là gì và đóng vai trò như thế nào trong self-attention.

Q, K, V là 3 vector đại diện biểu diễn cho từng token trong câu được tạo ra bằng cách nhân ma trận biểu diễn các token đầu vào với 3 ma trận học tương ứng là $W_{Q}\ W_{K}\ W_{V}$.

Trong đó:

* **Q**: Query vector dùng để chứa thông tin của câu tìm kiếm (ví dụ như chứa các thông tin của token đang cần xem xét).
* **K**: Key vector dùng để biểu diễn thông tin so sánh giữa các token trong câu với token đang được query.
* **V**: value vector biểu diễn nội dung của các token.

Để dễ hình dung hơn, ta có thể coi Q, K, V như việc tìm kiếm sách trong thư viện. Giả sử trong thư viện có 10000 quyển sách tương ứng 10000 token thì self-attention khi áp dụng vào đây sẽ như thế nào. Với mỗi một quyển sách ta sẽ có 3 vector Q, K, V ứng với đó là 3 ma trận trọng số $W_{Q}\ W_{K}\ W_{V}$. Thì khi nhìn vào một quyển sách làm sao để thủ thư có thể tìm được quyển sác đó trong thư viện.

* Thì việc đầu tiên thủ thư cần làm là nhìn vào đặc điểm của quyển sách cần tìm để xem nó màu gì, thể loại gì ..v.v. để có thể biết được nó nằm ở vị trí nào trong thư viện. Vì có rất nhiều sách được hỏi mượn thường xuyên cho nên thủ thư phải nhận biết được đặc điểm của nhiều quyển sách khác nhau, thì việc nhận biết được các đặc điểm đó của các quyển sách cần tìm chính là việc học ma trận $W_{Q}$
* Sau khi có thông tin đặc điểm của quyển sách cần tìm thì thủ thư sẽ nhận định được rằng quyển sách đó sẽ nằm ở kệ nào hàng bao nhiêu, và chỉ tập trung tìm trong khu vực đó và không quan tâm nhiều tới các khu vực còn lại. Thì việc thủ thư nhận biết được từng khu vực khác nhau trong thư viện cũng cần một quá trình học tập và quá trình đó là quá trình học ma trận $W_{K}$
* Và khi có khu vực tìm kiếm cụ thể thì thủ thư sẽ tìm kiếm trong khu đó và đưa ra quyển sách như mong muốn. Việc thủ thư đưa ra được quyển sách mong muốn cũng cần phải học để nhận biết được đâu là quyển sách mong muốn quá trình đó là quá trình học ma trận $W_{V}$.

Về mặt trực quan, K như kiểu là cầu nối giữa Q (cái ta đang tìm kiếm) và V (thứ chúng ta thực sự nhận được).

còn phần d căn k chưa đọc

#### Multi-Head Attention

Về cơ bản Multi-head attention có thể được định nghĩa là việc sử dụng nhiều lớp self-attention rồi nối chúng lại với nhau, sau đó nhân với một ma trận trọng số $W_{O}$

Thông thường, để hiểu được vai trò của một từ trong câu, nó sẽ cần được nhìn ở nhiều khía cạnh khác nhau như cấu trúc câu, ngữ nghĩa, ..v.v. Thì thay vì chỉ sử dụng 1 self-attention hay còn gọi là 1 head thì nhóm tác giả sử dụng nhiều self-attention hay multi-head để mỗi head sẽ tập trung vào học một khía cạnh khác nhau.

## Attention Và Self-Attention

Trước đó, với mô hình seq2seq, ở đâu đó ta đã biết qua một số cơ chế **attention** khác nhau để giúp seq2seq model có sự chú ý tới các token trong câu có thể đến như **Bahdanau** và **Luong Attention**.

Sau đây ta sẽ đi qua một nhận định giữa Attention và Self-Attention theo góc nhìn và hiểu biết của mình.

* Với các neural network ta sẽ đưa input (câu đầu vào) qua các layer và các activate fuction, và trong các mạng RNN và biến thể của nó thì ta sẽ có thêm state của các layer nữa. Thì khi đó Attention được áp dụng sẽ nhận đầu vào là các input đã đi qua các layer và các activate fuction. Trong khi đó Self-Attention sẽ thực hiện attention tại chính câu đầu vào ở mỗi layer có sử dụng nó.
* Attention thường được áp dụng để giúp bộ phận decoder có thể có thêm thông tin về phía encoder. Self-Attention hoạt động độc lập trên cả 2 bộ phân encoder và decoder mà không có sự kết nối nào ở đây. Do hoạt động độc lập nên có cả các biến thể của Transformer được sinh ra khi chỉ dùng Encoder hoặc Decoder (BERT là một ví dụ điển hình).

# Tài Liệu Tham Khảo
