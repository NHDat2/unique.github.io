---
title: The Rest of Transformer
layout: default
excerpt: ...
tags: ["NLP", "Transformer"]
---

# Giới Thiệu

Ở các phần trước đó ta đã đi qua các phần kiến thức trọng yếu trong kiến trúc Transformer. Trong phần này, ta sẽ cùng nhau tìm hiểu nốt một số kiến thức nhỏ khác được thêm vào trong transformer như các phụ gia trong món ăn và cùng nhau đi qua một số nhận định mà mình survey được khi tìm hiểu về Transformer.

# Residual Connections

Vector đầu ra của Multi-Head Attention được cộng thêm với positional embedding, được gọi là residual connection. Và sau đó đầu ra của residual connection sẽ được chuẩn hóa thông qua layer Normalization rồi được đưa qua lớp Feed Forward. Bao gồm 2 linear layer và kẹp ở giữa là một activate function RELU.

Residual connections và feed forward là 2 thành phần nhỏ nhưng góp phần quan trọng trong quá trình train của model. Như đã biết, self-attention là cơ chế giúp các token nắm bắt mối quan hệ của chính nó với các token khác trong câu. Tuy nhiên, ở đó thì self-attention không chứa các thông tin về vị trí và cho phép luồng thông tin tùy ý đi qua mạng. Do vậy, residual connection đóng một vai trò để luôn nhắc nhở cho self-attention rằng "các token được biểu diễn theo thứ tự như vậy là nó có ý nghĩa của nó đấy nên đừng có đảo loạn các token lên nhé".

Sau đó, đầu ra của residual connection được đưa vào layer feed forward với các kết nối đầy đủ để mang lại khả năng biểu diễn phong phú hơn cho các features với 2 lớp linear và kẹp ở giữa là hàm relu. Để có thể ổn định mạng, trước khi đưa vào layer feed forward, đầu ra của residual được chuẩn hóa thông qua layer normalization.

# Một Số Nhận Định Và Các Nghiên Cứu Liên Quan


