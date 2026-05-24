# Book Mining Project - MVC Version

Source đã được chuyển từ `app.py` monolithic sang mô hình MVC đơn giản cho Flask.

## Cấu trúc mới

```text
app.py                          # Entry point chạy Flask
legacy_app_monolith.py           # Bản app.py cũ để đối chiếu
app_mvc/
  __init__.py                    # create_app(), đăng ký routes
  config.py                      # Cấu hình MongoDB
  extensions.py                  # Khởi tạo MongoClient và collections
  models/
    collections.py               # Model layer: khai báo collection MongoDB
  services/
    data_mining_service.py       # Business logic, TF-IDF, RFM, KMeans, luật kết hợp
  controllers/
    dashboard_controller.py      # Trang chủ
    book_controller.py           # Quản lý sách, search, gợi ý sách
    analytics_controller.py      # Analytics, khai phá dữ liệu
    customer_controller.py       # Quản lý khách hàng
    order_controller.py          # Quản lý đơn hàng, customer_orders
    review_controller.py         # customer_reviews
templates/                       # View layer
static/                          # CSS/JS/assets
```

## Cách chạy

```bash
pip install -r requirements.txt
python app.py
```

## Cấu hình MongoDB

Mặc định hệ thống dùng:

```text
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=book_store
```

Có thể đổi bằng biến môi trường:

```bash
set MONGO_URI=mongodb://localhost:27017/
set MONGO_DB_NAME=book_store
python app.py
```

## Ghi chú

- Endpoint cũ được giữ nguyên, ví dụ `/books`, `/customers`, `/orders`, `/analytics`.
- Tên endpoint trong `url_for()` vẫn giữ nguyên, ví dụ `url_for("books")`, `url_for("analytics")`.
- Các template hiện tại không cần đổi route.
- File `legacy_app_monolith.py` chỉ để tham khảo, không còn là file chạy chính.
- Các file font binary trong `static/fonts` không được đóng gói lại trong bản zip trả về. Nếu project hiện tại của bạn đã có thư mục này thì giữ nguyên thư mục cũ.
