from pymongo import MongoClient

# Kết nối MongoDB local
client = MongoClient("mongodb://localhost:27017/")

# Chọn database và collection
db = client["book_mining_db"]
books_collection = db["books"]

# Đếm số sách
total_books = books_collection.count_documents({})

print("Kết nối MongoDB thành công!")
print("Tổng số sách trong collection books:", total_books)

# Lấy thử 1 sách
book = books_collection.find_one()

if book:
    print("\nThông tin sách đầu tiên:")
    print("Mã sách:", book.get("book_id"))
    print("Tên sách:", book.get("title"))
    print("Tác giả:", book.get("author"))
    print("Thể loại:", book.get("category"))
    print("Giá:", book.get("price"))
    print("Số đánh giá:", len(book.get("reviews", [])))
else:
    print("Chưa có dữ liệu trong collection books.")