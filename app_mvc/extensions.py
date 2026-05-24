from pymongo import MongoClient

client = None
db = None

books_collection = None
customers_collection = None
orders_collection = None
customer_orders_collection = None
customer_reviews_collection = None


def init_db(app):
    """Khởi tạo kết nối MongoDB và các collection dùng trong hệ thống."""
    global client
    global db
    global books_collection
    global customers_collection
    global orders_collection
    global customer_orders_collection
    global customer_reviews_collection

    client = MongoClient(app.config["MONGO_URI"])
    db = client[app.config["MONGO_DB_NAME"]]

    books_collection = db["books"]
    customers_collection = db["customers"]
    orders_collection = db["orders"]
    customer_orders_collection = db["customer_orders"]
    customer_reviews_collection = db["customer_reviews"]
