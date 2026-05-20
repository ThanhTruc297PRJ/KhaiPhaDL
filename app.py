from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
import math
import common.extract_keyword

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["book_mining_db"]
books_collection = db["books"]
customers_collection = db["customers"]
orders_collection = db["orders"]


STOPWORDS = {
    "và", "là", "có", "cho", "của", "với", "trong", "một", "những",
    "các", "được", "này", "rất", "khá", "hơi", "thì", "mà", "để",
    "tôi", "sách", "nội", "dung", "phần", "chủ", "đề", "đọc", "học",
    "người", "nhiều", "còn", "nên", "về", "vào", "khi", "sau", "trên",
    "nếu", "như", "đã", "đang", "sẽ", "từ", "ra", "lại", "hơn", "nội dung"
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentiment_by_rating(rating):
    if rating >= 4:
        return "Tích cực"
    elif rating == 3:
        return "Trung lập"
    else:
        return "Tiêu cực"


def build_book_text(book):
    title = book.get("title", "")
    category = book.get("category", "")
    description = book.get("description", "")
    tags = " ".join(book.get("tags", []))
    reviews = " ".join([review.get("comment", "") for review in book.get("reviews", [])])

    full_text = f"{title} {category} {description} {tags} {reviews}"
    return clean_text(full_text)


def get_book_top_keywords(book, limit=12):
    word_counter = Counter()

    text_parts = [
        book.get("title", ""),
        book.get("category", ""),
        book.get("description", ""),
        " ".join(book.get("tags", [])),
        " ".join([review.get("comment", "") for review in book.get("reviews", [])])
    ]

    full_text = clean_text(" ".join(text_parts))
    common.extract_keyword.extract_text(full_text)
    words = full_text.split()

    for word in words:
        if len(word) > 2 and word not in STOPWORDS:
            word_counter[word] += 1
    # print(word_counter)
    return word_counter.most_common(limit)# chổ này là code cũ của em
    # tạm có thể trình bày mình sử dụng thư viện, a xem có thư viện khác ổn hơn không
    # return common.extract_keyword.extract_text(full_text)


def get_tfidf_recommendations(book_id, limit=5):
    books_data = list(books_collection.find({}, {"_id": 0}))

    if not books_data:
        return []

    current_index = None

    for index, book in enumerate(books_data):
        if book.get("book_id") == book_id:
            current_index = index
            break

    if current_index is None:
        return []

    corpus = [build_book_text(book) for book in books_data]

    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    current_scores = similarity_matrix[current_index]

    similar_items = []

    for index, score in enumerate(current_scores):
        if index != current_index:
            item = books_data[index].copy()
            item["similarity_score"] = round(float(score), 4)
            item["similarity_percent"] = round(float(score) * 100, 2)
            similar_items.append(item)

    similar_items = sorted(
        similar_items,
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    return similar_items[:limit]


def generate_next_book_id():
    books_data = list(books_collection.find({}, {"book_id": 1, "_id": 0}))

    max_number = 0

    for book in books_data:
        book_id = book.get("book_id", "")

        if book_id.startswith("B"):
            try:
                number = int(book_id.replace("B", ""))
                if number > max_number:
                    max_number = number
            except ValueError:
                pass

    next_number = max_number + 1
    return f"B{next_number:03d}"


@app.route("/")
def home():
    total_books = books_collection.count_documents({})
    total_reviews = 0
    categories = set()

    books = books_collection.find()

    for book in books:
        total_reviews += len(book.get("reviews", []))
        categories.add(book.get("category", "Không xác định"))

    return render_template(
        "index.html",
        total_books=total_books,
        total_reviews=total_reviews,
        total_categories=len(categories)
    )


@app.route("/books")
def books():
    keyword = request.args.get("keyword", "").strip()
    search_type = request.args.get("search_type", "all")
    category = request.args.get("category", "").strip()

    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1

    per_page = 10
    query = {}

    if category:
        query["category"] = category

    if keyword:
        regex = {"$regex": keyword, "$options": "i"}

        if search_type == "title":
            query["title"] = regex

        elif search_type == "author":
            query["author"] = regex

        elif search_type == "tag":
            query["tags"] = regex

        elif search_type == "review":
            query["reviews.comment"] = regex

        else:
            query["$or"] = [
                {"book_id": regex},
                {"title": regex},
                {"author": regex},
                {"category": regex},
                {"description": regex},
                {"tags": regex},
                {"reviews.comment": regex}
            ]

    total_books = books_collection.count_documents(query)
    total_pages = math.ceil(total_books / per_page) if total_books > 0 else 1

    if page < 1:
        page = 1

    if page > total_pages:
        page = total_pages

    skip_books = (page - 1) * per_page

    books_data = list(
        books_collection
        .find(query, {"_id": 0})
        .skip(skip_books)
        .limit(per_page)
    )

    start_index = skip_books + 1 if total_books > 0 else 0
    end_index = min(skip_books + per_page, total_books)

    categories = books_collection.distinct("category")

    return render_template(
        "books.html",
        books=books_data,
        page=page,
        total_pages=total_pages,
        total_books=total_books,
        per_page=per_page,
        start_index=start_index,
        end_index=end_index,
        categories=categories,
        keyword=keyword,
        search_type=search_type,
        selected_category=category
    )


@app.route("/book/<book_id>")
def book_detail(book_id):
    book = books_collection.find_one({"book_id": book_id}, {"_id": 0})

    if not book:
        return "Không tìm thấy sách", 404

    all_reviews = book.get("reviews", [])
    total_reviews = len(all_reviews)

    review_keyword = request.args.get("review_keyword", "").strip()
    rating_filter = request.args.get("rating", "").strip()

    reviews = all_reviews

    if review_keyword:
        keyword_lower = review_keyword.lower()
        reviews = [
            review for review in reviews
            if keyword_lower in review.get("comment", "").lower()
            or keyword_lower in review.get("user", "").lower()
        ]

    if rating_filter:
        try:
            rating_number = int(rating_filter)
            reviews = [
                review for review in reviews
                if int(review.get("rating", 0)) == rating_number
            ]
        except ValueError:
            pass

    filtered_review_count = len(reviews)

    avg_rating = 0
    sentiment_counter = {
        "Tích cực": 0,
        "Trung lập": 0,
        "Tiêu cực": 0
    }

    if total_reviews > 0:
        total_rating = sum(review.get("rating", 0) for review in all_reviews)
        avg_rating = round(total_rating / total_reviews, 2)

        for review in all_reviews:
            sentiment = get_sentiment_by_rating(review.get("rating", 0))
            sentiment_counter[sentiment] += 1

    top_keywords = get_book_top_keywords(book, limit=12)
    recommended_books = get_tfidf_recommendations(book_id, limit=5)

    return render_template(
        "book_detail.html",
        book=book,
        reviews=reviews,
        all_reviews=all_reviews,
        total_reviews=total_reviews,
        filtered_review_count=filtered_review_count,
        review_keyword=review_keyword,
        rating_filter=rating_filter,
        avg_rating=avg_rating,
        sentiment_counter=sentiment_counter,
        top_keywords=top_keywords,
        recommended_books=recommended_books,
        sentiment_labels=list(sentiment_counter.keys()),
        sentiment_values=list(sentiment_counter.values())
    )


@app.route("/add-book", methods=["GET", "POST"])
def add_book():
    categories = books_collection.distinct("category")

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        author = request.form.get("author", "").strip()
        category = request.form.get("category", "").strip()
        price = request.form.get("price", "").strip()
        description = request.form.get("description", "").strip()
        tags_text = request.form.get("tags", "").strip()

        if not title or not author or not category or not price:
            return render_template(
                "add_book.html",
                categories=categories,
                error="Vui lòng nhập đầy đủ tên sách, tác giả, thể loại và giá.",
                form_data=request.form
            )

        try:
            price = int(price)
        except ValueError:
            return render_template(
                "add_book.html",
                categories=categories,
                error="Giá sách phải là số.",
                form_data=request.form
            )

        tags = []
        if tags_text:
            tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]

        new_book = {
            "book_id": generate_next_book_id(),
            "title": title,
            "author": author,
            "category": category,
            "price": price,
            "description": description,
            "tags": tags,
            "reviews": [],
            "created_at": datetime.now().strftime("%d/%m/%Y")
        }

        books_collection.insert_one(new_book)

        return redirect(url_for("books"))

    return render_template(
        "add_book.html",
        categories=categories,
        error=None,
        form_data={}
    )


@app.route("/edit-book/<book_id>", methods=["GET", "POST"])
def edit_book(book_id):
    book = books_collection.find_one({"book_id": book_id}, {"_id": 0})

    if not book:
        return "Không tìm thấy sách", 404

    categories = books_collection.distinct("category")

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        author = request.form.get("author", "").strip()
        category = request.form.get("category", "").strip()
        price = request.form.get("price", "").strip()
        description = request.form.get("description", "").strip()
        tags_text = request.form.get("tags", "").strip()

        if not title or not author or not category or not price:
            return render_template(
                "edit_book.html",
                book=book,
                categories=categories,
                error="Vui lòng nhập đầy đủ tên sách, tác giả, thể loại và giá.",
                form_data=request.form
            )

        try:
            price = int(price)
        except ValueError:
            return render_template(
                "edit_book.html",
                book=book,
                categories=categories,
                error="Giá sách phải là số.",
                form_data=request.form
            )

        tags = []
        if tags_text:
            tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]

        books_collection.update_one(
            {"book_id": book_id},
            {
                "$set": {
                    "title": title,
                    "author": author,
                    "category": category,
                    "price": price,
                    "description": description,
                    "tags": tags
                }
            }
        )

        return redirect(url_for("books"))

    form_data = {
        "title": book.get("title", ""),
        "author": book.get("author", ""),
        "category": book.get("category", ""),
        "price": book.get("price", ""),
        "tags": ", ".join(book.get("tags", [])),
        "description": book.get("description", "")
    }

    return render_template(
        "edit_book.html",
        book=book,
        categories=categories,
        error=None,
        form_data=form_data
    )


@app.route("/delete-book/<book_id>", methods=["POST"])
def delete_book(book_id):
    books_collection.delete_one({"book_id": book_id})
    return redirect(url_for("books"))


@app.route("/search", methods=["GET"])
def search():
    return redirect(url_for(
        "books",
        keyword=request.args.get("keyword", ""),
        search_type=request.args.get("search_type", "all"),
        category=request.args.get("category", "")
    ))


@app.route("/analytics")
def analytics():
    books_data = list(books_collection.find({}, {"_id": 0}))

    total_books = len(books_data)
    total_reviews = 0

    category_counter = Counter()
    sentiment_counter = Counter()
    word_counter = Counter()
    rating_by_category = defaultdict(list)

    for book in books_data:
        category = book.get("category", "Không xác định")
        category_counter[category] += 1

        reviews = book.get("reviews", [])
        total_reviews += len(reviews)

        for review in reviews:
            rating = review.get("rating", 0)
            comment = review.get("comment", "")

            sentiment = get_sentiment_by_rating(rating)
            sentiment_counter[sentiment] += 1

            rating_by_category[category].append(rating)

            cleaned = clean_text(comment)
            words = cleaned.split()

            for word in words:
                if len(word) > 2 and word not in STOPWORDS:
                    word_counter[word] += 1

    top_keywords = word_counter.most_common(15)

    avg_rating_by_category = []

    for category, ratings in rating_by_category.items():
        avg_rating = sum(ratings) / len(ratings)
        avg_rating_by_category.append({
            "category": category,
            "avg_rating": round(avg_rating, 2)
        })

    avg_rating_by_category = sorted(
        avg_rating_by_category,
        key=lambda x: x["avg_rating"],
        reverse=True
    )

    return render_template(
        "analytics.html",
        total_books=total_books,
        total_reviews=total_reviews,
        total_categories=len(category_counter),

        category_labels=list(category_counter.keys()),
        category_values=list(category_counter.values()),

        sentiment_labels=list(sentiment_counter.keys()),
        sentiment_values=list(sentiment_counter.values()),

        top_keywords=top_keywords,
        avg_rating_by_category=avg_rating_by_category
    )


def generate_next_customer_id():
    customers_data = list(customers_collection.find({}, {"customer_id": 1, "_id": 0}))
    max_number = 0

    for customer in customers_data:
        customer_id = customer.get("customer_id", "")
        if customer_id.startswith("C"):
            try:
                number = int(customer_id.replace("C", ""))
                max_number = max(max_number, number)
            except ValueError:
                pass

    return f"C{max_number + 1:03d}"


def generate_next_order_id():
    orders_data = list(orders_collection.find({}, {"order_id": 1, "_id": 0}))
    max_number = 0

    for order in orders_data:
        order_id = order.get("order_id", "")
        if order_id.startswith("O"):
            try:
                number = int(order_id.replace("O", ""))
                max_number = max(max_number, number)
            except ValueError:
                pass

    return f"O{max_number + 1:03d}"


def get_order_status_list():
    return ["Chờ xử lý", "Đã xác nhận", "Đang giao", "Hoàn thành", "Đã hủy"]


@app.route("/customers")
def customers():
    keyword = request.args.get("keyword", "").strip()

    query = {}
    if keyword:
        regex = {"$regex": keyword, "$options": "i"}
        query["$or"] = [
            {"customer_id": regex},
            {"name": regex},
            {"email": regex},
            {"phone": regex},
            {"address": regex}
        ]

    customers_data = list(customers_collection.find(query, {"_id": 0}).sort("customer_id", 1))

    return render_template(
        "customers.html",
        customers=customers_data,
        keyword=keyword,
        total_customers=len(customers_data)
    )


@app.route("/add-customer", methods=["GET", "POST"])
def add_customer():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()

        if not name or not phone:
            return render_template(
                "add_customer.html",
                error="Vui lòng nhập tên khách hàng và số điện thoại.",
                form_data=request.form
            )

        new_customer = {
            "customer_id": generate_next_customer_id(),
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
            "created_at": datetime.now().strftime("%d/%m/%Y")
        }

        customers_collection.insert_one(new_customer)
        return redirect(url_for("customers"))

    return render_template("add_customer.html", error=None, form_data={})


@app.route("/edit-customer/<customer_id>", methods=["GET", "POST"])
def edit_customer(customer_id):
    customer = customers_collection.find_one({"customer_id": customer_id}, {"_id": 0})

    if not customer:
        return "Không tìm thấy khách hàng", 404

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()

        if not name or not phone:
            return render_template(
                "edit_customer.html",
                customer=customer,
                error="Vui lòng nhập tên khách hàng và số điện thoại.",
                form_data=request.form
            )

        customers_collection.update_one(
            {"customer_id": customer_id},
            {"$set": {
                "name": name,
                "email": email,
                "phone": phone,
                "address": address
            }}
        )

        return redirect(url_for("customers"))

    return render_template(
        "edit_customer.html",
        customer=customer,
        error=None,
        form_data=customer
    )


@app.route("/delete-customer/<customer_id>", methods=["POST"])
def delete_customer(customer_id):
    orders_count = orders_collection.count_documents({"customer_id": customer_id})

    if orders_count > 0:
        return "Không thể xóa khách hàng vì đã có đơn hàng liên quan.", 400

    customers_collection.delete_one({"customer_id": customer_id})
    return redirect(url_for("customers"))


@app.route("/orders")
def orders():
    keyword = request.args.get("keyword", "").strip()
    status = request.args.get("status", "").strip()

    query = {}
    if status:
        query["status"] = status

    if keyword:
        regex = {"$regex": keyword, "$options": "i"}
        query["$or"] = [
            {"order_id": regex},
            {"customer_id": regex},
            {"customer_name": regex},
            {"items.book_id": regex},
            {"items.title": regex}
        ]

    orders_data = list(orders_collection.find(query, {"_id": 0}).sort("created_at", -1))
    print(orders_data)
    print("status =", status)
    print("query =", query)
    print("total all =", orders_collection.count_documents({}))
    print("total filtered =", orders_collection.count_documents(query))
    return render_template(
        "orders.html",
        orders=orders_data,
        keyword=keyword,
        selected_status=status,
        status_list=get_order_status_list(),
        total_orders=len(orders_data)
    )


@app.route("/add-order", methods=["GET", "POST"])
def add_order():
    customers_data = list(customers_collection.find({}, {"_id": 0}).sort("customer_id", 1))
    books_data = list(books_collection.find({}, {"_id": 0}).sort("book_id", 1))
    status_list = get_order_status_list()

    if request.method == "POST":
        customer_id = request.form.get("customer_id", "").strip()
        book_id = request.form.get("book_id", "").strip()
        quantity_text = request.form.get("quantity", "1").strip()
        status = request.form.get("status", "Chờ xử lý").strip()
        note = request.form.get("note", "").strip()

        if not customer_id or not book_id:
            return render_template(
                "add_order.html",
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                error="Vui lòng chọn khách hàng và sách.",
                form_data=request.form
            )

        try:
            quantity = int(quantity_text)
            if quantity <= 0:
                raise ValueError
        except ValueError:
            return render_template(
                "add_order.html",
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                error="Số lượng phải là số nguyên lớn hơn 0.",
                form_data=request.form
            )

        customer = customers_collection.find_one({"customer_id": customer_id}, {"_id": 0})
        book = books_collection.find_one({"book_id": book_id}, {"_id": 0})

        if not customer or not book:
            return render_template(
                "add_order.html",
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                error="Khách hàng hoặc sách không tồn tại.",
                form_data=request.form
            )

        price = int(book.get("price", 0))
        total_amount = price * quantity

        new_order = {
            "order_id": generate_next_order_id(),
            "customer_id": customer.get("customer_id"),
            "customer_name": customer.get("name"),
            "customer_phone": customer.get("phone"),
            "items": [
                {
                    "book_id": book.get("book_id"),
                    "title": book.get("title"),
                    "price": price,
                    "quantity": quantity,
                    "amount": total_amount
                }
            ],
            "total_amount": total_amount,
            "status": status,
            "note": note,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        orders_collection.insert_one(new_order)
        return redirect(url_for("orders"))

    return render_template(
        "add_order.html",
        customers=customers_data,
        books=books_data,
        status_list=status_list,
        error=None,
        form_data={}
    )


@app.route("/order/<order_id>")
def order_detail(order_id):
    order = orders_collection.find_one({"order_id": order_id}, {"_id": 0})

    if not order:
        return "Không tìm thấy đơn hàng", 404

    return render_template("order_detail.html", order=order)


@app.route("/edit-order/<order_id>", methods=["GET", "POST"])
def edit_order(order_id):
    order = orders_collection.find_one({"order_id": order_id}, {"_id": 0})

    if not order:
        return "Không tìm thấy đơn hàng", 404

    status_list = get_order_status_list()

    if request.method == "POST":
        status = request.form.get("status", "").strip()
        note = request.form.get("note", "").strip()

        if status not in status_list:
            return render_template(
                "edit_order.html",
                order=order,
                status_list=status_list,
                error="Trạng thái đơn hàng không hợp lệ.",
                form_data=request.form
            )

        orders_collection.update_one(
            {"order_id": order_id},
            {"$set": {
                "status": status,
                "note": note,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }}
        )

        return redirect(url_for("orders"))

    return render_template(
        "edit_order.html",
        order=order,
        status_list=status_list,
        error=None,
        form_data=order
    )


@app.route("/delete-order/<order_id>", methods=["POST"])
def delete_order(order_id):
    orders_collection.delete_one({"order_id": order_id})
    return redirect(url_for("orders"))


if __name__ == "__main__":
    app.run(debug=True)