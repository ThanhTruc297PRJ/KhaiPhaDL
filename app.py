from flask import Flask, render_template, request, redirect, url_for, jsonify
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
db = client["book_store"]
books_collection = db["books"]
customers_collection = db["customers"]
orders_collection = db["orders"]
customer_orders_collection = db["customer_orders"]


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

    # Bổ sung cho dashboard analytics
    author_counter = Counter()
    book_review_list = []

    for book in books_data:
        category = book.get("category", "Không xác định")
        category_counter[category] += 1

        # 1) Thống kê số lượng sách theo tác giả
        author = book.get("author", "Không xác định")
        if isinstance(author, str):
            authors = [a.strip() for a in author.split(",") if a.strip()]
            if not authors:
                authors = ["Không xác định"]
        elif isinstance(author, list):
            authors = [str(a).strip() for a in author if str(a).strip()]
            if not authors:
                authors = ["Không xác định"]
        else:
            authors = ["Không xác định"]

        for item in authors:
            author_counter[item] += 1

        reviews = book.get("reviews", [])
        review_count = len(reviews)
        total_reviews += review_count

        # 2) Top 10 sách có nhiều review nhất
        book_review_list.append({
            "book_id": book.get("book_id", ""),
            "title": book.get("title", "Không có tên"),
            "review_count": review_count
        })

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
    avg_rating_by_category = sorted(avg_rating_by_category, key=lambda x: x["avg_rating"], reverse=True)

    # Nếu muốn chart tác giả gọn hơn, đổi dòng dưới thành [:10]
    author_stats = sorted(author_counter.items(), key=lambda x: x[1], reverse=True)

    top_reviewed_books = sorted(
        book_review_list,
        key=lambda x: x["review_count"],
        reverse=True
    )[:10]

    # Nội dung khai phá dữ liệu đơn hàng được gộp trực tiếp vào trang analytics.
    try:
        min_support = float(request.args.get("min_support", 0.03))
    except ValueError:
        min_support = 0.03

    try:
        min_confidence = float(request.args.get("min_confidence", 0.2))
    except ValueError:
        min_confidence = 0.2

    association_rules = mine_association_rules(
        min_support=min_support,
        min_confidence=min_confidence,
        limit=20
    )
    customer_segments, segment_counter = get_customer_segments()
    sales_summary = get_sales_mining_summary()

    total_orders = orders_collection.count_documents({})
    total_customers = customers_collection.count_documents({})

    return render_template(
        "analytics.html",
        total_books=total_books,
        total_reviews=total_reviews,
        total_categories=len(category_counter),
        total_customers=total_customers,
        total_orders=total_orders,

        category_labels=list(category_counter.keys()),
        category_values=list(category_counter.values()),
        sentiment_labels=list(sentiment_counter.keys()),
        sentiment_values=list(sentiment_counter.values()),

        top_keywords=top_keywords,
        avg_rating_by_category=avg_rating_by_category,

        author_labels=[item[0] for item in author_stats],
        author_values=[item[1] for item in author_stats],
        top_review_book_labels=[item["title"] for item in top_reviewed_books],
        top_review_book_values=[item["review_count"] for item in top_reviewed_books],

        min_support=min_support,
        min_confidence=min_confidence,
        association_rules=association_rules,
        customer_segments=customer_segments[:20],
        segment_labels=list(segment_counter.keys()),
        segment_values=list(segment_counter.values()),
        top_selling_books=sales_summary["top_selling_books"],
        top_revenue_books=sales_summary["top_revenue_books"],
        status_labels=list(sales_summary["status_counter"].keys()),
        status_values=list(sales_summary["status_counter"].values()),
        monthly_labels=[item[0] for item in sales_summary["monthly_revenue"]],
        monthly_values=[item[1] for item in sales_summary["monthly_revenue"]]
    )

@app.route("/api/association-rules")
def api_association_rules():
    """Trả kết quả luật kết hợp dạng JSON để cập nhật bảng mà không reload trang."""
    try:
        min_support = float(request.args.get("min_support", 0.03))
    except ValueError:
        min_support = 0.03

    try:
        min_confidence = float(request.args.get("min_confidence", 0.2))
    except ValueError:
        min_confidence = 0.2

    rules = mine_association_rules(
        min_support=min_support,
        min_confidence=min_confidence,
        limit=20
    )
    return jsonify({"rules": rules})


@app.route("/api/books-search")
def api_books_search():
    """API tìm kiếm sách dùng fetch/AJAX, không reload trang."""
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
    page = max(1, min(page, total_pages))
    skip_books = (page - 1) * per_page

    books_data = list(
        books_collection
        .find(query, {"_id": 0})
        .skip(skip_books)
        .limit(per_page)
    )

    for book in books_data:
        book["review_count"] = len(book.get("reviews", []))
        book["price_text"] = f'{int(book.get("price", 0) or 0):,} VNĐ'
        book["created_text"] = book.get("created_at") or book.get("ngay_nhap") or "20/05/2026"
        book.pop("reviews", None)

    return jsonify({
        "books": books_data,
        "page": page,
        "total_pages": total_pages,
        "total_books": total_books,
        "start_index": skip_books + 1 if total_books > 0 else 0,
        "end_index": min(skip_books + per_page, total_books),
        "keyword": keyword,
        "search_type": search_type,
        "category": category
    })

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



def parse_order_datetime(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(str(value), "%d/%m/%Y")
        except Exception:
            return None


def mine_association_rules(min_support=0.03, min_confidence=0.2, limit=20):
    """Khai phá luật kết hợp đơn giản từ các sách xuất hiện cùng đơn hàng."""
    orders_data = list(orders_collection.find({}, {"_id": 0, "items": 1}))
    total_orders = len(orders_data)

    if total_orders == 0:
        return []

    item_counter = Counter()
    pair_counter = Counter()

    for order in orders_data:
        book_ids = sorted({item.get("book_id") for item in order.get("items", []) if item.get("book_id")})

        for book_id in book_ids:
            item_counter[book_id] += 1

        for i in range(len(book_ids)):
            for j in range(i + 1, len(book_ids)):
                pair_counter[(book_ids[i], book_ids[j])] += 1

    book_title_map = {}
    for book in books_collection.find({}, {"_id": 0, "book_id": 1, "title": 1}):
        book_title_map[book.get("book_id")] = book.get("title", book.get("book_id"))

    rules = []

    for (book_a, book_b), pair_count in pair_counter.items():
        support = pair_count / total_orders

        if support < min_support:
            continue

        confidence_ab = pair_count / item_counter[book_a] if item_counter[book_a] else 0
        confidence_ba = pair_count / item_counter[book_b] if item_counter[book_b] else 0
        support_b = item_counter[book_b] / total_orders if total_orders else 0
        support_a = item_counter[book_a] / total_orders if total_orders else 0

        lift_ab = confidence_ab / support_b if support_b else 0
        lift_ba = confidence_ba / support_a if support_a else 0

        if confidence_ab >= min_confidence:
            rules.append({
                "antecedent": book_a,
                "antecedent_title": book_title_map.get(book_a, book_a),
                "consequent": book_b,
                "consequent_title": book_title_map.get(book_b, book_b),
                "support": round(support, 3),
                "confidence": round(confidence_ab, 3),
                "lift": round(lift_ab, 3),
                "count": pair_count
            })

        if confidence_ba >= min_confidence:
            rules.append({
                "antecedent": book_b,
                "antecedent_title": book_title_map.get(book_b, book_b),
                "consequent": book_a,
                "consequent_title": book_title_map.get(book_a, book_a),
                "support": round(support, 3),
                "confidence": round(confidence_ba, 3),
                "lift": round(lift_ba, 3),
                "count": pair_count
            })

    rules = sorted(rules, key=lambda x: (x["lift"], x["confidence"], x["support"]), reverse=True)
    return rules[:limit]


def get_customer_segments():
    """Phân khúc khách hàng theo RFM: Recency, Frequency, Monetary."""
    orders_data = list(orders_collection.find({}, {"_id": 0}))

    if not orders_data:
        return [], Counter()

    max_date = None
    for order in orders_data:
        dt = parse_order_datetime(order.get("created_at"))
        if dt and (max_date is None or dt > max_date):
            max_date = dt

    if max_date is None:
        max_date = datetime.now()

    customer_stats = defaultdict(lambda: {
        "customer_id": "",
        "customer_name": "",
        "frequency": 0,
        "monetary": 0,
        "last_order_date": None
    })

    for order in orders_data:
        customer_id = order.get("customer_id", "Không xác định")
        stats = customer_stats[customer_id]
        stats["customer_id"] = customer_id
        stats["customer_name"] = order.get("customer_name", "Không xác định")
        stats["frequency"] += 1
        stats["monetary"] += int(order.get("total_amount", 0) or 0)

        dt = parse_order_datetime(order.get("created_at"))
        if dt and (stats["last_order_date"] is None or dt > stats["last_order_date"]):
            stats["last_order_date"] = dt

    segments = []
    segment_counter = Counter()

    for stats in customer_stats.values():
        last_order_date = stats["last_order_date"]
        recency = (max_date - last_order_date).days if last_order_date else 999
        frequency = stats["frequency"]
        monetary = stats["monetary"]

        if frequency >= 4 and monetary >= 1000000 and recency <= 60:
            segment = "Khách hàng giá trị cao"
        elif frequency >= 3:
            segment = "Khách hàng trung thành"
        elif recency <= 30:
            segment = "Khách hàng mới/gần đây"
        elif monetary >= 700000:
            segment = "Khách hàng tiềm năng"
        else:
            segment = "Khách hàng phổ thông"

        segment_counter[segment] += 1
        segments.append({
            "customer_id": stats["customer_id"],
            "customer_name": stats["customer_name"],
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "last_order_date": last_order_date.strftime("%Y-%m-%d") if last_order_date else "",
            "segment": segment
        })

    segments = sorted(segments, key=lambda x: (x["monetary"], x["frequency"]), reverse=True)
    return segments, segment_counter


def get_sales_mining_summary():
    orders_data = list(orders_collection.find({}, {"_id": 0}))
    book_counter = Counter()
    book_revenue = Counter()
    status_counter = Counter()
    monthly_revenue = Counter()

    for order in orders_data:
        status_counter[order.get("status", "Không xác định")] += 1

        dt = parse_order_datetime(order.get("created_at"))
        if dt:
            monthly_revenue[dt.strftime("%Y-%m")] += int(order.get("total_amount", 0) or 0)

        for item in order.get("items", []):
            title = item.get("title", item.get("book_id", "Không xác định"))
            qty = int(item.get("quantity", 0) or 0)
            amount = int(item.get("amount", 0) or 0)
            book_counter[title] += qty
            book_revenue[title] += amount

    return {
        "top_selling_books": book_counter.most_common(10),
        "top_revenue_books": book_revenue.most_common(10),
        "status_counter": status_counter,
        "monthly_revenue": sorted(monthly_revenue.items())
    }


def get_order_status_list():
    return ["Chờ xử lý", "Đã xác nhận", "Đang giao", "Hoàn thành", "Đã hủy"]


def get_int_arg(name, default=1, min_value=1):
    """Đọc tham số số nguyên từ query string và chống lỗi dữ liệu không hợp lệ."""
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default

    if value < min_value:
        value = min_value

    return value


def get_pagination_data(collection, query, sort_field, sort_direction=-1, default_per_page=10):
    """
    Hàm phân trang dùng chung cho orders và customer_orders.
    Trả về danh sách dữ liệu trang hiện tại và metadata phân trang.
    """
    page = get_int_arg("page", 1)
    per_page = get_int_arg("per_page", default_per_page)

    allowed_per_pages = [5, 10, 20, 50]
    if per_page not in allowed_per_pages:
        per_page = default_per_page

    total_items = collection.count_documents(query)
    total_pages = math.ceil(total_items / per_page) if total_items > 0 else 1

    if page > total_pages:
        page = total_pages

    skip_items = (page - 1) * per_page

    data = list(
        collection
        .find(query, {"_id": 0})
        .sort(sort_field, sort_direction)
        .skip(skip_items)
        .limit(per_page)
    )

    start_index = skip_items + 1 if total_items > 0 else 0
    end_index = min(skip_items + per_page, total_items)

    return data, {
        "page": page,
        "per_page": per_page,
        "total_items": total_items,
        "total_pages": total_pages,
        "start_index": start_index,
        "end_index": end_index,
        "allowed_per_pages": allowed_per_pages
    }


def build_customer_order_doc(order):
    """
    Tạo document tổng hợp cho collection customer_orders từ document orders.
    Collection này giúp hiển thị rõ quan hệ khách hàng - đơn hàng.
    """
    customer_id = order.get("customer_id")

    customer = customers_collection.find_one(
        {"customer_id": customer_id},
        {"_id": 0}
    ) or {}

    items = order.get("items", [])
    total_items = sum(int(item.get("quantity", 0) or 0) for item in items)

    return {
        "customer_order_id": f"CO_{order.get('order_id', '')}",
        "customer_id": customer_id,
        "customer_name": customer.get("name") or order.get("customer_name", ""),
        "customer_phone": customer.get("phone") or order.get("customer_phone", ""),
        "customer_email": customer.get("email", ""),
        "customer_address": customer.get("address", ""),
        "customer_segment": customer.get("segment_demo", ""),
        "order_id": order.get("order_id"),
        "order_status": order.get("status", ""),
        "order_note": order.get("note", ""),
        "order_date": order.get("created_at", ""),
        "created_at": order.get("created_at", ""),
        "updated_at": order.get("updated_at", ""),
        "total_amount": int(order.get("total_amount", 0) or 0),
        "total_items": total_items,
        "book_count": len(items),
        "items": items
    }


def upsert_customer_order(order):
    """Thêm mới hoặc cập nhật customer_orders theo order_id."""
    if not order or not order.get("order_id"):
        return

    customer_order_doc = build_customer_order_doc(order)

    customer_orders_collection.update_one(
        {"order_id": order.get("order_id")},
        {"$set": customer_order_doc},
        upsert=True
    )


def sync_customer_orders_for_customer(customer_id):
    """
    Khi sửa thông tin khách hàng, cập nhật lại thông tin denormalized
    trong customer_orders.
    """
    orders_data = list(orders_collection.find({"customer_id": customer_id}, {"_id": 0}))

    for order in orders_data:
        upsert_customer_order(order)


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

        orders_collection.update_many(
            {"customer_id": customer_id},
            {"$set": {
                "customer_name": name,
                "customer_phone": phone
            }}
        )

        sync_customer_orders_for_customer(customer_id)

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
            {"customer_phone": regex},
            {"items.book_id": regex},
            {"items.title": regex}
        ]

    orders_data, pagination = get_pagination_data(
        orders_collection,
        query,
        sort_field="created_at",
        sort_direction=-1,
        default_per_page=10
    )

    return render_template(
        "orders.html",
        orders=orders_data,
        keyword=keyword,
        selected_status=status,
        status_list=get_order_status_list(),
        total_orders=pagination["total_items"],
        page=pagination["page"],
        per_page=pagination["per_page"],
        total_pages=pagination["total_pages"],
        start_index=pagination["start_index"],
        end_index=pagination["end_index"],
        allowed_per_pages=pagination["allowed_per_pages"]
    )




def parse_order_items_from_form():
    """
    Đọc danh sách sản phẩm từ form thêm/sửa đơn hàng.
    Form dùng book_id[] và quantity[] để cho phép một đơn có nhiều sản phẩm.
    """
    book_ids = request.form.getlist("book_id[]")
    quantities = request.form.getlist("quantity[]")

    items = []
    total_amount = 0
    errors = []

    for index, (book_id, quantity_text) in enumerate(zip(book_ids, quantities), start=1):
        book_id = (book_id or "").strip()
        quantity_text = (quantity_text or "1").strip()

        if not book_id:
            continue

        try:
            quantity = int(quantity_text)
            if quantity <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"Số lượng dòng {index} phải là số nguyên lớn hơn 0.")
            continue

        book = books_collection.find_one({"book_id": book_id}, {"_id": 0})

        if not book:
            errors.append(f"Không tìm thấy sách có mã {book_id}.")
            continue

        price = int(book.get("price", 0))
        amount = price * quantity

        items.append({
            "book_id": book.get("book_id"),
            "title": book.get("title"),
            "price": price,
            "quantity": quantity,
            "amount": amount
        })

        total_amount += amount

    if not items:
        errors.append("Đơn hàng phải có ít nhất 1 sản phẩm.")

    return items, total_amount, errors


def get_form_items_for_render():
    """
    Tạo lại danh sách sản phẩm đã nhập để render lại form khi có lỗi validate.
    """
    book_ids = request.form.getlist("book_id[]")
    quantities = request.form.getlist("quantity[]")

    form_items = []

    for book_id, quantity in zip(book_ids, quantities):
        if not book_id and not quantity:
            continue

        form_items.append({
            "book_id": book_id,
            "quantity": quantity or "1"
        })

    return form_items or [{"book_id": "", "quantity": 1}]

@app.route("/add-order", methods=["GET", "POST"])
def add_order():
    customers_data = list(customers_collection.find({}, {"_id": 0}).sort("customer_id", 1))
    books_data = list(books_collection.find({}, {"_id": 0}).sort("book_id", 1))
    status_list = get_order_status_list()

    if request.method == "POST":
        customer_id = request.form.get("customer_id", "").strip()
        status = request.form.get("status", "Chờ xử lý").strip()
        note = request.form.get("note", "").strip()

        if status not in status_list:
            return render_template(
                "add_order.html",
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                current_items=get_form_items_for_render(),
                error="Trạng thái đơn hàng không hợp lệ.",
                form_data=request.form
            )

        customer = customers_collection.find_one({"customer_id": customer_id}, {"_id": 0})

        if not customer:
            return render_template(
                "add_order.html",
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                current_items=get_form_items_for_render(),
                error="Vui lòng chọn khách hàng hợp lệ.",
                form_data=request.form
            )

        items, total_amount, errors = parse_order_items_from_form()

        if errors:
            return render_template(
                "add_order.html",
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                current_items=get_form_items_for_render(),
                error=" ".join(errors),
                form_data=request.form
            )

        new_order = {
            "order_id": generate_next_order_id(),
            "customer_id": customer.get("customer_id"),
            "customer_name": customer.get("name"),
            "customer_phone": customer.get("phone"),
            "items": items,
            "total_amount": total_amount,
            "status": status,
            "note": note,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        orders_collection.insert_one(new_order)
        upsert_customer_order(new_order)

        return redirect(url_for("order_detail", order_id=new_order["order_id"]))

    return render_template(
        "add_order.html",
        customers=customers_data,
        books=books_data,
        status_list=status_list,
        current_items=[{"book_id": "", "quantity": 1}],
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

    customers_data = list(customers_collection.find({}, {"_id": 0}).sort("customer_id", 1))
    books_data = list(books_collection.find({}, {"_id": 0}).sort("book_id", 1))
    status_list = get_order_status_list()
    current_items = order.get("items", []) or [{"book_id": "", "quantity": 1}]

    if request.method == "POST":
        customer_id = request.form.get("customer_id", "").strip()
        status = request.form.get("status", "").strip()
        note = request.form.get("note", "").strip()

        if status not in status_list:
            return render_template(
                "edit_order.html",
                order=order,
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                current_items=get_form_items_for_render(),
                error="Trạng thái đơn hàng không hợp lệ.",
                form_data=request.form
            )

        customer = customers_collection.find_one({"customer_id": customer_id}, {"_id": 0})

        if not customer:
            return render_template(
                "edit_order.html",
                order=order,
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                current_items=get_form_items_for_render(),
                error="Vui lòng chọn khách hàng hợp lệ.",
                form_data=request.form
            )

        items, total_amount, errors = parse_order_items_from_form()

        if errors:
            return render_template(
                "edit_order.html",
                order=order,
                customers=customers_data,
                books=books_data,
                status_list=status_list,
                current_items=get_form_items_for_render(),
                error=" ".join(errors),
                form_data=request.form
            )

        update_data = {
            "customer_id": customer.get("customer_id"),
            "customer_name": customer.get("name"),
            "customer_phone": customer.get("phone"),
            "items": items,
            "total_amount": total_amount,
            "status": status,
            "note": note,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        orders_collection.update_one(
            {"order_id": order_id},
            {"$set": update_data}
        )

        updated_order = orders_collection.find_one({"order_id": order_id}, {"_id": 0})

        if updated_order:
            upsert_customer_order(updated_order)

        return redirect(url_for("order_detail", order_id=order_id))

    return render_template(
        "edit_order.html",
        order=order,
        customers=customers_data,
        books=books_data,
        status_list=status_list,
        current_items=current_items,
        error=None,
        form_data=order
    )


@app.route("/delete-order/<order_id>", methods=["POST"])
def delete_order(order_id):
    orders_collection.delete_one({"order_id": order_id})
    customer_orders_collection.delete_one({"order_id": order_id})
    return redirect(url_for("orders"))


@app.route("/sync-customer-orders")
def sync_customer_orders():
    """
    Đồng bộ lại toàn bộ customer_orders từ orders.
    Dùng khi import dữ liệu mới hoặc khi customer_orders bị lệch.
    """
    orders_data = list(orders_collection.find({}, {"_id": 0}))

    customer_orders_collection.delete_many({})

    docs = [build_customer_order_doc(order) for order in orders_data]

    if docs:
        customer_orders_collection.insert_many(docs)

    return redirect(url_for("customer_orders"))


@app.route("/customer-orders")
def customer_orders():
    keyword = request.args.get("keyword", "").strip()
    status = request.args.get("status", "").strip()

    query = {}

    if status:
        query["order_status"] = status

    if keyword:
        regex = {"$regex": keyword, "$options": "i"}
        query["$or"] = [
            {"customer_order_id": regex},
            {"customer_id": regex},
            {"customer_name": regex},
            {"customer_phone": regex},
            {"customer_email": regex},
            {"customer_address": regex},
            {"order_id": regex},
            {"order_status": regex},
            {"items.book_id": regex},
            {"items.title": regex}
        ]

    customer_orders_data, pagination = get_pagination_data(
        customer_orders_collection,
        query,
        sort_field="order_date",
        sort_direction=-1,
        default_per_page=10
    )

    statuses = get_order_status_list()

    total_revenue = sum(item.get("total_amount", 0) for item in customer_orders_data)
    total_items = sum(item.get("total_items", 0) for item in customer_orders_data)
    total_customers = len(set(item.get("customer_id") for item in customer_orders_data))

    return render_template(
        "customer_orders.html",
        customer_orders=customer_orders_data,
        keyword=keyword,
        selected_status=status,
        statuses=statuses,
        total_customer_orders=pagination["total_items"],
        total_customers=total_customers,
        total_items=total_items,
        total_revenue=total_revenue,
        page=pagination["page"],
        per_page=pagination["per_page"],
        total_pages=pagination["total_pages"],
        start_index=pagination["start_index"],
        end_index=pagination["end_index"],
        allowed_per_pages=pagination["allowed_per_pages"]
    )


@app.route("/data-mining")
def data_mining():
    # Đã gộp nội dung data_mining vào analytics để tránh tách trang.
    return redirect(url_for("analytics", min_support=request.args.get("min_support", 0.03), min_confidence=request.args.get("min_confidence", 0.2)))


if __name__ == "__main__":
    app.run(debug=True)