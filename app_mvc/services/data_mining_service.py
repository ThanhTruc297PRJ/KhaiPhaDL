from collections import Counter, defaultdict
from datetime import datetime
import math
import re

import common.extract_keyword
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import request
from app_mvc.models.collections import (
    books_collection,
    customers_collection,
    orders_collection,
    customer_orders_collection,
    customer_reviews_collection,
)


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
    return word_counter.most_common(limit)# chổ này là code cũ 

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

def normalize_customer_name(name):
    return re.sub(r"\s+", " ", str(name or "").strip().lower())

def find_customer_by_name(name):
    normalized = normalize_customer_name(name)
    if not normalized:
        return None

    for customer in customers_collection.find({}, {"_id": 0}):
        if normalize_customer_name(customer.get("name")) == normalized:
            return customer

    return None

def get_or_create_reviewer_customer(reviewer_name):
    """
    Reviewer cũng là customer.
    Nếu reviewer chưa có trong customers thì tạo customer mới với source = reviewer.
    """
    reviewer_name = re.sub(r"\s+", " ", str(reviewer_name or "").strip())

    if not reviewer_name:
        reviewer_name = "Khách ẩn danh"

    existing_customer = find_customer_by_name(reviewer_name)

    if existing_customer:
        return existing_customer

    customer_id = generate_next_customer_id()
    slug = re.sub(r"[^a-zA-Z0-9]+", "", reviewer_name.lower()) or customer_id.lower()

    new_customer = {
        "customer_id": customer_id,
        "name": reviewer_name,
        "email": f"{slug}.{customer_id.lower()}@demo.local",
        "phone": "",
        "address": "",
        "segment_demo": "Reviewer",
        "source": "reviewer",
        "created_at": datetime.now().strftime("%d/%m/%Y")
    }

    customers_collection.insert_one(new_customer)

    return new_customer

def build_customer_review_doc(book, review):
    customer_id = review.get("customer_id", "")
    customer = customers_collection.find_one({"customer_id": customer_id}, {"_id": 0}) if customer_id else None

    if not customer:
        customer = get_or_create_reviewer_customer(review.get("user") or review.get("customer_name"))

    customer_name = customer.get("name", review.get("user", ""))
    review_id = review.get("review_id") or f'{book.get("book_id")}_RV'

    return {
        "customer_review_id": f'{book.get("book_id")}_{review_id}',
        "review_id": review_id,
        "customer_id": customer.get("customer_id"),
        "customer_name": customer_name,
        "customer_phone": customer.get("phone", ""),
        "customer_email": customer.get("email", ""),
        "book_id": book.get("book_id"),
        "book_title": book.get("title"),
        "book_category": book.get("category"),
        "rating": int(review.get("rating", 0) or 0),
        "comment": review.get("comment", ""),
        "sentiment": get_sentiment_by_rating(int(review.get("rating", 0) or 0)),
        "review_date": review.get("created_at", ""),
        "source": "books.reviews",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def upsert_customer_review(book, review):
    doc = build_customer_review_doc(book, review)

    customer_reviews_collection.update_one(
        {"customer_review_id": doc["customer_review_id"]},
        {"$set": doc},
        upsert=True
    )

    return doc

def sync_customer_reviews_from_books():
    """
    Đồng bộ collection customer_reviews từ mảng books.reviews.
    Đồng thời chuẩn hóa mỗi review để có customer_id và customer_name.
    """
    customer_reviews_collection.delete_many({})

    synced_count = 0
    created_customer_count = 0

    for book in books_collection.find({}, {"_id": 0}):
        reviews = book.get("reviews", [])
        updated_reviews = []

        for index, review in enumerate(reviews, start=1):
            reviewer_name = review.get("user") or review.get("customer_name") or "Khách ẩn danh"
            before_count = customers_collection.count_documents({})

            customer = None
            if review.get("customer_id"):
                customer = customers_collection.find_one({"customer_id": review.get("customer_id")}, {"_id": 0})

            if not customer:
                customer = get_or_create_reviewer_customer(reviewer_name)

            after_count = customers_collection.count_documents({})
            if after_count > before_count:
                created_customer_count += 1

            review_id = review.get("review_id") or f'{book.get("book_id")}_R{index:03d}'

            normalized_review = {
                **review,
                "review_id": review_id,
                "customer_id": customer.get("customer_id"),
                "customer_name": customer.get("name"),
                "user": customer.get("name")
            }

            updated_reviews.append(normalized_review)
            upsert_customer_review(book, normalized_review)
            synced_count += 1

        books_collection.update_one(
            {"book_id": book.get("book_id")},
            {"$set": {"reviews": updated_reviews}}
        )

    return {
        "synced_count": synced_count,
        "created_customer_count": created_customer_count
    }

def sync_customer_reviews_for_customer(customer_id):
    """
    Khi sửa khách hàng, cập nhật lại thông tin reviewer trong books.reviews
    và collection customer_reviews.
    """
    customer = customers_collection.find_one({"customer_id": customer_id}, {"_id": 0})

    if not customer:
        return

    for book in books_collection.find({"reviews.customer_id": customer_id}, {"_id": 0}):
        reviews = book.get("reviews", [])
        changed = False
        updated_reviews = []

        for review in reviews:
            if review.get("customer_id") == customer_id:
                review = {
                    **review,
                    "customer_name": customer.get("name"),
                    "user": customer.get("name")
                }
                changed = True

            updated_reviews.append(review)

        if changed:
            books_collection.update_one(
                {"book_id": book.get("book_id")},
                {"$set": {"reviews": updated_reviews}}
            )

    customer_reviews_collection.update_many(
        {"customer_id": customer_id},
        {"$set": {
            "customer_name": customer.get("name"),
            "customer_phone": customer.get("phone", ""),
            "customer_email": customer.get("email", ""),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }}
    )

def get_customer_review_stats():
    total_customer_reviews = customer_reviews_collection.count_documents({})
    reviewer_count = len(customer_reviews_collection.distinct("customer_id"))

    sentiment_counter = Counter()
    rating_counter = Counter()

    for item in customer_reviews_collection.find({}, {"_id": 0, "sentiment": 1, "rating": 1}):
        sentiment_counter[item.get("sentiment", "Không xác định")] += 1
        rating_counter[str(item.get("rating", 0))] += 1

    return {
        "total_customer_reviews": total_customer_reviews,
        "reviewer_count": reviewer_count,
        "sentiment_counter": sentiment_counter,
        "rating_counter": rating_counter
    }

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

def build_rfm_customer_data():
    """
    Xây dựng dữ liệu RFM từ orders:
    R - Recency: số ngày từ lần mua gần nhất
    F - Frequency: số đơn hàng
    M - Monetary: tổng doanh thu của khách hàng
    """
    orders_data = list(orders_collection.find({}, {"_id": 0}))

    if not orders_data:
        return []

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

    rfm_data = []

    for stats in customer_stats.values():
        last_order_date = stats["last_order_date"]
        recency = (max_date - last_order_date).days if last_order_date else 999

        rfm_data.append({
            "customer_id": stats["customer_id"],
            "customer_name": stats["customer_name"],
            "recency": recency,
            "frequency": stats["frequency"],
            "monetary": stats["monetary"],
            "last_order_date": last_order_date.strftime("%Y-%m-%d") if last_order_date else ""
        })

    return rfm_data

def evaluate_customer_clusters(selected_k=4):
    """
    Đánh giá phân cụm khách hàng bằng KMeans trên dữ liệu RFM.
    Trả về:
    - Elbow Method: inertia theo từng k
    - Silhouette Score theo từng k
    - Số lượng khách theo cụm
    - Doanh thu theo cụm
    - Giá trị trung bình R/F/M theo cụm
    """
    rfm_data = build_rfm_customer_data()
    n_customers = len(rfm_data)

    result = {
        "enabled": False,
        "message": "Cần ít nhất 3 khách hàng có đơn hàng để đánh giá phân cụm.",
        "selected_k": selected_k,
        "silhouette_score": None,
        "elbow_labels": [],
        "elbow_values": [],
        "silhouette_labels": [],
        "silhouette_values": [],
        "cluster_count_labels": [],
        "cluster_count_values": [],
        "cluster_revenue_labels": [],
        "cluster_revenue_values": [],
        "avg_recency_values": [],
        "avg_frequency_values": [],
        "avg_monetary_values": [],
        "cluster_summary": [],
        "clustered_customers": []
    }

    if n_customers < 3:
        return result

    max_k = min(8, n_customers - 1)
    min_k = 2

    if selected_k < min_k:
        selected_k = min_k
    if selected_k > max_k:
        selected_k = max_k

    features = [
        [
            float(item["recency"]),
            float(item["frequency"]),
            float(item["monetary"])
        ]
        for item in rfm_data
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    elbow_labels = []
    elbow_values = []
    silhouette_labels = []
    silhouette_values = []

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)

        elbow_labels.append(str(k))
        elbow_values.append(round(float(model.inertia_), 4))

        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            score = silhouette_score(scaled_features, labels)
            silhouette_labels.append(str(k))
            silhouette_values.append(round(float(score), 4))

    final_model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    final_labels = final_model.fit_predict(scaled_features)

    final_silhouette = None
    if len(set(final_labels)) > 1 and len(set(final_labels)) < len(final_labels):
        final_silhouette = round(float(silhouette_score(scaled_features, final_labels)), 4)

    cluster_stats = defaultdict(lambda: {
        "cluster": "",
        "customer_count": 0,
        "total_revenue": 0,
        "total_recency": 0,
        "total_frequency": 0,
        "total_monetary": 0
    })

    clustered_customers = []

    for item, cluster_id in zip(rfm_data, final_labels):
        cluster_name = f"Cụm {int(cluster_id) + 1}"
        stats = cluster_stats[cluster_name]
        stats["cluster"] = cluster_name
        stats["customer_count"] += 1
        stats["total_revenue"] += item["monetary"]
        stats["total_recency"] += item["recency"]
        stats["total_frequency"] += item["frequency"]
        stats["total_monetary"] += item["monetary"]

        clustered_customers.append({
            **item,
            "cluster": cluster_name
        })

    cluster_summary = []
    for cluster_name, stats in sorted(cluster_stats.items()):
        count = stats["customer_count"] or 1
        avg_recency = round(stats["total_recency"] / count, 2)
        avg_frequency = round(stats["total_frequency"] / count, 2)
        avg_monetary = round(stats["total_monetary"] / count, 2)

        if avg_frequency >= 3 and avg_monetary >= 700000:
            interpretation = "Khách hàng trung thành / giá trị cao"
        elif avg_recency <= 30:
            interpretation = "Khách hàng mới hoặc vừa mua gần đây"
        elif avg_monetary >= 700000:
            interpretation = "Khách hàng chi tiêu cao nhưng tần suất chưa ổn định"
        else:
            interpretation = "Khách hàng phổ thông / cần kích hoạt lại"

        cluster_summary.append({
            "cluster": cluster_name,
            "customer_count": stats["customer_count"],
            "total_revenue": stats["total_revenue"],
            "avg_recency": avg_recency,
            "avg_frequency": avg_frequency,
            "avg_monetary": avg_monetary,
            "interpretation": interpretation
        })

    clustered_customers = sorted(
        clustered_customers,
        key=lambda x: (x["cluster"], -x["monetary"], -x["frequency"])
    )

    result.update({
        "enabled": True,
        "message": "",
        "selected_k": selected_k,
        "silhouette_score": final_silhouette,
        "elbow_labels": elbow_labels,
        "elbow_values": elbow_values,
        "silhouette_labels": silhouette_labels,
        "silhouette_values": silhouette_values,
        "cluster_count_labels": [item["cluster"] for item in cluster_summary],
        "cluster_count_values": [item["customer_count"] for item in cluster_summary],
        "cluster_revenue_labels": [item["cluster"] for item in cluster_summary],
        "cluster_revenue_values": [item["total_revenue"] for item in cluster_summary],
        "avg_recency_values": [item["avg_recency"] for item in cluster_summary],
        "avg_frequency_values": [item["avg_frequency"] for item in cluster_summary],
        "avg_monetary_values": [item["avg_monetary"] for item in cluster_summary],
        "cluster_summary": cluster_summary,
        "clustered_customers": clustered_customers[:50]
    })

    return result

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

def get_customer_purchase_behavior():
    """
    Phân tích hành vi mua hàng:
    - Top khách hàng theo doanh thu
    - Top khách hàng theo số đơn
    - Giá trị đơn hàng trung bình
    - Số sản phẩm trung bình / đơn
    - Tỷ lệ khách mua lặp lại
    - Số lượng mua theo thể loại
    """
    orders_data = list(orders_collection.find({}, {"_id": 0}))

    customer_order_counter = Counter()
    customer_revenue_counter = Counter()
    customer_name_map = {}
    category_purchase_counter = Counter()

    total_revenue = 0
    total_items = 0
    total_orders = len(orders_data)

    book_category_map = {}
    for book in books_collection.find({}, {"_id": 0, "book_id": 1, "category": 1}):
        book_category_map[book.get("book_id")] = book.get("category", "Không xác định")

    for order in orders_data:
        customer_id = order.get("customer_id", "Không xác định")
        customer_name = order.get("customer_name", "Không xác định")
        order_amount = int(order.get("total_amount", 0) or 0)

        customer_order_counter[customer_id] += 1
        customer_revenue_counter[customer_id] += order_amount
        customer_name_map[customer_id] = customer_name
        total_revenue += order_amount

        for item in order.get("items", []):
            quantity = int(item.get("quantity", 0) or 0)
            total_items += quantity

            book_id = item.get("book_id")
            category = book_category_map.get(book_id, "Không xác định")
            category_purchase_counter[category] += quantity

    unique_buyers = len(customer_order_counter)
    repeat_customers = sum(1 for count in customer_order_counter.values() if count >= 2)

    avg_order_value = round(total_revenue / total_orders, 2) if total_orders else 0
    avg_items_per_order = round(total_items / total_orders, 2) if total_orders else 0
    repeat_rate = round((repeat_customers / unique_buyers) * 100, 2) if unique_buyers else 0

    top_customers_by_revenue = []
    for customer_id, revenue in customer_revenue_counter.most_common(10):
        top_customers_by_revenue.append({
            "customer_id": customer_id,
            "customer_name": customer_name_map.get(customer_id, "Không xác định"),
            "total_revenue": revenue,
            "order_count": customer_order_counter.get(customer_id, 0)
        })

    top_customers_by_orders = []
    for customer_id, order_count in customer_order_counter.most_common(10):
        top_customers_by_orders.append({
            "customer_id": customer_id,
            "customer_name": customer_name_map.get(customer_id, "Không xác định"),
            "order_count": order_count,
            "total_revenue": customer_revenue_counter.get(customer_id, 0)
        })

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_items": total_items,
        "unique_buyers": unique_buyers,
        "repeat_customers": repeat_customers,
        "repeat_rate": repeat_rate,
        "avg_order_value": avg_order_value,
        "avg_items_per_order": avg_items_per_order,
        "top_customers_by_revenue": top_customers_by_revenue,
        "top_customers_by_orders": top_customers_by_orders,
        "category_purchase": category_purchase_counter.most_common()
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
