from flask import Flask, render_template, request
from pymongo import MongoClient
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import math

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["book_mining_db"]
books_collection = db["books"]


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
    words = full_text.split()

    for word in words:
        if len(word) > 2 and word not in STOPWORDS:
            word_counter[word] += 1

    return word_counter.most_common(limit)


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

    reviews = book.get("reviews", [])
    total_reviews = len(reviews)

    avg_rating = 0
    sentiment_counter = {
        "Tích cực": 0,
        "Trung lập": 0,
        "Tiêu cực": 0
    }

    if total_reviews > 0:
        total_rating = sum(review.get("rating", 0) for review in reviews)
        avg_rating = round(total_rating / total_reviews, 2)

        for review in reviews:
            sentiment = get_sentiment_by_rating(review.get("rating", 0))
            sentiment_counter[sentiment] += 1

    top_keywords = get_book_top_keywords(book, limit=12)
    recommended_books = get_tfidf_recommendations(book_id, limit=5)

    return render_template(
        "book_detail.html",
        book=book,
        reviews=reviews,
        total_reviews=total_reviews,
        avg_rating=avg_rating,
        sentiment_counter=sentiment_counter,
        top_keywords=top_keywords,
        recommended_books=recommended_books,
        sentiment_labels=list(sentiment_counter.keys()),
        sentiment_values=list(sentiment_counter.values())
    )


@app.route("/search", methods=["GET"])
def search():
    keyword = request.args.get("keyword", "").strip()
    category = request.args.get("category", "").strip()
    search_type = request.args.get("search_type", "title")

    query = {}

    if category:
        query["category"] = category

    if keyword:
        regex = {"$regex": keyword, "$options": "i"}

        if search_type == "title":
            query["title"] = regex

        elif search_type == "review":
            query["reviews.comment"] = regex

        elif search_type == "tag":
            query["tags"] = regex

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

    results = list(books_collection.find(query, {"_id": 0}))

    # SỬA LỖI:
    # Nếu tìm theo bình luận đánh giá thì chỉ hiển thị các bình luận có chứa từ khóa.
    # Trước đó MongoDB lọc đúng sách, nhưng giao diện lại hiện 3 bình luận đầu tiên
    # nên nhìn giống như tìm kiếm bị sai.
    if keyword and search_type == "review":
        keyword_lower = keyword.lower()

        for book in results:
            matched_reviews = []

            for review in book.get("reviews", []):
                comment = review.get("comment", "")

                if keyword_lower in comment.lower():
                    matched_reviews.append(review)

            book["display_reviews"] = matched_reviews

    else:
        for book in results:
            book["display_reviews"] = book.get("reviews", [])[:3]

    categories = books_collection.distinct("category")

    return render_template(
        "search.html",
        results=results,
        categories=categories,
        keyword=keyword,
        selected_category=category,
        search_type=search_type
    )


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


if __name__ == "__main__":
    app.run(debug=True)