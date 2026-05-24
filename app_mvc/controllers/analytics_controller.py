from flask import render_template, request, redirect, url_for, jsonify
from collections import Counter, defaultdict
from datetime import datetime
import math
import re

from app_mvc.models.collections import (
    books_collection,
    customers_collection,
    orders_collection,
    customer_orders_collection,
    customer_reviews_collection,
)
from app_mvc.services.data_mining_service import *


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

    try:
        cluster_k = int(request.args.get("cluster_k", 4))
    except ValueError:
        cluster_k = 4

    cluster_evaluation = evaluate_customer_clusters(selected_k=cluster_k)

    association_rules = mine_association_rules(
        min_support=min_support,
        min_confidence=min_confidence,
        limit=20
    )
    customer_segments, segment_counter = get_customer_segments()
    sales_summary = get_sales_mining_summary()
    behavior_summary = get_customer_purchase_behavior()
    customer_review_stats = get_customer_review_stats()

    total_orders = orders_collection.count_documents({})
    total_customers = customers_collection.count_documents({})

    return render_template(
        "analytics.html",
        total_books=total_books,
        total_reviews=total_reviews,
        total_categories=len(category_counter),
        total_customers=total_customers,
        total_orders=total_orders,
        total_customer_reviews=customer_review_stats["total_customer_reviews"],
        reviewer_count=customer_review_stats["reviewer_count"],

        category_labels=list(category_counter.keys()),
        category_values=list(category_counter.values()),
        sentiment_labels=list(sentiment_counter.keys()),
        sentiment_values=list(sentiment_counter.values()),
        customer_review_sentiment_labels=list(customer_review_stats["sentiment_counter"].keys()),
        customer_review_sentiment_values=list(customer_review_stats["sentiment_counter"].values()),
        customer_review_rating_labels=list(customer_review_stats["rating_counter"].keys()),
        customer_review_rating_values=list(customer_review_stats["rating_counter"].values()),

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
        monthly_values=[item[1] for item in sales_summary["monthly_revenue"]],

        top_customers_by_revenue=behavior_summary["top_customers_by_revenue"],
        top_customers_by_orders=behavior_summary["top_customers_by_orders"],
        customer_revenue_labels=[
            f'{item["customer_id"]} - {item["customer_name"]}'
            for item in behavior_summary["top_customers_by_revenue"]
        ],
        customer_revenue_values=[
            item["total_revenue"]
            for item in behavior_summary["top_customers_by_revenue"]
        ],
        customer_order_labels=[
            f'{item["customer_id"]} - {item["customer_name"]}'
            for item in behavior_summary["top_customers_by_orders"]
        ],
        customer_order_values=[
            item["order_count"]
            for item in behavior_summary["top_customers_by_orders"]
        ],
        purchase_category_labels=[item[0] for item in behavior_summary["category_purchase"]],
        purchase_category_values=[item[1] for item in behavior_summary["category_purchase"]],
        behavior_summary=behavior_summary,
        cluster_evaluation=cluster_evaluation,
        cluster_k=cluster_evaluation["selected_k"]
    )

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

def data_mining():
    # Đã gộp nội dung data_mining vào analytics để tránh tách trang.
    return redirect(url_for("analytics", min_support=request.args.get("min_support", 0.03), min_confidence=request.args.get("min_confidence", 0.2)))



def register_routes(app):
    app.add_url_rule('/analytics', endpoint="analytics", view_func=analytics)
    app.add_url_rule('/api/association-rules', endpoint="api_association_rules", view_func=api_association_rules)
    app.add_url_rule('/data-mining', endpoint="data_mining", view_func=data_mining)
