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


def sync_customer_reviews():
    result = sync_customer_reviews_from_books()

    return redirect(url_for("customer_reviews"))

def customer_reviews():
    keyword = request.args.get("keyword", "").strip()
    sentiment = request.args.get("sentiment", "").strip()
    rating = request.args.get("rating", "").strip()

    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1

    try:
        per_page = int(request.args.get("per_page", 10))
    except ValueError:
        per_page = 10

    allowed_per_pages = [10, 20, 50, 100]

    if per_page not in allowed_per_pages:
        per_page = 10

    query = {}

    if sentiment:
        query["sentiment"] = sentiment

    if rating:
        try:
            query["rating"] = int(rating)
        except ValueError:
            pass

    if keyword:
        regex = {"$regex": keyword, "$options": "i"}
        query["$or"] = [
            {"customer_review_id": regex},
            {"review_id": regex},
            {"customer_id": regex},
            {"customer_name": regex},
            {"customer_phone": regex},
            {"customer_email": regex},
            {"book_id": regex},
            {"book_title": regex},
            {"book_category": regex},
            {"comment": regex}
        ]

    total_customer_reviews = customer_reviews_collection.count_documents(query)
    total_pages = math.ceil(total_customer_reviews / per_page) if total_customer_reviews > 0 else 1

    if page < 1:
        page = 1

    if page > total_pages:
        page = total_pages

    skip_reviews = (page - 1) * per_page

    customer_reviews_data = list(
        customer_reviews_collection
        .find(query, {"_id": 0})
        .sort("review_date", -1)
        .skip(skip_reviews)
        .limit(per_page)
    )

    start_index = skip_reviews + 1 if total_customer_reviews > 0 else 0
    end_index = min(skip_reviews + per_page, total_customer_reviews)

    stats = get_customer_review_stats()
    sentiments = ["Tích cực", "Trung lập", "Tiêu cực"]

    return render_template(
        "customer_reviews.html",
        customer_reviews=customer_reviews_data,
        keyword=keyword,
        selected_sentiment=sentiment,
        selected_rating=rating,
        sentiments=sentiments,
        total_customer_reviews=total_customer_reviews,
        total_all_customer_reviews=stats["total_customer_reviews"],
        reviewer_count=stats["reviewer_count"],
        sentiment_labels=list(stats["sentiment_counter"].keys()),
        sentiment_values=list(stats["sentiment_counter"].values()),
        rating_labels=list(stats["rating_counter"].keys()),
        rating_values=list(stats["rating_counter"].values()),
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        start_index=start_index,
        end_index=end_index,
        allowed_per_pages=allowed_per_pages
    )



def register_routes(app):
    app.add_url_rule('/sync-customer-reviews', endpoint="sync_customer_reviews", view_func=sync_customer_reviews)
    app.add_url_rule('/customer-reviews', endpoint="customer_reviews", view_func=customer_reviews)
