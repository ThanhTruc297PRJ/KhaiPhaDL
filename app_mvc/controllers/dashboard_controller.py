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


def home():
    total_books = books_collection.count_documents({})
    total_reviews = 0
    categories = set()

    books = books_collection.find({}, {"_id": 0})

    for book in books:
        total_reviews += len(book.get("reviews", []))
        categories.add(book.get("category", "Không xác định"))

    total_customers = customers_collection.count_documents({})
    total_orders = orders_collection.count_documents({})
    total_customer_reviews = customer_reviews_collection.count_documents({})

    revenue_pipeline = [
        {"$group": {"_id": None, "total": {"$sum": "$total_amount"}}}
    ]
    revenue_result = list(orders_collection.aggregate(revenue_pipeline))
    total_revenue = revenue_result[0]["total"] if revenue_result else 0

    recent_orders = list(
        orders_collection
        .find({}, {"_id": 0})
        .sort("created_at", -1)
        .limit(6)
    )

    return render_template(
        "index.html",
        total_books=total_books,
        total_reviews=total_reviews,
        total_categories=len(categories),
        total_customers=total_customers,
        total_orders=total_orders,
        total_customer_reviews=total_customer_reviews,
        total_revenue=total_revenue,
        recent_orders=recent_orders
    )



def register_routes(app):
    app.add_url_rule('/', endpoint="home", view_func=home)
