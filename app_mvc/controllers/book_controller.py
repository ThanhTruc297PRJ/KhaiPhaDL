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

def delete_book(book_id):
    books_collection.delete_one({"book_id": book_id})
    return redirect(url_for("books"))

def search():
    return redirect(url_for(
        "books",
        keyword=request.args.get("keyword", ""),
        search_type=request.args.get("search_type", "all"),
        category=request.args.get("category", "")
    ))

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



def register_routes(app):
    app.add_url_rule('/books', endpoint="books", view_func=books)
    app.add_url_rule('/book/<book_id>', endpoint="book_detail", view_func=book_detail)
    app.add_url_rule('/add-book', endpoint="add_book", view_func=add_book, methods=['GET', 'POST'])
    app.add_url_rule('/edit-book/<book_id>', endpoint="edit_book", view_func=edit_book, methods=['GET', 'POST'])
    app.add_url_rule('/delete-book/<book_id>', endpoint="delete_book", view_func=delete_book, methods=['POST'])
    app.add_url_rule('/search', endpoint="search", view_func=search, methods=['GET'])
    app.add_url_rule('/api/books-search', endpoint="api_books_search", view_func=api_books_search)
