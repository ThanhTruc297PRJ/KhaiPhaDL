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


def customers():
    keyword = request.args.get("keyword", "").strip()

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
    if keyword:
        regex = {"$regex": keyword, "$options": "i"}
        query["$or"] = [
            {"customer_id": regex},
            {"name": regex},
            {"phone": regex},
            {"email": regex},
            {"address": regex},
            {"segment_demo": regex}
        ]

    total_customers = customers_collection.count_documents(query)
    total_pages = math.ceil(total_customers / per_page) if total_customers > 0 else 1

    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    skip_customers = (page - 1) * per_page

    customers_data = list(
        customers_collection
        .find(query, {"_id": 0})
        .sort("customer_id", 1)
        .skip(skip_customers)
        .limit(per_page)
    )

    for customer in customers_data:
        customer_id = customer.get("customer_id")
        customer["order_count"] = orders_collection.count_documents({"customer_id": customer_id})
        customer["review_count"] = customer_reviews_collection.count_documents({"customer_id": customer_id})

    start_index = skip_customers + 1 if total_customers > 0 else 0
    end_index = min(skip_customers + per_page, total_customers)

    return render_template(
        "customers.html",
        customers=customers_data,
        keyword=keyword,
        total_customers=total_customers,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        start_index=start_index,
        end_index=end_index,
        allowed_per_pages=allowed_per_pages
    )

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
        sync_customer_reviews_for_customer(customer_id)

        return redirect(url_for("customers"))

    return render_template(
        "edit_customer.html",
        customer=customer,
        error=None,
        form_data=customer
    )

def delete_customer(customer_id):
    orders_count = orders_collection.count_documents({"customer_id": customer_id})
    reviews_count = customer_reviews_collection.count_documents({"customer_id": customer_id})

    if orders_count > 0 or reviews_count > 0:
        return "Không thể xóa khách hàng vì đã có đơn hàng hoặc review liên quan.", 400

    customers_collection.delete_one({"customer_id": customer_id})
    return redirect(url_for("customers"))



def register_routes(app):
    app.add_url_rule('/customers', endpoint="customers", view_func=customers)
    app.add_url_rule('/add-customer', endpoint="add_customer", view_func=add_customer, methods=['GET', 'POST'])
    app.add_url_rule('/edit-customer/<customer_id>', endpoint="edit_customer", view_func=edit_customer, methods=['GET', 'POST'])
    app.add_url_rule('/delete-customer/<customer_id>', endpoint="delete_customer", view_func=delete_customer, methods=['POST'])
