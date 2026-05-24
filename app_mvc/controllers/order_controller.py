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

def order_detail(order_id):
    order = orders_collection.find_one({"order_id": order_id}, {"_id": 0})

    if not order:
        return "Không tìm thấy đơn hàng", 404

    return render_template("order_detail.html", order=order)

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

def delete_order(order_id):
    orders_collection.delete_one({"order_id": order_id})
    customer_orders_collection.delete_one({"order_id": order_id})
    return redirect(url_for("orders"))

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



def register_routes(app):
    app.add_url_rule('/orders', endpoint="orders", view_func=orders)
    app.add_url_rule('/add-order', endpoint="add_order", view_func=add_order, methods=['GET', 'POST'])
    app.add_url_rule('/order/<order_id>', endpoint="order_detail", view_func=order_detail)
    app.add_url_rule('/edit-order/<order_id>', endpoint="edit_order", view_func=edit_order, methods=['GET', 'POST'])
    app.add_url_rule('/delete-order/<order_id>', endpoint="delete_order", view_func=delete_order, methods=['POST'])
    app.add_url_rule('/sync-customer-orders', endpoint="sync_customer_orders", view_func=sync_customer_orders)
    app.add_url_rule('/customer-orders', endpoint="customer_orders", view_func=customer_orders)
