from flask import Flask

from app_mvc.config import Config
from app_mvc.extensions import init_db


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config_class)

    init_db(app)

    # Import controller modules sau khi MongoDB đã được khởi tạo
    # để các collection trong model layer không bị None.
    from app_mvc.controllers.dashboard_controller import register_routes as register_dashboard_routes
    from app_mvc.controllers.book_controller import register_routes as register_book_routes
    from app_mvc.controllers.analytics_controller import register_routes as register_analytics_routes
    from app_mvc.controllers.customer_controller import register_routes as register_customer_routes
    from app_mvc.controllers.order_controller import register_routes as register_order_routes
    from app_mvc.controllers.review_controller import register_routes as register_review_routes

    register_dashboard_routes(app)
    register_book_routes(app)
    register_analytics_routes(app)
    register_customer_routes(app)
    register_order_routes(app)
    register_review_routes(app)

    return app
