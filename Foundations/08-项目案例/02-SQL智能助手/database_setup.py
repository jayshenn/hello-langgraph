"""
数据库初始化脚本

创建示例数据库，包含：
- 产品表 (products)
- 客户表 (customers)
- 订单表 (orders)
- 订单明细表 (order_items)
- 产品分类表 (categories)

用于演示SQL智能助手的功能。
"""

import sqlite3
import pandas as pd
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any


class DatabaseSetup:
    """数据库初始化类"""

    def __init__(self, db_path: str = "./data/sample_database.db"):
        """初始化"""
        self.db_path = db_path
        self.connection = None

        # 确保数据目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        """连接数据库"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            print(f"✅ 已连接到数据库: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False

    def create_tables(self):
        """创建数据表"""
        if not self.connection:
            return False

        try:
            cursor = self.connection.cursor()

            # 1. 创建产品分类表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. 创建产品表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    price DECIMAL(10, 2) NOT NULL,
                    category_id INTEGER,
                    stock_quantity INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (category_id) REFERENCES categories(id)
                )
            """)

            # 3. 创建客户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(200) UNIQUE NOT NULL,
                    phone VARCHAR(20),
                    address TEXT,
                    city VARCHAR(100),
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # 4. 创建订单表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_amount DECIMAL(10, 2) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    shipping_address TEXT,
                    payment_method VARCHAR(50),
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
            """)

            # 5. 创建订单明细表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price DECIMAL(10, 2) NOT NULL,
                    total_price DECIMAL(10, 2) NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(id),
                    FOREIGN KEY (product_id) REFERENCES products(id)
                )
            """)

            self.connection.commit()
            print("✅ 数据表创建成功")
            return True

        except Exception as e:
            print(f"❌ 创建表失败: {e}")
            return False

    def insert_sample_data(self):
        """插入示例数据"""
        if not self.connection:
            return False

        try:
            cursor = self.connection.cursor()

            # 1. 插入产品分类数据
            categories_data = [
                ("电子产品", "各种电子设备和配件"),
                ("服装鞋帽", "时尚服饰和鞋子"),
                ("家居用品", "家庭生活用品"),
                ("图书音像", "书籍、音乐和影视产品"),
                ("运动户外", "运动器材和户外用品"),
                ("美妆个护", "化妆品和个人护理用品"),
                ("食品饮料", "各类食品和饮品"),
                ("汽车用品", "汽车配件和用品")
            ]

            cursor.executemany(
                "INSERT INTO categories (name, description) VALUES (?, ?)",
                categories_data
            )

            # 2. 插入产品数据
            products_data = self._generate_products_data()
            cursor.executemany(
                "INSERT INTO products (name, description, price, category_id, stock_quantity) VALUES (?, ?, ?, ?, ?)",
                products_data
            )

            # 3. 插入客户数据
            customers_data = self._generate_customers_data()
            cursor.executemany(
                "INSERT INTO customers (name, email, phone, address, city, registration_date) VALUES (?, ?, ?, ?, ?, ?)",
                customers_data
            )

            # 4. 插入订单数据
            orders_data = self._generate_orders_data()
            cursor.executemany(
                "INSERT INTO orders (customer_id, order_date, total_amount, status, shipping_address, payment_method) VALUES (?, ?, ?, ?, ?, ?)",
                orders_data
            )

            # 5. 插入订单明细数据
            order_items_data = self._generate_order_items_data()
            cursor.executemany(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES (?, ?, ?, ?, ?)",
                order_items_data
            )

            self.connection.commit()
            print("✅ 示例数据插入成功")
            return True

        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
            return False

    def _generate_products_data(self) -> List[tuple]:
        """生成产品数据"""
        products = [
            # 电子产品 (category_id: 1)
            ("iPhone 15 Pro", "苹果最新旗舰手机", 8999.00, 1, 50),
            ("MacBook Air M2", "轻薄笔记本电脑", 9499.00, 1, 30),
            ("AirPods Pro", "主动降噪无线耳机", 1999.00, 1, 100),
            ("iPad Air", "10.9英寸平板电脑", 4399.00, 1, 40),
            ("Apple Watch", "智能手表", 2999.00, 1, 60),

            # 服装鞋帽 (category_id: 2)
            ("Nike Air Max", "经典运动鞋", 899.00, 2, 80),
            ("Adidas T恤", "纯棉运动T恤", 199.00, 2, 120),
            ("Levis牛仔裤", "经典款牛仔裤", 699.00, 2, 90),
            ("Columbia冲锋衣", "户外防风外套", 1299.00, 2, 45),
            ("Uniqlo羽绒服", "轻薄保暖羽绒服", 599.00, 2, 70),

            # 家居用品 (category_id: 3)
            ("Dyson吸尘器", "无线手持吸尘器", 2999.00, 3, 25),
            ("IKEA书架", "简约风格书架", 399.00, 3, 35),
            ("飞利浦电饭煲", "智能电饭煲", 899.00, 3, 50),
            ("小米空气净化器", "智能空气净化器", 1299.00, 3, 40),
            ("宜家沙发", "三人位布艺沙发", 2599.00, 3, 15),

            # 图书音像 (category_id: 4)
            ("《Python编程》", "编程入门教程", 89.00, 4, 200),
            ("《经济学原理》", "经济学经典教材", 128.00, 4, 150),
            ("周杰伦专辑", "最新音乐专辑", 68.00, 4, 300),
            ("《三体》套装", "科幻小说三部曲", 158.00, 4, 180),
            ("BBC纪录片", "自然世界纪录片", 98.00, 4, 100),

            # 运动户外 (category_id: 5)
            ("瑜伽垫", "环保TPE瑜伽垫", 168.00, 5, 90),
            ("哑铃套装", "可调节哑铃", 599.00, 5, 40),
            ("登山背包", "40L户外背包", 399.00, 5, 60),
            ("跑步机", "家用折叠跑步机", 2999.00, 5, 20),
            ("羽毛球拍", "碳纤维羽毛球拍", 299.00, 5, 80),

            # 美妆个护 (category_id: 6)
            ("SK-II神仙水", "护肤精华水", 1590.00, 6, 30),
            ("兰蔻口红", "经典红色口红", 350.00, 6, 70),
            ("欧莱雅面膜", "补水保湿面膜", 89.00, 6, 150),
            ("雅诗兰黛眼霜", "抗衰老眼霜", 680.00, 6, 40),
            ("飞利浦剃须刀", "电动剃须刀", 899.00, 6, 50),

            # 食品饮料 (category_id: 7)
            ("茅台酒", "53度飞天茅台", 2699.00, 7, 10),
            ("星巴克咖啡豆", "中度烘焙咖啡豆", 128.00, 7, 100),
            ("进口巧克力", "瑞士黑巧克力", 68.00, 7, 200),
            ("有机蜂蜜", "纯天然蜂蜜", 158.00, 7, 80),
            ("五常大米", "东北优质大米", 89.00, 7, 300),

            # 汽车用品 (category_id: 8)
            ("行车记录仪", "高清夜视记录仪", 599.00, 8, 60),
            ("汽车脚垫", "全包围皮革脚垫", 299.00, 8, 80),
            ("车载充电器", "快充车载充电器", 79.00, 8, 150),
            ("汽车香水", "车载香薰", 39.00, 8, 200),
            ("轮胎", "米其林轮胎", 899.00, 8, 40)
        ]

        return products

    def _generate_customers_data(self) -> List[tuple]:
        """生成客户数据"""
        first_names = ["张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴"]
        second_names = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋"]
        cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安"]

        customers = []
        base_date = datetime.datetime(2022, 1, 1)

        for i in range(1, 101):  # 生成100个客户
            first_name = random.choice(first_names)
            second_name = random.choice(second_names)
            name = first_name + second_name + ("先生" if random.random() > 0.5 else "女士")

            email = f"user{i:03d}@example.com"
            phone = f"1{random.randint(3, 9)}{random.randint(100000000, 999999999)}"

            city = random.choice(cities)
            address = f"{city}市{random.choice(['朝阳', '海淀', '西城', '东城', '丰台'])}区{random.choice(['中关村', '王府井', '三里屯', '国贸', '五道口'])}街道{random.randint(1, 999)}号"

            # 随机注册时间
            days_since_base = random.randint(0, 700)  # 过去2年内
            registration_date = base_date + datetime.timedelta(days=days_since_base)

            customers.append((
                name, email, phone, address, city,
                registration_date.strftime("%Y-%m-%d %H:%M:%S")
            ))

        return customers

    def _generate_orders_data(self) -> List[tuple]:
        """生成订单数据"""
        orders = []
        statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
        payment_methods = ["credit_card", "alipay", "wechat_pay", "cash_on_delivery"]

        base_date = datetime.datetime(2023, 1, 1)

        for i in range(1, 501):  # 生成500个订单
            customer_id = random.randint(1, 100)  # 随机客户ID

            # 随机订单日期（2023年内）
            days_since_base = random.randint(0, 365)
            order_date = base_date + datetime.timedelta(days=days_since_base)

            total_amount = round(random.uniform(50, 5000), 2)
            status = random.choice(statuses)
            shipping_address = f"配送地址{i}"
            payment_method = random.choice(payment_methods)

            orders.append((
                customer_id,
                order_date.strftime("%Y-%m-%d %H:%M:%S"),
                total_amount,
                status,
                shipping_address,
                payment_method
            ))

        return orders

    def _generate_order_items_data(self) -> List[tuple]:
        """生成订单明细数据"""
        order_items = []

        for order_id in range(1, 501):  # 为每个订单生成1-5个明细
            items_count = random.randint(1, 5)

            for _ in range(items_count):
                product_id = random.randint(1, 40)  # 随机产品ID
                quantity = random.randint(1, 10)
                unit_price = round(random.uniform(10, 1000), 2)
                total_price = round(unit_price * quantity, 2)

                order_items.append((
                    order_id,
                    product_id,
                    quantity,
                    unit_price,
                    total_price
                ))

        return order_items

    def create_indexes(self):
        """创建索引以提高查询性能"""
        if not self.connection:
            return False

        try:
            cursor = self.connection.cursor()

            # 创建常用查询的索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id)",
                "CREATE INDEX IF NOT EXISTS idx_products_price ON products(price)",
                "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)",
                "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)",
                "CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id)",
                "CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id)",
                "CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)",
                "CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city)"
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            self.connection.commit()
            print("✅ 索引创建成功")
            return True

        except Exception as e:
            print(f"❌ 创建索引失败: {e}")
            return False

    def verify_data(self):
        """验证数据完整性"""
        if not self.connection:
            return False

        try:
            cursor = self.connection.cursor()

            # 检查各表的记录数
            tables = ['categories', 'products', 'customers', 'orders', 'order_items']

            print("\n📊 数据库统计信息:")
            print("-" * 40)

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table:15}: {count:6} 条记录")

            # 检查数据完整性
            print("\n🔍 数据完整性检查:")
            print("-" * 40)

            # 检查外键约束
            cursor.execute("""
                SELECT COUNT(*) FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                WHERE c.id IS NULL AND p.category_id IS NOT NULL
            """)
            orphan_products = cursor.fetchone()[0]
            print(f"孤立产品 (无分类):     {orphan_products} 个")

            cursor.execute("""
                SELECT COUNT(*) FROM orders o
                LEFT JOIN customers c ON o.customer_id = c.id
                WHERE c.id IS NULL
            """)
            orphan_orders = cursor.fetchone()[0]
            print(f"孤立订单 (无客户):     {orphan_orders} 个")

            cursor.execute("""
                SELECT COUNT(*) FROM order_items oi
                LEFT JOIN orders o ON oi.order_id = o.id
                WHERE o.id IS NULL
            """)
            orphan_items = cursor.fetchone()[0]
            print(f"孤立订单项 (无订单):   {orphan_items} 个")

            # 价格范围检查
            cursor.execute("SELECT MIN(price), MAX(price), AVG(price) FROM products")
            min_price, max_price, avg_price = cursor.fetchone()
            print(f"\n💰 产品价格范围:")
            print(f"最低价格: ¥{min_price:.2f}")
            print(f"最高价格: ¥{max_price:.2f}")
            print(f"平均价格: ¥{avg_price:.2f}")

            # 订单统计
            cursor.execute("SELECT MIN(total_amount), MAX(total_amount), AVG(total_amount) FROM orders")
            min_amount, max_amount, avg_amount = cursor.fetchone()
            print(f"\n📦 订单金额范围:")
            print(f"最小订单: ¥{min_amount:.2f}")
            print(f"最大订单: ¥{max_amount:.2f}")
            print(f"平均订单: ¥{avg_amount:.2f}")

            return True

        except Exception as e:
            print(f"❌ 数据验证失败: {e}")
            return False

    def export_sample_data(self):
        """导出示例数据到CSV文件"""
        if not self.connection:
            return False

        try:
            # 确保data目录存在
            data_dir = Path("./data")
            data_dir.mkdir(exist_ok=True)

            # 导出各表数据
            tables = ['categories', 'products', 'customers', 'orders', 'order_items']

            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", self.connection)
                csv_path = data_dir / f"{table}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"✅ {table} 数据已导出到 {csv_path}")

            return True

        except Exception as e:
            print(f"❌ 导出数据失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("✅ 数据库连接已关闭")


def main():
    """主函数"""
    print("🚀 SQL智能助手 - 数据库初始化")
    print("=" * 50)

    # 初始化数据库
    db_setup = DatabaseSetup()

    if not db_setup.connect():
        print("❌ 数据库初始化失败")
        return

    print("📝 开始创建数据库结构...")

    # 创建表
    if not db_setup.create_tables():
        print("❌ 创建表失败")
        return

    print("📊 开始插入示例数据...")

    # 插入数据
    if not db_setup.insert_sample_data():
        print("❌ 插入数据失败")
        return

    print("🔍 创建数据库索引...")

    # 创建索引
    if not db_setup.create_indexes():
        print("❌ 创建索引失败")
        return

    print("✅ 验证数据完整性...")

    # 验证数据
    if not db_setup.verify_data():
        print("❌ 数据验证失败")
        return

    print("\n📤 导出示例数据...")

    # 导出数据
    if not db_setup.export_sample_data():
        print("❌ 导出数据失败")
        return

    # 关闭连接
    db_setup.close()

    print("\n🎉 数据库初始化完成！")
    print("\n📋 接下来你可以:")
    print("1. 运行 python sql_agent.py 启动SQL智能助手")
    print("2. 尝试自然语言查询，如：'显示所有产品的名称和价格'")
    print("3. 查看 data/ 目录下的CSV文件了解数据结构")


if __name__ == "__main__":
    main()