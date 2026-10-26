"""Synthetic marketplace: product catalog, orders DB and the bilingual knowledge base."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from market_assistant.data.policies import (
    CATEGORIES,
    CATEGORY_BY_SLUG,
    FREE_DELIVERY_FROM_UZS,
    INSTALLMENT_MONTHS,
    PICKUP_STORAGE_DAYS,
    REFUND_DAYS,
    REGIONS,
    RETURN_DAYS,
    SUPPORT_HOURS,
)

REFERENCE_DATE = date(2026, 10, 20)  # simulated "today" shared by synth data and tools

_PRODUCTS: dict[str, list[tuple[str, str, list[str]]]] = {
    # slug: [(ru base name, uz base name, brands)]
    "smartphones": [
        (
            "Смартфон",
            "Smartfon",
            [
                "Samsung Galaxy A55",
                "Samsung Galaxy S24",
                "iPhone 15",
                "iPhone 16",
                "Xiaomi Redmi Note 13",
                "Honor X9b",
                "Tecno Spark 20",
            ],
        )
    ],
    "laptops": [
        (
            "Ноутбук",
            "Noutbuk",
            ["Lenovo IdeaPad 3", "HP Pavilion 15", "ASUS VivoBook 16", "Apple MacBook Air M3", "Acer Aspire 5"],
        )
    ],
    "headphones": [
        (
            "Наушники",
            "Quloqchin",
            ["AirPods Pro 2", "Samsung Galaxy Buds FE", "JBL Tune 520BT", "Xiaomi Redmi Buds 5", "Sony WH-1000XM5"],
        )
    ],
    "tv": [
        ("Телевизор", "Televizor", ["Samsung Crystal UHD", "LG NanoCell", "Artel UA50", "Xiaomi TV A Pro", "TCL C645"])
    ],
    "appliances": [
        ("Стиральная машина", "Kir yuvish mashinasi", ["Artel", "Samsung", "LG"]),
        ("Холодильник", "Muzlatkich", ["Artel", "Indesit", "Samsung"]),
        ("Пылесос", "Changyutgich", ["Dyson V12", "Xiaomi", "Philips"]),
    ],
    "clothes": [
        ("Куртка", "Kurtka", ["Zara", "LC Waikiki", "Koton"]),
        ("Футболка", "Futbolka", ["Uniqlo", "Koton", "DeFacto"]),
        ("Платье", "Ko'ylak", ["Zara", "Mango", "Koton"]),
    ],
    "shoes": [("Кроссовки", "Krossovka", ["Nike Air Max", "Adidas Ultraboost", "Puma RS-X", "New Balance 574"])],
    "kids": [
        ("Коляска", "Aravacha", ["Chicco", "Happy Baby"]),
        ("Конструктор", "Konstruktor", ["LEGO Classic", "LEGO City"]),
    ],
    "beauty": [
        ("Крем для лица", "Yuz kremi", ["Nivea", "La Roche-Posay", "CeraVe"]),
        ("Парфюм", "Atir", ["Dior", "Chanel", "Lattafa"]),
    ],
    "food": [("Чай", "Choy", ["Ahmad Tea", "Greenfield"]), ("Кофе", "Qahva", ["Jacobs", "Lavazza"])],
    "home": [
        ("Сковорода", "Tova", ["Tefal", "Rondell"]),
        ("Постельное бельё", "Choyshab to'plami", ["Home Line", "Ecotex"]),
    ],
    "sports": [("Велосипед", "Velosiped", ["Stels", "Forward"]), ("Гантели", "Gantel", ["Torneo", "Kettler"])],
}
_COLORS = [
    ("чёрный", "qora"),
    ("белый", "oq"),
    ("синий", "ko'k"),
    ("серый", "kulrang"),
    ("красный", "qizil"),
    ("зелёный", "yashil"),
]
_MEMORY = ["128 ГБ", "256 ГБ", "512 ГБ"]
_SIZES = ["S", "M", "L", "XL", "40", "41", "42", "43"]


def _round_price(value: float) -> int:
    return int(round(value / 1000) * 1000)


def generate_products(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate ``n`` products with ru/uz titles, attributes and UZS prices."""
    products: list[dict[str, Any]] = []
    slugs = [c.slug for c in CATEGORIES]
    for i in range(n):
        cat = CATEGORY_BY_SLUG[rng.choice(slugs)]
        ru_base, uz_base, brands = rng.choice(_PRODUCTS[cat.slug])
        brand = rng.choice(brands)
        color_ru, color_uz = rng.choice(_COLORS)
        attrs: dict[str, str] = {"color": color_ru}
        extra = ""
        if cat.slug == "smartphones":
            attrs["memory"] = rng.choice(_MEMORY)
            extra = f" {attrs['memory']}"
        elif cat.slug in {"clothes", "shoes"}:
            attrs["size"] = rng.choice(_SIZES)
        lo, hi = cat.price_range
        products.append(
            {
                "product_id": f"P{100000 + i}",
                "category": cat.slug,
                "brand": brand,
                "title_ru": f"{ru_base} {brand}{extra}, {color_ru}",
                "title_uz": f"{uz_base} {brand}{extra}, {color_uz}",
                "price": _round_price(rng.uniform(lo, hi)),
                "attrs": attrs,
                "rating": round(rng.uniform(3.6, 5.0), 1),
                "in_stock": rng.random() > 0.08,
            }
        )
    return products


def create_db(db_path: Path, products: list[dict[str, Any]], n_orders: int, rng: random.Random) -> None:
    """Create the SQLite DB with ``products``, ``orders`` and an empty ``returns`` table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE products (product_id TEXT PRIMARY KEY, category TEXT, brand TEXT, title_ru TEXT,
            title_uz TEXT, price INTEGER, color TEXT, rating REAL, in_stock INTEGER);
        CREATE TABLE orders (order_id TEXT PRIMARY KEY, product_id TEXT, region TEXT, status TEXT,
            created_at TEXT, eta TEXT, delivered_at TEXT, pickup_point TEXT, total INTEGER);
        CREATE TABLE returns (return_id TEXT PRIMARY KEY, order_id TEXT UNIQUE, reason TEXT, created_at TEXT);
        """)
    con.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (p["product_id"], p["category"], p["brand"], p["title_ru"], p["title_uz"], p["price"],
             p["attrs"]["color"], p["rating"], int(p["in_stock"]))
            for p in products
        ],
    )  # fmt: skip
    statuses = ["created", "paid", "assembling", "in_transit", "at_pickup", "delivered", "delivered", "cancelled"]
    orders = []
    for i in range(n_orders):
        product = rng.choice(products)
        region = rng.choice(REGIONS)
        status = rng.choice(statuses)
        transit = timedelta(days=rng.randint(*region.eta_days))
        delivered = None
        if status in {"created", "paid", "assembling", "in_transit"}:  # still on the way: ETA today or later
            eta = REFERENCE_DATE + timedelta(days=rng.randint(0, region.eta_days[1]))
            created = min(eta - transit, REFERENCE_DATE)
        elif status == "at_pickup":  # arrived within the pickup storage window
            eta = REFERENCE_DATE - timedelta(days=rng.randint(0, PICKUP_STORAGE_DAYS - 1))
            created = eta - transit
        else:  # delivered / cancelled: in the past
            created = REFERENCE_DATE - timedelta(days=rng.randint(int(transit.days) + 1, 30))
            eta = created + transit
            if status == "delivered":
                delivered = min(eta + timedelta(days=rng.randint(0, 1)), REFERENCE_DATE)
        pickup = f"{region.ru}, пункт выдачи №{rng.randint(1, 40)}"
        orders.append(
            (str(100000 + i * 7 % 900000), product["product_id"], region.ru, status, created.isoformat(),
             eta.isoformat(), delivered.isoformat() if delivered else None, pickup, product["price"])
        )  # fmt: skip
    con.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,?)", orders)
    con.commit()
    con.close()


# --------------------------------------------------------------------------- knowledge base


def _article(doc_id: str, lang: str, topic: str, title: str, text: str, questions: list[str]) -> dict[str, Any]:
    return {"doc_id": doc_id, "lang": lang, "topic": topic, "title": title, "text": text, "questions": questions}


def generate_kb() -> list[dict[str, Any]]:
    """Bilingual FAQ / policy articles. ``questions`` are used as the RAG eval set."""
    lo, hi = REFUND_DAYS
    free = f"{FREE_DELIVERY_FROM_UZS:_}".replace("_", " ")
    months = ", ".join(map(str, INSTALLMENT_MONTHS))
    arts: list[dict[str, Any]] = []

    for i, r in enumerate(REGIONS):
        a, b = r.eta_days
        eta_ru = f"{a} день" if a == b == 1 else f"{a}–{b} дня" if b < 5 else f"{a}–{b} дней"
        eta_uz = f"{a} kun" if a == b else f"{a}–{b} kun"
        arts.append(_article(
            f"delivery-{i}-ru", "ru", "delivery", f"Доставка: {r.ru}",
            f"Сроки доставки в регион «{r.ru}» — {eta_ru} с момента оплаты. Доставка бесплатная при заказе "
            f"от {free} сум, иначе стоимость рассчитывается при оформлении. Заказ можно "
            f"получить курьером или в пункте выдачи.",
            [f"Сколько идёт доставка в {r.ru}?", f"Когда привезут заказ в {r.ru}?", f"доставка {r.ru} сроки"],
        ))  # fmt: skip
        arts.append(_article(
            f"delivery-{i}-uz", "uz", "delivery", f"Yetkazib berish: {r.uz}",
            f"«{r.uz}» hududiga yetkazib berish muddati to'lovdan keyin {eta_uz}. {free} "
            f"so'mdan ortiq buyurtmalarda yetkazib berish bepul. Buyurtmani kuryer orqali yoki topshirish "
            f"punktidan olish mumkin.",
            [f"{r.uz}ga yetkazib berish necha kun?", f"{r.uz} buyurtma qachon keladi?", f"{r.uz} dostavka"],
        ))  # fmt: skip

    for c in CATEGORIES:
        if c.returnable:
            ru = (
                f"Товары категории «{c.ru}» можно вернуть в течение {RETURN_DAYS} дней после получения, если "
                f"сохранены товарный вид и упаковка. Деньги возвращаются на карту за {lo}–{hi} рабочих дня "
                f"после приёмки возврата."
            )
            uz = (
                f"«{c.uz}» toifasidagi mahsulotlarni olgandan keyin {RETURN_DAYS} kun ichida qaytarish mumkin, "
                f"agar tashqi ko'rinishi va qadog'i saqlangan bo'lsa. Pul qaytarish qabul qilingandan keyin "
                f"{lo}–{hi} ish kunida kartaga qaytariladi."
            )
        else:
            ru = (
                f"Товары категории «{c.ru}» надлежащего качества возврату не подлежат. Если товар бракованный "
                f"или пришёл не тот, напишите в поддержку в течение {RETURN_DAYS} дней — мы заменим его или "
                f"вернём деньги."
            )
            uz = (
                f"«{c.uz}» toifasidagi sifatli mahsulotlar qaytarilmaydi. Agar mahsulot nuqsonli bo'lsa yoki "
                f"boshqa mahsulot kelgan bo'lsa, {RETURN_DAYS} kun ichida qo'llab-quvvatlash xizmatiga yozing."
            )
        arts.append(
            _article(
                f"return-{c.slug}-ru",
                "ru",
                "return",
                f"Возврат: {c.ru}",
                ru,
                [f"Можно ли вернуть {c.ru.lower()}?", f"как вернуть товар из категории {c.ru.lower()}"],
            )
        )
        arts.append(_article(f"return-{c.slug}-uz", "uz", "return", f"Qaytarish: {c.uz}", uz,
                             [f"{c.uz}ni qaytarsa bo'ladimi?", f"{c.uz.lower()} qaytarish qoidasi"]))  # fmt: skip

    general: list[tuple[str, str, str, list[str], str, str, list[str]]] = [
        ("payment-methods", "Способы оплаты",
         "Оплатить заказ можно картами Uzcard, Humo, Visa и Mastercard, наличными курьеру или в рассрочку. "
         "Оплата картой проходит сразу при оформлении заказа.",
         ["Какие способы оплаты есть?", "можно оплатить humo?", "оплата наличными курьеру"],
         "To'lov usullari",
         "Buyurtmani Uzcard, Humo, Visa va Mastercard kartalari, kuryerga naqd pul yoki muddatli to'lov bilan "
         "to'lash mumkin.",
         ["Qanday to'lov usullari bor?", "humo bilan to'lasa bo'ladimi?", "naqd pul bilan to'lash"]),
        ("installments", "Рассрочка",
         f"Рассрочка доступна на {months} месяцев без первоначального взноса. Для оформления нужен паспорт и "
         f"карта. Решение приходит за несколько минут.",
         ["Как оформить рассрочку?", "на сколько месяцев рассрочка?", "рассрочка без первого взноса"],
         "Muddatli to'lov",
         f"Muddatli to'lov {months} oyga boshlang'ich to'lovsiz mavjud. Rasmiylashtirish uchun pasport va karta "
         f"kerak.",
         ["Muddatli to'lovni qanday rasmiylashtiraman?", "necha oyga bo'lib to'lash mumkin?", "nasiya shartlari"]),
        ("pickup-storage", "Хранение в пункте выдачи",
         f"Заказ хранится в пункте выдачи {PICKUP_STORAGE_DAYS} дней. После этого он возвращается на склад, а "
         f"деньги за оплаченный заказ возвращаются на карту.",
         ["Сколько хранится заказ в пункте выдачи?", "не успел забрать заказ что делать"],
         "Topshirish punktida saqlash",
         f"Buyurtma topshirish punktida {PICKUP_STORAGE_DAYS} kun saqlanadi. Shundan keyin omborga qaytariladi.",
         ["Buyurtma punktda necha kun turadi?", "buyurtmani olishga ulgurmadim"]),
        ("cancel-order", "Отмена заказа",
         "Заказ можно отменить в приложении до передачи в доставку: «Мои заказы» → заказ → «Отменить». Деньги "
         f"вернутся на карту за {lo}–{hi} рабочих дня.",
         ["Как отменить заказ?", "хочу отменить заказ", "отмена заказа деньги"],
         "Buyurtmani bekor qilish",
         "Buyurtmani yetkazib berishga topshirilgunga qadar ilovada bekor qilish mumkin: «Buyurtmalarim» → "
         "buyurtma → «Bekor qilish».",
         ["Buyurtmani qanday bekor qilaman?", "buyurtmani bekor qilmoqchiman"]),
        ("support-hours", "Часы работы поддержки",
         f"Поддержка работает ежедневно с {SUPPORT_HOURS}. Чат-ассистент отвечает круглосуточно.",
         ["Когда работает поддержка?", "часы работы оператора"],
         "Qo'llab-quvvatlash ish vaqti",
         f"Qo'llab-quvvatlash xizmati har kuni {SUPPORT_HOURS} ishlaydi. Chat-yordamchi kecha-kunduz javob beradi.",
         ["Operator qachon ishlaydi?", "qo'llab-quvvatlash ish vaqti"]),
        ("promo-codes", "Промокоды",
         "Промокод вводится в корзине перед оплатой. Один заказ — один промокод. Промокоды не действуют на "
         "товары со скидкой больше 50%.",
         ["Как применить промокод?", "промокод не работает", "куда вводить промокод"],
         "Promokodlar",
         "Promokod to'lovdan oldin savatchada kiritiladi. Bitta buyurtma — bitta promokod.",
         ["Promokodni qayerga yozaman?", "promokod ishlamayapti"]),
        ("warranty", "Гарантия",
         "На технику действует гарантия производителя от 12 месяцев. Гарантийный талон и чек хранятся в "
         "разделе «Мои заказы».",
         ["Какая гарантия на технику?", "где взять гарантийный талон"],
         "Kafolat",
         "Texnikaga ishlab chiqaruvchining kamida 12 oylik kafolati bor. Kafolat taloni «Buyurtmalarim» bo'limida.",
         ["Texnikaga kafolat bormi?", "kafolat taloni qayerda"]),
        ("change-address", "Изменение адреса доставки",
         "Адрес можно изменить, пока заказ не передан курьеру: «Мои заказы» → заказ → «Изменить адрес».",
         ["Как поменять адрес доставки?", "ошибся адресом в заказе"],
         "Manzilni o'zgartirish",
         "Buyurtma kuryerga topshirilmaguncha manzilni o'zgartirish mumkin: «Buyurtmalarim» → «Manzilni o'zgartirish».",
         ["Yetkazib berish manzilini qanday o'zgartiraman?", "manzilni xato yozdim"]),
        ("damaged-item", "Пришёл бракованный товар",
         f"Если товар пришёл с браком или повреждён, сфотографируйте его и напишите в поддержку в течение "
         f"{RETURN_DAYS} дней. Мы заменим товар или вернём деньги.",
         ["Пришёл бракованный товар что делать?", "товар сломан при доставке"],
         "Nuqsonli mahsulot keldi",
         f"Mahsulot nuqsonli kelgan bo'lsa, rasmga olib {RETURN_DAYS} kun ichida qo'llab-quvvatlashga yozing.",
         ["Nuqsonli mahsulot keldi nima qilaman?", "mahsulot singan holda keldi"]),
        ("seller-onboarding", "Как стать продавцом",
         "Зарегистрируйтесь в кабинете продавца, загрузите документы ИП или юрлица и подпишите оферту онлайн. "
         "Модерация занимает до 3 рабочих дней.",
         ["Как начать продавать на маркетплейсе?", "стать продавцом"],
         "Sotuvchi bo'lish",
         "Sotuvchi kabinetida ro'yxatdan o'ting, YaTT yoki yuridik shaxs hujjatlarini yuklang va ofertani imzolang.",
         ["Marketpleysda qanday sotaman?", "sotuvchi bo'lish"]),
    ]  # fmt: skip
    for slug, t_ru, x_ru, q_ru, t_uz, x_uz, q_uz in general:
        arts.append(_article(f"{slug}-ru", "ru", slug, t_ru, x_ru, q_ru))
        arts.append(_article(f"{slug}-uz", "uz", slug, t_uz, x_uz, q_uz))
    return arts
