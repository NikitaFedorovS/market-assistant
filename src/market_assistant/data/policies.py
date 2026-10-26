"""Single source of truth for marketplace rules.

Used by the KB generator (articles), the agent tools (eligibility checks) and the SFT filter
(answers that contradict these numbers are dropped).
"""

from __future__ import annotations

from dataclasses import dataclass

RETURN_DAYS = 10
REFUND_DAYS = (3, 5)  # business days to refund after the return is accepted
FREE_DELIVERY_FROM_UZS = 200_000
INSTALLMENT_MONTHS = (3, 6, 12)
PICKUP_STORAGE_DAYS = 5
SUPPORT_HOURS = "09:00–21:00"


@dataclass(frozen=True)
class Category:
    """Product category with bilingual names and return policy."""

    slug: str
    ru: str
    uz: str
    returnable: bool
    price_range: tuple[int, int]  # UZS


CATEGORIES: tuple[Category, ...] = (
    Category("smartphones", "Смартфоны", "Smartfonlar", True, (1_500_000, 18_000_000)),
    Category("laptops", "Ноутбуки", "Noutbuklar", True, (4_000_000, 30_000_000)),
    Category("headphones", "Наушники", "Quloqchinlar", True, (150_000, 4_000_000)),
    Category("tv", "Телевизоры", "Televizorlar", True, (2_500_000, 25_000_000)),
    Category("appliances", "Бытовая техника", "Maishiy texnika", True, (500_000, 12_000_000)),
    Category("clothes", "Одежда", "Kiyim-kechak", True, (80_000, 1_500_000)),
    Category("shoes", "Обувь", "Poyabzal", True, (150_000, 2_000_000)),
    Category("kids", "Детские товары", "Bolalar mollari", True, (50_000, 3_000_000)),
    Category("beauty", "Косметика", "Kosmetika", False, (30_000, 900_000)),
    Category("food", "Продукты", "Oziq-ovqat", False, (10_000, 400_000)),
    Category("home", "Товары для дома", "Uy uchun mollar", True, (40_000, 5_000_000)),
    Category("sports", "Спорт", "Sport", True, (60_000, 8_000_000)),
)
CATEGORY_BY_SLUG = {c.slug: c for c in CATEGORIES}


@dataclass(frozen=True)
class Region:
    """Delivery region with ETA in days."""

    ru: str
    uz: str
    eta_days: tuple[int, int]


REGIONS: tuple[Region, ...] = (
    Region("Ташкент", "Toshkent", (1, 1)),
    Region("Ташкентская область", "Toshkent viloyati", (1, 2)),
    Region("Самарканд", "Samarqand", (2, 3)),
    Region("Бухара", "Buxoro", (2, 4)),
    Region("Андижан", "Andijon", (2, 3)),
    Region("Фергана", "Farg'ona", (2, 3)),
    Region("Наманган", "Namangan", (2, 3)),
    Region("Наманган (районы)", "Namangan tumanlari", (3, 4)),
    Region("Кашкадарья", "Qashqadaryo", (3, 4)),
    Region("Сурхандарья", "Surxondaryo", (3, 5)),
    Region("Хорезм", "Xorazm", (3, 5)),
    Region("Навои", "Navoiy", (2, 4)),
    Region("Джизак", "Jizzax", (2, 3)),
    Region("Каракалпакстан", "Qoraqalpog'iston", (4, 5)),
)

ORDER_STATUSES: dict[str, tuple[str, str]] = {
    "created": ("оформлен", "rasmiylashtirildi"),
    "paid": ("оплачен", "to'landi"),
    "assembling": ("собирается на складе", "omborda yig'ilmoqda"),
    "in_transit": ("в пути", "yo'lda"),
    "at_pickup": ("ждёт в пункте выдачи", "topshirish punktida kutmoqda"),
    "delivered": ("доставлен", "yetkazib berildi"),
    "cancelled": ("отменён", "bekor qilindi"),
    "returned": ("возвращён", "qaytarildi"),
}
