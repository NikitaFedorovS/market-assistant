"""Synthetic support dialogs (ru / uz / mixed), router utterances and the RAG eval set.

Every assistant text turn carries both a dry operator reply (``content``) and a persona rendering
(``persona``) — see :mod:`market_assistant.data.persona`. Tool results come from the real
:class:`~market_assistant.agent.tools.MarketTools` over the synthetic DB.

A controlled share of noise is injected on purpose (duplicates, one-word replies, toxicity, policy
contradictions, PII, broken tool-call JSON) so that the cleaning pipeline has something to catch.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from market_assistant.agent.tools import MarketTools
from market_assistant.data.policies import CATEGORY_BY_SLUG, PICKUP_STORAGE_DAYS, REFUND_DAYS, RETURN_DAYS

Msg = dict[str, Any]

_MONTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
              "октября", "ноября", "декабря"]  # fmt: skip
_MONTHS_UZ = ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentabr", "oktabr",
              "noyabr", "dekabr"]  # fmt: skip


def fmt_date(iso: str | None, lang: str) -> str:
    """``2026-10-22`` -> ``22 октября`` / ``22-oktabr``."""
    if not iso:
        return "—"
    d = date.fromisoformat(iso)
    return f"{d.day} {_MONTHS_RU[d.month - 1]}" if lang != "uz" else f"{d.day}-{_MONTHS_UZ[d.month - 1]}"


def fmt_price(value: int, lang: str) -> str:
    """``1250000`` -> ``1 250 000 сум`` / ``1 250 000 so'm``."""
    num = f"{value:_}".replace("_", " ")
    return f"{num} so'm" if lang == "uz" else f"{num} сум"


def _call_id(rng: random.Random) -> str:
    return f"call_{rng.getrandbits(40):010x}"


def assistant(dry: str, persona: str) -> Msg:
    """Assistant text turn with dry operator reply + persona rendering."""
    return {"role": "assistant", "content": dry, "persona": persona}


def tool_call(rng: random.Random, name: str, args: dict[str, Any]) -> tuple[Msg, str]:
    """Assistant tool-call turn (OpenAI format, arguments as JSON string)."""
    cid = _call_id(rng)
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }
    return msg, cid


def tool_result(cid: str, name: str, result: dict[str, Any]) -> Msg:
    return {"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(result, ensure_ascii=False)}


def user(text: str) -> Msg:
    return {"role": "user", "content": text}


# --------------------------------------------------------------------------- phrase banks

ORDER_Q = {
    "ru": ["где мой заказ {id}?", "Заказ {id} когда приедет?", "подскажите статус заказа {id}", "что с заказом №{id}",
           "мой заказ {id} уже отправили?", "Добрый день! Хочу узнать, где посылка {id}", "{id} где сейчас?"],
    "uz": ["{id} buyurtmam qayerda?", "Buyurtma {id} qachon keladi?", "{id} raqamli buyurtma holati qanday?",
           "Salom, {id} buyurtmam yo'lga chiqdimi?", "{id} buyurtma qayerda hozir?"],
    "mixed": ["Salom, где мой заказ {id}?", "{id} buyurtma где сейчас?", "заказ {id} qachon keladi?",
              "Assalomu alaykum, мой заказ {id} отправили?"],
}  # fmt: skip
ORDER_Q_NOID = {
    "ru": ["где мой заказ?", "когда привезут мою посылку?", "хочу узнать статус заказа"],
    "uz": ["buyurtmam qayerda?", "posilkam qachon keladi?", "buyurtma holatini bilmoqchiman"],
    "mixed": ["Salom, заказ qachon keladi?", "buyurtmam где?", "где моя posilka?"],
}  # fmt: skip
GIVE_ID = {"ru": ["{id}", "номер {id}", "вот: {id}"], "uz": ["{id}", "raqami {id}"], "mixed": ["{id}", "raqami {id}"]}

RETURN_Q = {
    "ru": ["хочу вернуть заказ {id}, {reason}", "как оформить возврат по заказу {id}? {reason}",
           "Здравствуйте, нужен возврат заказа {id} — {reason}"],
    "uz": ["{id} buyurtmani qaytarmoqchiman, {reason}", "{id} buyurtma bo'yicha qaytarish rasmiylashtiring, {reason}"],
    "mixed": ["Salom, хочу qaytarish заказ {id}, {reason}", "{id} buyurtmani вернуть надо, {reason}"],
}  # fmt: skip
REASONS = {
    "ru": ["не подошёл размер", "не понравился цвет", "передумал", "товар не соответствует описанию"],
    "uz": ["o'lchami to'g'ri kelmadi", "rangi yoqmadi", "fikrimdan qaytdim", "tavsifga mos emas"],
}
CONFIRM = {"ru": ["да", "да, оформляй", "подтверждаю"], "uz": ["ha", "ha, rasmiylashtiring", "tasdiqlayman"],
           "mixed": ["ha, давай", "да, hop"]}  # fmt: skip
DECLINE = {"ru": ["нет, передумал", "не надо"], "uz": ["yo'q, kerak emas", "fikrimdan qaytdim"],
           "mixed": ["yo'q, не надо"]}  # fmt: skip

SEARCH_Q = {
    "ru": ["есть {q}?", "ищу {q} до {price}", "покажи {q}, цвет {color}", "хочу купить {q} недорого", "{q} в наличии?"],
    "uz": ["{q} bormi?", "{q} {price} gacha bormi?", "{color_uz} {q} kerak", "{q} sotib olmoqchiman"],
    "mixed": ["Salom, есть {q} {price} gacha?", "{q} bormi, цвет {color}?", "ищу {q}, arzonroq"],
}  # fmt: skip

CHITCHAT = {
    "ru": [("привет", "Salom! 👋 Я Dilkash, помогу с заказами, возвратами и поиском товаров. Чем помочь?"),
           ("как дела?", "Zo'r, rahmat! Готов помогать 🙂 Что ищешь или что случилось с заказом?"),
           ("спасибо!", "Rahmat тебе! Если что — я тут 🙂"),
           ("ты бот?", "Да, я ассистент Dilkash 🤖 Отвечаю сразу, а если нужно — подключу живого оператора."),
           ("доброе утро", "Доброе утро! ☀️ Чем могу помочь?"),
           ("расскажи анекдот", "Курьер звонит: «Вы дома?» — «Нет». — «А я уже у двери» 😄 Чем помочь с заказами?")],
    "uz": [("salom", "Salom! 👋 Men Dilkash, buyurtmalar, qaytarish va mahsulot qidirishda yordam beraman."),
           ("qalaysiz?", "Zo'r, rahmat! 🙂 Sizga qanday yordam bera olaman?"),
           ("rahmat!", "Arzimaydi! Savol bo'lsa — men shu yerdaman 🙂"),
           ("siz botmisiz?", "Ha, men Dilkash yordamchisiman 🤖 Kerak bo'lsa, operatorni ulayman.")],
    "mixed": [("Salom, как дела?", "Salom! Zo'r, rahmat 🙂 Чем помочь?"),
              ("rahmat, всё понятно", "Arzimaydi! Если что — я тут 🙂")],
}  # fmt: skip

HANDOFF = {
    "ru": [("позовите оператора", "Hop, подключаю оператора 🙂 Обычно отвечают за пару минут."),
           ("хочу поговорить с живым человеком", "Понял, переключаю на оператора. Он скоро ответит."),
           ("это уже третий раз, верните деньги!!! где оператор", "Понимаю, это очень неприятно. Сейчас "
            "подключу оператора — он разберётся с вашим вопросом в приоритете."),
           ("вы меня обманули, товар не тот, хочу жалобу", "Мне очень жаль, что так вышло. Подключаю старшего "
            "оператора, чтобы оформить жалобу и решить вопрос.")],
    "uz": [("operatorni chaqiring", "Xo'p, operatorni ulayapman 🙂 Odatda bir necha daqiqada javob berishadi."),
           ("tirik odam bilan gaplashmoqchiman", "Tushundim, operatorga ulayapman."),
           ("uchinchi marta yozyapman, pulimni qaytaring!", "Tushunaman, bu juda yoqimsiz. Hozir operatorni "
            "ulayman — u masalangizni birinchi navbatda hal qiladi.")],
    "mixed": [("оператор kerak", "Hop, подключаю оператора 🙂"),
              ("odam bilan gaplashmoqchiman, не бот", "Понял, переключаю на оператора.")],
}  # fmt: skip

UNKNOWN_FAQ = {
    "ru": ["можно оплатить биткоином?", "у вас есть подарочные сертификаты?", "можно ли забронировать товар?"],
    "uz": ["bitkoin bilan to'lasa bo'ladimi?", "sovg'a sertifikati bormi?"],
    "mixed": ["Salom, можно оплатить kriptovalyuta?"],
}  # fmt: skip

TOXIC = ["ты тупой бот", "идиоты, а не магазин", "ahmoq bot"]
PII_SNIPPETS = ["мой номер +998 90 123 45 67", "карта 8600 1234 5678 9012", "telefonim 99 876 54 32",
                "пишите на test.user@mail.uz"]  # fmt: skip


# --------------------------------------------------------------------------- persona helpers


def _ru_open(rng: random.Random, lang: str) -> str:
    if lang == "mixed":
        return rng.choice(["Salom! ", "Hop, ", "Bir daqiqa, "])
    return rng.choice(["Сейчас гляну 👀 ", "", "Hop, "])


def _uz_open(rng: random.Random) -> str:
    return rng.choice(["Hozir qarab beraman 👀 ", "", "Xo'p, "])


def order_final(rng: random.Random, res: dict[str, Any], lang: str) -> Msg:
    """Final reply for get_order_status (dry + persona)."""
    oid = res.get("order_id")
    if "error" in res:
        if lang == "uz":
            return assistant(
                f"{oid} raqamli buyurtma topilmadi.",
                f"Hmm, {oid} buyurtmani topa olmadim 🤔 Raqamni tekshirib ko'ring — u «Buyurtmalarim» bo'limida.",
            )
        return assistant(f"Заказ {oid} не найден.",
                         f"Хм, заказ {oid} не нахожу 🤔 Проверь, пожалуйста, номер — он есть в «Мои заказы».")  # fmt: skip
    st = res["status"]
    if lang == "uz":
        eta, done = fmt_date(res["eta"], "uz"), fmt_date(res["delivered_at"], "uz")
        dry = f"{oid} buyurtma holati: {res['status_uz']}. Taxminiy yetkazish: {eta}."
        persona = {
            "delivered": f"{oid} buyurtma {done} kuni yetkazib berildi — {res['product_uz']}. Muammo bo'lsa, yozing 🙂",
            "in_transit": f"{_uz_open(rng)}{oid} buyurtma yo'lda, {eta} kuni kutilmoqda.",
            "at_pickup": f"Zo'r xabar: {oid} buyurtma sizni kutmoqda — {res['pickup_point']}. "
            f"{PICKUP_STORAGE_DAYS} kun ichida olib keting.",
            "cancelled": f"{oid} buyurtma bekor qilingan. Pul qaytmagan bo'lsa, yozing — tekshiraman.",
            "returned": f"{oid} buyurtma bo'yicha qaytarish rasmiylashtirilgan.",
        }.get(st, f"{_uz_open(rng)}{oid} buyurtma {res['status_uz']}, taxminiy yetkazish — {eta}.")
        return assistant(dry, persona)
    eta, done = fmt_date(res["eta"], "ru"), fmt_date(res["delivered_at"], "ru")
    op = _ru_open(rng, lang)
    tail = " Rahmat за терпение!" if lang == "mixed" else ""
    dry = f"Статус заказа {oid}: {res['status_ru']}. Ожидаемая дата доставки: {eta}."
    persona = {
        "delivered": f"{op}Заказ {oid} уже доставлен {done} — {res['product_ru']}. Если что-то не так, напиши 🙂",
        "in_transit": f"{op}Заказ {oid} в пути, ожидаем {eta}.{tail}",
        "at_pickup": f"Zo'r новость: заказ {oid} уже ждёт тебя — {res['pickup_point']}. Забрать можно в течение "
                     f"{PICKUP_STORAGE_DAYS} дней.",
        "cancelled": f"{op}Заказ {oid} отменён. Если деньги ещё не вернулись — напиши, проверю.",
        "returned": f"{op}По заказу {oid} уже оформлен возврат.",
    }.get(st, f"{op}Заказ {oid} {res['status_ru']}, ориентировочная доставка — {eta}.{tail}")  # fmt: skip
    return assistant(dry, persona)


def return_final(res: dict[str, Any], order_id: str, lang: str) -> Msg:
    """Final reply for create_return."""
    lo, hi = REFUND_DAYS
    uz = lang == "uz"
    if "error" not in res and res.get("status") == "created":
        if uz:
            return assistant(
                f"Qaytarish {res['return_id']} rasmiylashtirildi.",
                f"Tayyor ✅ Qaytarish {res['return_id']} rasmiylashtirildi. Pul {lo}–{hi} ish kunida kartaga qaytadi.",
            )
        return assistant(f"Возврат {res['return_id']} оформлен. Срок возврата денег: {lo}–{hi} рабочих дня.",
                         f"Готово ✅ Возврат {res['return_id']} оформлен. Деньги вернутся на карту за {lo}–{hi} "
                         f"рабочих дня после приёмки товара.")  # fmt: skip
    reason = res.get("reason")
    ru = {
        "period_expired": (f"Срок возврата истёк ({RETURN_DAYS} дней).",
                           f"Ой, срок возврата уже прошёл: вернуть можно в течение {RETURN_DAYS} дней после "
                           f"получения. Если товар бракованный — подключу оператора."),
        "non_returnable_category": ("Товары этой категории возврату не подлежат.",
                                    "К сожалению, товары этой категории надлежащего качества не возвращаются. "
                                    "Если есть брак — напиши, подключу оператора."),
        "not_delivered": (f"Заказ {order_id} ещё не доставлен.",
                          f"Заказ {order_id} ещё не доставлен, поэтому возврат оформить нельзя. Можно отменить "
                          f"заказ в «Мои заказы»."),
        "already_requested": (f"Возврат по заказу {order_id} уже оформлен.",
                              f"По заказу {order_id} возврат уже оформлен — {res.get('return_id')}."),
        "not_found": (f"Заказ {order_id} не найден.", f"Не нахожу заказ {order_id} 🤔 Проверь номер, пожалуйста."),
    }  # fmt: skip
    uzt = {
        "period_expired": (f"Qaytarish muddati o'tgan ({RETURN_DAYS} kun).",
                           f"Afsuski, qaytarish muddati o'tib ketgan: olgandan keyin {RETURN_DAYS} kun ichida "
                           f"qaytarish mumkin. Nuqson bo'lsa — operatorni ulayman."),
        "non_returnable_category": ("Bu toifadagi mahsulotlar qaytarilmaydi.",
                                    "Afsuski, bu toifadagi sifatli mahsulotlar qaytarilmaydi. Nuqson bo'lsa, yozing."),
        "not_delivered": (f"{order_id} buyurtma hali yetkazilmagan.",
                          f"{order_id} buyurtma hali yetkazilmagan, shuning uchun qaytarib bo'lmaydi."),
        "already_requested": (f"{order_id} bo'yicha qaytarish allaqachon bor.",
                              f"{order_id} bo'yicha qaytarish allaqachon rasmiylashtirilgan."),
        "not_found": (f"{order_id} topilmadi.", f"{order_id} buyurtmani topa olmadim 🤔 Raqamni tekshiring."),
    }  # fmt: skip
    dry, persona = (uzt if uz else ru)[reason or "not_found"]
    return assistant(dry, persona)


# --------------------------------------------------------------------------- generator


class DialogGenerator:
    """Builds dialogs for each intent using the synthetic DB, catalog and KB."""

    def __init__(self, db_path: Path, kb: list[dict[str, Any]], products: list[dict[str, Any]],
                 today: date, seed: int = 42) -> None:  # fmt: skip
        self.rng = random.Random(seed)
        self.tools = MarketTools(db_path, today, dry_run=True)
        self.kb = kb
        self.kb_by_lang = {lang: [a for a in kb if a["lang"] == lang] for lang in ("ru", "uz")}
        self.products = products
        con = sqlite3.connect(db_path)
        self.orders_by_status: dict[str, list[str]] = {}
        for oid, st in con.execute("SELECT order_id, status FROM orders"):
            self.orders_by_status.setdefault(st, []).append(oid)
        con.close()
        self.all_orders = [o for ids in self.orders_by_status.values() for o in ids]

    def _lang(self) -> str:
        return self.rng.choices(["ru", "uz", "mixed"], weights=[0.45, 0.30, 0.25])[0]

    def _fake_order_id(self) -> str:
        return str(self.rng.randint(900000, 999999))

    # ------------------------------------------------------------- intents

    def order_status(self, lang: str) -> list[Msg]:
        r = self.rng
        oid = self._fake_order_id() if r.random() < 0.1 else r.choice(self.all_orders)
        msgs: list[Msg] = []
        if r.random() < 0.25:
            msgs.append(user(r.choice(ORDER_Q_NOID[lang])))
            if lang == "uz":
                msgs.append(
                    assistant(
                        "Buyurtma raqamini yozing.",
                        "Hozir qarab beraman 👀 Buyurtma raqamini yozib yuboring — u «Buyurtmalarim» bo'limida.",
                    )
                )
            else:
                msgs.append(
                    assistant(
                        "Укажите номер заказа.",
                        f"{_ru_open(r, lang)}Подскажи номер заказа — он есть в разделе «Мои заказы».",
                    )
                )
            msgs.append(user(r.choice(GIVE_ID[lang]).format(id=oid)))
        else:
            msgs.append(user(r.choice(ORDER_Q[lang]).format(id=oid)))
        call, cid = tool_call(r, "get_order_status", {"order_id": oid})
        res = self.tools.call("get_order_status", {"order_id": oid})
        msgs += [call, tool_result(cid, "get_order_status", res), order_final(r, res, lang)]
        return msgs

    def return_flow(self, lang: str) -> list[Msg]:
        r = self.rng
        pool = r.choice([self.orders_by_status.get("delivered", []), self.orders_by_status.get("delivered", []),
                         self.orders_by_status.get("in_transit", []), self.all_orders])  # fmt: skip
        oid = r.choice(pool or self.all_orders)
        reason = r.choice(REASONS["uz" if lang == "uz" else "ru"])
        msgs: list[Msg] = [user(r.choice(RETURN_Q[lang]).format(id=oid, reason=reason))]
        if lang == "uz":
            msgs.append(
                assistant(
                    f"{oid} buyurtma bo'yicha qaytarishni rasmiylashtiraymi? Tasdiqlang.",
                    f"Xo'p, rasmiylashtiramiz 🙂 {oid} buyurtma, sabab — «{reason}». Tasdiqlaysizmi?",
                )
            )
        else:
            msgs.append(
                assistant(
                    f"Оформить возврат по заказу {oid}? Подтвердите.",
                    f"Hop, давай оформим 🙂 Возврат по заказу {oid}, причина — «{reason}». Подтверждаешь?",
                )
            )
        if r.random() < 0.15:
            msgs.append(user(r.choice(DECLINE[lang])))
            msgs.append(
                assistant("Возврат не оформлен.", "Xo'p, ничего не оформляю. Если передумаешь — я тут 🙂")
                if lang != "uz"
                else assistant(
                    "Qaytarish rasmiylashtirilmadi.",
                    "Xo'p, hech narsa rasmiylashtirmayman. Fikringiz o'zgarsa — yozing 🙂",
                )
            )
            return msgs  # fmt: skip
        msgs.append(user(r.choice(CONFIRM[lang])))
        args = {"order_id": oid, "reason": reason}
        call, cid = tool_call(r, "create_return", args)
        res = self.tools.call("create_return", args)
        msgs += [call, tool_result(cid, "create_return", res), return_final(res, oid, lang)]
        return msgs

    def product_search(self, lang: str) -> list[Msg]:
        r = self.rng
        p = r.choice(self.products)
        cat = CATEGORY_BY_SLUG[p["category"]]
        base_ru = p["title_ru"].split(",")[0].replace(p["brand"], "").strip().lower()
        base_uz = p["title_uz"].split(",")[0].replace(p["brand"], "").strip().lower()
        q = f"{base_uz} {p['brand']}" if lang == "uz" else f"{base_ru} {p['brand']}"
        budget = int(p["price"] * r.uniform(0.9, 1.4) // 100_000 * 100_000) or 100_000
        color = p["attrs"]["color"]
        template = r.choice(SEARCH_Q[lang])
        price_txt = fmt_price(budget, "uz" if "gacha" in template else "ru")
        text = template.format(q=q, price=price_txt, color=color, color_uz=p["title_uz"].rsplit(", ", 1)[-1])
        args: dict[str, Any] = {"query": q, "category": cat.slug}
        if "{price}" in template:
            args["max_price"] = budget
        if "{color" in template:
            args["color"] = color
        msgs: list[Msg] = [user(text)]
        call, cid = tool_call(r, "search_products", args)
        res = self.tools.call("search_products", args)
        msgs += [call, tool_result(cid, "search_products", res)]
        items = res["items"][:3]
        out_lang = "uz" if lang == "uz" else "ru"
        title = "title_uz" if out_lang == "uz" else "title_ru"
        lines = "\n".join(f"{i + 1}. {it[title]} — {fmt_price(it['price'], out_lang)}" for i, it in enumerate(items))
        if not items:
            if out_lang == "uz":
                msgs.append(
                    assistant(
                        "Bunday parametrlar bo'yicha mahsulot topilmadi.",
                        "Hmm, bunday parametrlar bo'yicha hozircha hech narsa yo'q 🤔 Byudjet yoki rangni o'zgartiramizmi?",
                    )
                )
            else:
                msgs.append(assistant("По заданным параметрам товаров нет.",
                                      "Хм, по таким параметрам сейчас ничего нет в наличии 🤔 Может, поднимем бюджет "
                                      "или посмотрим другой цвет?"))  # fmt: skip
        elif out_lang == "uz":
            msgs.append(
                assistant(
                    f"Topilgan mahsulotlar:\n{lines}", f"Zo'r tanlov! Mana variantlar:\n{lines}\nYana ko'rsataymi?"
                )
            )
        else:
            msgs.append(
                assistant(f"Найденные товары:\n{lines}", f"Zo'r выбор! Нашёл варианты:\n{lines}\nПоказать ещё?")
            )
        return msgs

    def faq(self, lang: str) -> tuple[list[Msg], list[str]]:
        r = self.rng
        kb_lang = "uz" if lang == "uz" else "ru"
        if r.random() < 0.1:  # question not covered by the KB -> honest "don't know"
            ctx = r.sample(self.kb_by_lang[kb_lang], 2)
            q = r.choice(UNKNOWN_FAQ[lang])
            if kb_lang == "uz":
                reply = assistant(
                    "Bu bo'yicha ma'lumot yo'q.",
                    "Rostini aytsam, bu bo'yicha aniq ma'lumotim yo'q 🙏 Operatorni ulaymi?",
                )
            else:
                reply = assistant(
                    "Информации по этому вопросу нет.",
                    "Честно, точной информации по этому у меня нет 🙏 Могу подключить оператора — он подскажет.",
                )
            return [user(q), reply], [c["doc_id"] for c in ctx]  # fmt: skip
        art = r.choice(self.kb_by_lang[kb_lang])
        distractor = r.choice([a for a in self.kb_by_lang[kb_lang] if a["topic"] != art["topic"]])
        ctx_ids = [art["doc_id"], distractor["doc_id"]]
        r.shuffle(ctx_ids)
        q = r.choice(art["questions"])
        if lang == "mixed":
            q = r.choice(["Salom, ", "Assalomu alaykum, ", ""]) + q
        sentences = [s.strip() for s in art["text"].split(". ") if s.strip()]
        core = ". ".join(sentences[:2]).rstrip(".") + "."
        if kb_lang == "uz":
            opener = r.choice(["", "Xo'p, ", "Albatta! "])
            persona = f"{opener}{core} Yana savol bo'lsa — yozing 🙂"
        else:
            opener = r.choice(["", "Hop, ", "Конечно! "])
            closer = r.choice(["Если что — я тут 🙂", "Ещё вопросы?", ""])
            persona = f"{opener}{core} {closer}".strip()
        return [user(q), assistant(art["text"], persona)], ctx_ids

    def chitchat(self, lang: str) -> list[Msg]:
        q, a = self.rng.choice(CHITCHAT[lang])
        return [user(q), assistant(a, a)]

    def handoff(self, lang: str) -> list[Msg]:
        q, a = self.rng.choice(HANDOFF[lang])
        return [user(q), assistant(a, a)]

    # ------------------------------------------------------------- noise

    def _inject_noise(self, d: dict[str, Any]) -> dict[str, Any]:
        r = self.rng
        roll = r.random()
        msgs = d["messages"]
        if roll < 0.02:
            last = next(m for m in reversed(msgs) if m["role"] == "assistant" and m.get("content"))
            last["content"] = last["persona"] = "Ок."
            d["noise"] = "short"
        elif roll < 0.04:
            msgs[0]["content"] += f" {r.choice(TOXIC)}"
            d["noise"] = "toxic"
        elif roll < 0.06 and d["intent"] in {"return", "faq"}:
            last = msgs[-1]
            last["persona"] = last["content"] = last["persona"] + " Вернуть товар можно в течение 30 дней."
            d["noise"] = "contradiction"
        elif roll < 0.10:
            msgs[0]["content"] += f", {r.choice(PII_SNIPPETS)}"
            d["noise"] = "pii"
        elif roll < 0.115:
            calls = [m for m in msgs if m.get("tool_calls")]
            if calls:
                calls[0]["tool_calls"][0]["function"]["arguments"] = '{"order_id": "1234'
                d["noise"] = "broken_json"
        return d

    def generate(self, per_intent: int) -> list[dict[str, Any]]:
        """Generate ``per_intent`` dialogs for each intent (+ ~3% exact duplicates)."""
        out: list[dict[str, Any]] = []
        builders = {
            "order_status": self.order_status,
            "return": self.return_flow,
            "product_question": self.product_search,
            "chitchat": self.chitchat,
            "human_handoff": self.handoff,
        }
        n = 0
        for intent in ("order_status", "return", "product_question", "faq", "chitchat", "human_handoff"):
            for _ in range(per_intent):
                lang = self._lang()
                ctx: list[str] = []
                if intent == "faq":
                    msgs, ctx = self.faq(lang)
                else:
                    msgs = builders[intent](lang)
                d = self._inject_noise({"id": f"d{n:06d}", "intent": intent, "lang": lang,
                                        "context_doc_ids": ctx, "messages": msgs})  # fmt: skip
                out.append(d)
                if self.rng.random() < 0.03:
                    out.append(json.loads(json.dumps({**d, "id": f"d{n:06d}-dup"})))
                n += 1
        self.rng.shuffle(out)
        return out


# --------------------------------------------------------------------------- router + RAG eval


def _augment(text: str, rng: random.Random) -> str:
    """Light noise typical for chat: lowercase, dropped punctuation, a typo."""
    if rng.random() < 0.4:
        text = text.lower()
    if rng.random() < 0.3:
        text = text.rstrip("?!.")
    if rng.random() < 0.2 and len(text) > 6:
        i = rng.randrange(1, len(text) - 2)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    return text


def router_samples(dialogs: list[dict[str, Any]], kb: list[dict[str, Any]], per_intent: int,
                   seed: int = 42) -> list[dict[str, Any]]:  # fmt: skip
    """Intent-labelled first user messages, balanced and augmented."""
    rng = random.Random(seed)
    by_intent: dict[str, list[str]] = {}
    for d in dialogs:
        if d.get("noise") in {"toxic", "pii"}:
            continue
        by_intent.setdefault(d["intent"], []).append(d["messages"][0]["content"])
    by_intent.setdefault("faq", []).extend(q for a in kb for q in a["questions"])
    for lang_bank in UNKNOWN_FAQ.values():
        by_intent["faq"].extend(lang_bank)
    samples = []
    for intent, texts in by_intent.items():
        for _ in range(per_intent):
            samples.append({"text": _augment(rng.choice(texts), rng), "label": intent})
    rng.shuffle(samples)
    return samples


def rag_eval_set(kb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(question, relevant doc_id) pairs from the KB article question banks."""
    return [{"question": q, "doc_id": a["doc_id"], "lang": a["lang"]} for a in kb for q in a["questions"]]
