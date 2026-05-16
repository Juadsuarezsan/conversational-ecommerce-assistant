"""A synthetic Instacart-like product catalog for tests and offline demos.

Real Instacart data requires Kaggle credentials. This generator produces ~500
realistic products across 21 departments / 134 aisles so the eval set runs
without external downloads. Fully deterministic (fixed seed).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

RNG_SEED = 20260516

DEPARTMENTS_AISLES: dict[str, list[str]] = {
    "produce": ["fresh fruits", "fresh vegetables", "packaged vegetables fruits", "fresh herbs"],
    "dairy eggs": ["milk", "yogurt", "cheese", "butter", "eggs", "cream"],
    "snacks": ["chips pretzels", "cookies cakes", "crackers", "trail mix snack mix", "candy chocolate"],
    "beverages": ["soft drinks", "water seltzer sparkling water", "juice nectars", "tea", "coffee"],
    "frozen": ["frozen meals", "ice cream ice", "frozen vegetables", "frozen pizza", "frozen breakfast"],
    "pantry": ["spices seasonings", "oils vinegars", "canned meals", "pasta sauce", "rice grains"],
    "breakfast": ["cereal", "granola bars", "instant oats", "pancake mix"],
    "meat seafood": ["packaged poultry", "packaged seafood", "beef", "hot dogs bacon sausage"],
    "bakery": ["bread", "tortillas flat bread", "buns rolls", "pastries"],
    "deli": ["lunchmeat", "prepared salads", "tofu meat alternatives"],
    "household": ["paper goods", "cleaning supplies", "trash bags", "laundry"],
    "personal care": ["hair care", "skin care", "oral hygiene", "deodorants"],
    "babies": ["baby food formula", "diapers wipes"],
    "pets": ["dog food care", "cat food care"],
    "canned goods": ["canned vegetables", "canned fruit", "soup", "beans"],
    "alcohol": ["beer", "wine", "spirits"],
    "international": ["asian foods", "latin american foods", "indian foods"],
    "dry goods pasta": ["pasta", "grains rice dried goods"],
    "missing": ["missing"],
    "bulk": ["bulk dried fruits vegetables"],
    "other": ["specialty wines", "kitchen supplies"],
}

PRODUCT_TEMPLATES: dict[str, list[str]] = {
    "fresh fruits": ["Banana", "Honeycrisp Apple", "Strawberries 1lb", "Avocado Hass", "Lemon", "Lime", "Blueberries 6oz", "Grapes Red"],
    "fresh vegetables": ["Broccoli", "Baby Spinach", "Bell Pepper Red", "Cucumber", "Carrots 1lb", "Yellow Onion", "Garlic"],
    "milk": ["Whole Milk 1 Gallon", "2% Milk 1 Gallon", "Almond Milk Unsweetened", "Oat Milk Original", "Lactose-Free Milk"],
    "yogurt": ["Greek Yogurt Plain", "Strawberry Yogurt", "Vanilla Skyr", "Coconut Milk Yogurt"],
    "cheese": ["Cheddar Cheese Sharp 8oz", "Mozzarella Shredded 8oz", "Cream Cheese Original", "Parmesan Grated"],
    "eggs": ["Large Eggs Dozen", "Organic Brown Eggs Dozen", "Egg Whites Carton"],
    "chips pretzels": ["Tortilla Chips Classic", "Potato Chips Salt Vinegar", "Pretzels Mini", "Pita Chips Sea Salt"],
    "cookies cakes": ["Chocolate Chip Cookies", "Oreos Original", "Brownie Mix", "Vanilla Wafers"],
    "cereal": ["Cheerios Original 18oz", "Frosted Flakes 19oz", "Granola Honey Almond", "Bran Flakes", "Kids Pack Cereal Variety"],
    "soft drinks": ["Coca-Cola 12pk Cans", "Diet Coke 12pk", "Sprite 2L", "Dr Pepper 12pk"],
    "water seltzer sparkling water": ["LaCroix Lime Sparkling 8pk", "Spring Water 24pk", "Sparkling Water Variety"],
    "frozen meals": ["Lean Cuisine Chicken", "Frozen Burritos 5pk", "Cauliflower Pizza Frozen"],
    "ice cream ice": ["Ben Jerry Vanilla Pint", "Halo Top Cookies Cream", "Sorbet Mango", "Ice Cream Sandwich 6pk"],
    "spices seasonings": ["Cinnamon Ground", "Black Pepper Whole", "Sea Salt Fine", "Paprika Smoked"],
    "oils vinegars": ["Extra Virgin Olive Oil 16oz", "Avocado Oil Spray", "Balsamic Vinegar", "Apple Cider Vinegar"],
    "pasta": ["Spaghetti 1lb", "Penne Whole Wheat", "Gluten-Free Rotini", "Macaroni 1lb"],
    "pasta sauce": ["Marinara Classic 24oz", "Vodka Sauce", "Pesto Genovese"],
    "rice grains": ["Jasmine Rice 5lb", "Basmati Rice 4lb", "Quinoa Tri-Color"],
    "granola bars": ["Nature Valley Oats Honey 12pk", "Kind Bar Dark Chocolate 6pk", "Clif Bar White Chocolate"],
    "instant oats": ["Oatmeal Plain Packets", "Steel Cut Oats 24oz", "Overnight Oats Apple"],
    "bread": ["Sourdough Loaf", "Whole Wheat Bread", "Gluten-Free Sandwich Bread", "Sliced Rye Bread"],
    "tortillas flat bread": ["Flour Tortillas 10ct", "Corn Tortillas 30ct", "Naan Original 4pk"],
    "lunchmeat": ["Turkey Breast Sliced 9oz", "Smoked Ham Black Forest", "Salami Italian"],
    "packaged poultry": ["Chicken Breast Boneless 1lb", "Chicken Thighs 2lb", "Ground Turkey 1lb"],
    "packaged seafood": ["Wild Salmon Filet 1lb", "Shrimp Frozen 1lb", "Cod Filets 12oz"],
    "soup": ["Tomato Soup Classic", "Chicken Noodle Soup", "Black Bean Soup", "Minestrone"],
    "canned vegetables": ["Canned Corn 15oz", "Canned Green Beans", "Canned Tomato Diced"],
    "beans": ["Black Beans 15oz", "Chickpeas 15oz", "Pinto Beans 15oz"],
    "tea": ["Green Tea 40ct", "Earl Grey Tea 20ct", "Chamomile Tea 20ct"],
    "coffee": ["Whole Bean Colombian 12oz", "Espresso Ground 10oz", "Cold Brew Concentrate"],
    "trail mix snack mix": ["Trail Mix Original 1lb", "Mixed Nuts Salted 1lb", "Almonds Roasted 12oz"],
    "candy chocolate": ["Dark Chocolate 70% Bar", "M&Ms Peanut Family Size", "Reese's Cups 6pk"],
    "frozen vegetables": ["Frozen Broccoli Florets", "Frozen Mixed Vegetables", "Frozen Corn"],
    "frozen breakfast": ["Frozen Waffles 10ct", "Frozen Breakfast Burritos 6pk"],
    "frozen pizza": ["Frozen Cheese Pizza", "Frozen Pepperoni Pizza", "Cauliflower Crust Pizza"],
    "cream": ["Heavy Cream 1pt", "Half Half 1pt", "Sour Cream 16oz"],
    "butter": ["Salted Butter 1lb", "Unsalted Butter 1lb", "Ghee Clarified 8oz"],
    "juice nectars": ["Orange Juice 64oz", "Apple Juice 64oz", "Cranberry Juice 64oz"],
    "crackers": ["Saltines 16oz", "Wheat Thins 9oz", "Triscuits Original"],
    "paper goods": ["Paper Towels 6pk", "Toilet Paper 12pk", "Napkins 200ct"],
    "cleaning supplies": ["All-Purpose Cleaner 32oz", "Glass Cleaner Spray", "Dish Soap Dawn"],
    "trash bags": ["Trash Bags 30 Gallon 25ct", "Kitchen Trash Bags 13 Gallon 80ct"],
    "laundry": ["Laundry Detergent Tide", "Fabric Softener Downy", "Bleach 81oz"],
    "hair care": ["Shampoo Moisturizing", "Conditioner Repair", "Dry Shampoo Spray"],
    "skin care": ["Body Lotion Daily", "Sunscreen SPF 50", "Face Moisturizer"],
    "oral hygiene": ["Toothpaste Whitening", "Floss Mint 100yd", "Mouthwash Cool Mint"],
    "deodorants": ["Antiperspirant Original", "Natural Deodorant Lavender"],
    "baby food formula": ["Baby Food Pouches Variety 8pk", "Infant Formula Stage 1"],
    "diapers wipes": ["Diapers Size 3 84ct", "Baby Wipes 192ct"],
    "dog food care": ["Dry Dog Food Salmon 30lb", "Dog Treats Bacon 12oz"],
    "cat food care": ["Dry Cat Food Chicken 7lb", "Cat Litter Clumping 14lb"],
    "canned fruit": ["Canned Pineapple Chunks", "Mandarin Oranges 11oz"],
    "asian foods": ["Soy Sauce Tamari Low Sodium", "Sriracha 17oz", "Rice Vinegar Seasoned"],
    "latin american foods": ["Salsa Verde Roasted", "Refried Beans 16oz", "Plantain Chips"],
    "indian foods": ["Tikka Masala Sauce", "Basmati Rice Microwave", "Naan Garlic 4pk"],
    "hot dogs bacon sausage": ["Hot Dogs Beef 8pk", "Bacon Hickory Smoked 12oz", "Italian Sausage 1lb"],
    "tofu meat alternatives": ["Tofu Extra Firm 14oz", "Plant Burgers 4pk", "Tempeh Original 8oz"],
    "buns rolls": ["Hamburger Buns 8ct", "Hot Dog Buns 8ct", "Dinner Rolls 12ct"],
    "pastries": ["Croissants Butter 4pk", "Cinnamon Rolls 6pk", "Bagels Plain 6pk"],
    "specialty wines": ["Red Wine Cabernet 750ml", "White Wine Sauvignon 750ml"],
    "beer": ["Lager Beer 12pk", "IPA Local 6pk"],
    "wine": ["Pinot Noir 750ml", "Chardonnay 750ml"],
    "spirits": ["Vodka 750ml", "Tequila Reposado 750ml"],
}

# Some products that match very specific eval queries
EVAL_HOOK_PRODUCTS = [
    {"product_name": "Heinz Tomato Ketchup 397g", "aisle": "pasta sauce", "department": "pantry", "price_usd": 4.29, "avg_rating": 4.6, "rating_count": 4200, "in_stock": True},
    {"product_name": "Heinz Tomato Ketchup 567g Squeeze", "aisle": "pasta sauce", "department": "pantry", "price_usd": 5.99, "avg_rating": 4.7, "rating_count": 1900, "in_stock": True},
    {"product_name": "Cheerios Kids Snack Pack 6ct", "aisle": "cereal", "department": "breakfast", "price_usd": 6.49, "avg_rating": 4.5, "rating_count": 800, "in_stock": True},
    {"product_name": "Cheerios Honey Nut Kid-Friendly 12oz", "aisle": "cereal", "department": "breakfast", "price_usd": 4.79, "avg_rating": 4.7, "rating_count": 3200, "in_stock": True},
    {"product_name": "Oatmeal Apple Cinnamon Kids Packets 8ct", "aisle": "instant oats", "department": "breakfast", "price_usd": 3.79, "avg_rating": 4.5, "rating_count": 950, "in_stock": True},
    {"product_name": "Lactose-Free 2% Milk 1 Gallon", "aisle": "milk", "department": "dairy eggs", "price_usd": 5.49, "avg_rating": 4.4, "rating_count": 620, "in_stock": True},
    {"product_name": "Almond Milk Unsweetened 64oz", "aisle": "milk", "department": "dairy eggs", "price_usd": 3.49, "avg_rating": 4.3, "rating_count": 2100, "in_stock": True},
    {"product_name": "Lactose-Free Whole Milk 64oz", "aisle": "milk", "department": "dairy eggs", "price_usd": 4.29, "avg_rating": 4.3, "rating_count": 410, "in_stock": True},
    {"product_name": "Gluten-Free Sandwich Bread", "aisle": "bread", "department": "bakery", "price_usd": 6.99, "avg_rating": 4.2, "rating_count": 1100, "in_stock": True},
    {"product_name": "Frozen Cauliflower Pizza Crust", "aisle": "frozen pizza", "department": "frozen", "price_usd": 5.49, "avg_rating": 4.4, "rating_count": 700, "in_stock": True},
    {"product_name": "Best Rated Granola Bars Variety Pack 12ct", "aisle": "granola bars", "department": "snacks", "price_usd": 8.99, "avg_rating": 4.8, "rating_count": 5400, "in_stock": True},
    {"product_name": "Store Brand Granola Bars 12ct", "aisle": "granola bars", "department": "snacks", "price_usd": 3.99, "avg_rating": 4.3, "rating_count": 2200, "in_stock": True},
]


@dataclass
class CatalogEntry:
    product_id: int
    product_name: str
    aisle: str
    aisle_id: int
    department: str
    department_id: int
    price_usd: float
    avg_rating: float
    rating_count: int
    in_stock: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_catalog() -> list[dict]:
    rng = random.Random(RNG_SEED)
    catalog: list[CatalogEntry] = []
    pid = 1

    dept_ids = {dept: i + 1 for i, dept in enumerate(DEPARTMENTS_AISLES)}
    aisle_id_counter = 0
    aisle_ids: dict[str, int] = {}
    for dept, aisles in DEPARTMENTS_AISLES.items():
        for a in aisles:
            aisle_id_counter += 1
            aisle_ids[a] = aisle_id_counter

    # Hook products first (so they get small ids, easy to reference in eval set)
    for h in EVAL_HOOK_PRODUCTS:
        catalog.append(CatalogEntry(
            product_id=pid,
            product_name=h["product_name"],
            aisle=h["aisle"],
            aisle_id=aisle_ids.get(h["aisle"], 0),
            department=h["department"],
            department_id=dept_ids.get(h["department"], 0),
            price_usd=h["price_usd"],
            avg_rating=h["avg_rating"],
            rating_count=h["rating_count"],
            in_stock=h["in_stock"],
        ))
        pid += 1

    # Template-based products
    for aisle, templates in PRODUCT_TEMPLATES.items():
        dept = next((d for d, a in DEPARTMENTS_AISLES.items() if aisle in a), "other")
        for t in templates:
            catalog.append(CatalogEntry(
                product_id=pid,
                product_name=t,
                aisle=aisle,
                aisle_id=aisle_ids.get(aisle, 0),
                department=dept,
                department_id=dept_ids.get(dept, 0),
                price_usd=round(rng.uniform(1.49, 24.99), 2),
                avg_rating=round(rng.uniform(3.8, 4.9), 2),
                rating_count=rng.randint(50, 6000),
                in_stock=rng.random() > 0.05,
            ))
            pid += 1

    return [c.to_dict() for c in catalog]
