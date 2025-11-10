import os  # Standard library: read environment variables
from datetime import datetime  # Timestamps for created/updated fields
from typing import List, Optional, Dict, Any  # Type hints for clarity

from dotenv import load_dotenv  # Load .env file into environment
from pymongo import MongoClient, ASCENDING, ReturnDocument  # MongoDB client and helpers


load_dotenv()  # Ensure env vars (like MONGODB_URI) are available


def _get_mongo_client() -> MongoClient:  # Create a MongoDB client from env
    uri = os.getenv("MONGODB_URI")  # Read the connection string
    if not uri:  # Validate presence
        raise ValueError("MONGODB_URI is not set in environment/.env")  # Fail fast if missing
    return MongoClient(uri)  # Return connected client (lazy until first op)


def _get_db():  # Get a handle to the target database
    client = _get_mongo_client()  # Create client
    db_name = os.getenv("DB_NAME", "ai_calorie")  # Default DB name if not provided
    return client[db_name]  # Return DB object


def _ensure_indexes(db) -> None:  # Create indexes if they don't exist
    db.users.create_index([("name", ASCENDING)], unique=True)  # Unique user by name
    db.meals.create_index([("date", ASCENDING)])  # Query meals quickly by date
    db.meals.create_index([("name", ASCENDING), ("date", ASCENDING)])  # Composite index
    db.daily_totals.create_index([("date", ASCENDING)], unique=True)  # One total per day


def init_db():  # Initialize DB and ensure indexes
    db = _get_db()  # Get DB
    _ensure_indexes(db)  # Ensure indices
    return db  # Return DB handle


def upsert_user(  # Insert or update a user profile
    name: str,
    age: int,
    gender: str,
    weight_kg: float,
    height_cm: float,
    tdee: float,
) -> Dict[str, Any]:
    db = init_db()  # DB handle
    doc = {  # Document fields
        "name": name,  # Person's name
        "age": age,  # Age in years
        "gender": gender,  # Gender as string
        "weight_kg": weight_kg,  # Weight in kilograms
        "height_cm": height_cm,  # Height in centimeters
        "tdee": tdee,  # Total Daily Energy Expenditure
        "updated_at": datetime.utcnow(),  # Audit timestamp
    }
    result = db.users.find_one_and_update(  # Upsert by name
        {"name": name},  # Match on unique name
        {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},  # Set/insert
        upsert=True,  # Create if missing
        return_document=ReturnDocument.AFTER,  # Return updated document
    )
    return result  # Return user doc


def get_user(name: str) -> Optional[Dict[str, Any]]:  # Fetch a user by name
    db = init_db()  # DB handle
    return db.users.find_one({"name": name})  # Return user or None


def _normalize_date_str(date: Optional[str] = None) -> str:  # Ensure YYYY-MM-DD string
    if date:  # If caller provided date
        return date  # Use it as-is
    return datetime.utcnow().strftime("%Y-%m-%d")  # Default to today (UTC)


def add_meal(  # Insert a meal and trigger totals recompute
    name: str,
    calories: float,
    ingredients: List[str],
    date: Optional[str] = None,
    protein_g: float = 0.0,  # Optional protein grams
    carbs_g: float = 0.0,  # Optional carbs grams
    fat_g: float = 0.0,  # Optional total fat grams
    healthy_fat_g: float = 0.0,  # Optional healthy fat grams
    unhealthy_fat_g: float = 0.0,  # Optional unhealthy fat grams
) -> Dict[str, Any]:
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    meal = {  # Construct meal document
        "name": name,  # Meal name
        "calories": calories,  # Energy content
        "ingredients": ingredients,  # Ingredient list
        "date": date_str,  # Logical meal date
        "protein_g": float(protein_g),  # Protein grams
        "carbs_g": float(carbs_g),  # Carbohydrate grams
        "fat_g": float(fat_g),  # Total fat grams
        "healthy_fat_g": float(healthy_fat_g),  # Healthy fat grams
        "unhealthy_fat_g": float(unhealthy_fat_g),  # Unhealthy fat grams
        "created_at": datetime.utcnow(),  # Audit timestamp
    }
    db.meals.insert_one(meal)  # Persist meal
    _recompute_daily_total(date_str)  # Update per-day total
    return meal  # Return inserted doc (local copy)


def list_meals_by_date(date: Optional[str] = None) -> List[Dict[str, Any]]:  # Query meals
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    return list(db.meals.find({"date": date_str}).sort("name", ASCENDING))  # Sorted list


def delete_meal_by_name(name: str, date: Optional[str] = None) -> int:  # Delete meals
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    res = db.meals.delete_many({"name": name, "date": date_str})  # Bulk delete
    _recompute_daily_total(date_str)  # Recompute total after deletion
    return res.deleted_count  # Return number deleted


def _recompute_daily_total(date: str) -> Dict[str, Any]:  # Recalculate daily totals
    db = init_db()  # DB handle
    meals = list(db.meals.find({"date": date}))  # Fetch meals for date
    
    # Filter out any incorrectly created "Daily Total" entries (these should not be meals)
    # Daily totals are automatically calculated, not stored as meal entries
    excluded_names = {"Daily Total", "Total", "Daily Total for", "Daily Total for "}
    valid_meals = [m for m in meals if m.get("name", "").strip() not in excluded_names and not m.get("name", "").startswith("Daily Total")]
    
    total = float(sum(m.get("calories", 0) or 0 for m in valid_meals))  # Sum calories
    total_protein = float(sum(m.get("protein_g", 0) or 0 for m in valid_meals))  # Sum protein
    total_carbs = float(sum(m.get("carbs_g", 0) or 0 for m in valid_meals))  # Sum carbs
    total_fat = float(sum(m.get("fat_g", 0) or 0 for m in valid_meals))  # Sum total fat
    total_healthy_fat = float(sum(m.get("healthy_fat_g", 0) or 0 for m in valid_meals))  # Sum healthy fat
    total_unhealthy_fat = float(sum(m.get("unhealthy_fat_g", 0) or 0 for m in valid_meals))  # Sum unhealthy fat
    meal_names = sorted({m.get("name", "") for m in valid_meals if m.get("name")})  # Unique names
    payload = {  # Document to upsert into daily_totals
        "date": date,
        "total_calories": total,
        "total_protein_g": total_protein,  # Aggregated protein for the date
        "total_carbs_g": total_carbs,  # Aggregated carbs for the date
        "total_fat_g": total_fat,  # Aggregated total fat for the date
        "total_healthy_fat_g": total_healthy_fat,  # Aggregated healthy fat
        "total_unhealthy_fat_g": total_unhealthy_fat,  # Aggregated unhealthy fat
        "meal_names": meal_names,
        "updated_at": datetime.utcnow(),
    }
    doc = db.daily_totals.find_one_and_update(  # Upsert totals
        {"date": date},
        {"$set": payload, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc  # Return updated totals document


def get_daily_total(date: Optional[str] = None) -> Optional[Dict[str, Any]]:  # Read daily totals
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    return db.daily_totals.find_one({"date": date_str})  # Return totals or None


def delete_daily_total(date: Optional[str] = None) -> int:  # Delete daily total for a date
    """Delete the daily total entry for a given date. Returns count deleted (0 or 1)."""
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    res = db.daily_totals.delete_many({"date": date_str})  # Delete daily total
    return res.deleted_count  # Return number deleted


def delete_all_meals_by_date(date: Optional[str] = None) -> int:  # Delete all meals for a date
    """Delete ALL meals for a given date. This will also trigger recomputation of daily totals (which will be 0)."""
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    res = db.meals.delete_many({"date": date_str})  # Delete all meals for date
    _recompute_daily_total(date_str)  # Recompute total after deletion (will be 0)
    return res.deleted_count  # Return number deleted


def update_meal(  # Update an existing meal
    name: str,
    date: Optional[str] = None,
    calories: Optional[float] = None,
    ingredients: Optional[List[str]] = None,
    protein_g: Optional[float] = None,
    carbs_g: Optional[float] = None,
    fat_g: Optional[float] = None,
    healthy_fat_g: Optional[float] = None,
    unhealthy_fat_g: Optional[float] = None,
) -> Optional[Dict[str, Any]]:  # Return updated meal or None if not found
    """Update an existing meal. Only provided fields will be updated. Returns updated meal or None if not found."""
    db = init_db()  # DB handle
    date_str = _normalize_date_str(date)  # Normalize date
    
    # Build update document with only provided fields
    update_doc = {"updated_at": datetime.utcnow()}
    if calories is not None:
        update_doc["calories"] = float(calories)
    if ingredients is not None:
        update_doc["ingredients"] = ingredients
    if protein_g is not None:
        update_doc["protein_g"] = float(protein_g)
    if carbs_g is not None:
        update_doc["carbs_g"] = float(carbs_g)
    if fat_g is not None:
        update_doc["fat_g"] = float(fat_g)
    if healthy_fat_g is not None:
        update_doc["healthy_fat_g"] = float(healthy_fat_g)
    if unhealthy_fat_g is not None:
        update_doc["unhealthy_fat_g"] = float(unhealthy_fat_g)
    
    # Update the meal
    result = db.meals.find_one_and_update(
        {"name": name, "date": date_str},
        {"$set": update_doc},
        return_document=ReturnDocument.AFTER,
    )
    
    if result:
        _recompute_daily_total(date_str)  # Recompute totals after update
    
    return result  # Return updated meal or None


def set_mustafa_profile(  # Convenience wrapper for your profile
    age: int,
    gender: str,
    weight_kg: float,
    height_cm: float,
    tdee: float,
) -> Dict[str, Any]:
    return upsert_user(  # Delegate to upsert_user with fixed name
        name="Mustafa Asghari",
        age=age,
        gender=gender,
        weight_kg=weight_kg,
        height_cm=height_cm,
        tdee=tdee,
    )

