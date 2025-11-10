import json  # For potential JSON handling (not strictly required by tools below)
import os  # Access environment variables if needed by the server
from typing import List, Optional, Union  # Type hints for tool parameters

from dotenv import load_dotenv  # Load .env values into environment
from mcp.server.fastmcp import FastMCP  # Minimal server helper for MCP over stdio

# Our data layer functions imported from quries.py
from quries import (
    set_mustafa_profile,  # Upsert Mustafa profile
    get_user,             # Fetch user document by name
    add_meal,             # Insert a meal
    list_meals_by_date,   # List meals by date
    delete_meal_by_name,  # Delete meals by name for a date
    get_daily_total,      # Get per-day total calories
    delete_daily_total,   # Delete daily total for a date
    delete_all_meals_by_date,  # Delete all meals for a date
    update_meal,          # Update an existing meal
)

load_dotenv()  # Load environment variables so quries.py can connect to MongoDB

app = FastMCP("ai_calorie_mongo")  # Initialize the MCP stdio server with a name


def _parse_ingredients(value: Union[str, List[str]]) -> List[str]:  # Normalize ingredients input
    if isinstance(value, list):  # If already a list
        return [str(v).strip() for v in value if str(v).strip()]  # Clean and filter empties
    if isinstance(value, str):  # If a single string
        # allow comma-separated
        return [v.strip() for v in value.split(",") if v.strip()]  # Split into a list
    return []  # Fallback to empty list


@app.tool()  # Expose as an MCP tool callable by the LLM
def set_profile(age: int, gender: str, weight_kg: float, height_cm: float, tdee: float):  # Tool signature
    """Upsert Mustafa Asghari profile and TDEE.
    Args:
      age: int
      gender: 'male'|'female'|string
      weight_kg: float
      height_cm: float
      tdee: float
    Returns: updated user document
    """
    doc = set_mustafa_profile(age=age, gender=gender, weight_kg=weight_kg, height_cm=height_cm, tdee=tdee)  # Write profile
    # ensure _id is serializable
    if doc and "_id" in doc:  # If Mongo object id present
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string for JSON
    return doc  # Return updated document


@app.tool()  # Expose as MCP tool
def get_profile(name: str = "Mustafa Asghari"):  # Default to your name
    """Fetch a user by name. Default is 'Mustafa Asghari'."""
    doc = get_user(name)  # Read from DB
    if doc and "_id" in doc:  # Normalize ObjectId
        doc["_id"] = str(doc["_id"])  # Convert to string
    return doc  # Return user document (or None)


@app.tool()  # Expose as MCP tool
def meal_add(  # Add meal tool with nutrition breakdown
    name: str,
    calories: float,
    ingredients: Union[str, List[str]],
    date: Optional[str] = None,
    protein_g: float = 0.0,
    carbs_g: float = 0.0,
    fat_g: float = 0.0,
    healthy_fat_g: float = 0.0,
    unhealthy_fat_g: float = 0.0,
):
    """Add a meal with calories, ingredients, and nutrition.
    
    IMPORTANT: Before adding a meal, always check existing meals for the date using meals_list() to avoid duplicates.
    DO NOT create entries named "Daily Total" or "Total" - these are automatically calculated in the daily_totals collection.
    
    REQUIRED FIELDS: All meals MUST include fat_g, healthy_fat_g, and unhealthy_fat_g values. Never omit fat information.
    
    Args:
      name: meal name (e.g., "Tuna Rice Bowl", "Protein Bar")
      calories: numeric calories
      ingredients: list or comma-separated string
      date: YYYY-MM-DD (optional, defaults to today UTC)
      protein_g: protein grams
      carbs_g: carbohydrate grams
      fat_g: REQUIRED - total fat grams (must equal healthy_fat_g + unhealthy_fat_g)
      healthy_fat_g: REQUIRED - healthy fat grams (e.g., from avocado, nuts, olive oil)
      unhealthy_fat_g: REQUIRED - unhealthy fat grams (e.g., saturated/trans fats)
    Returns: inserted meal document
    
    Note: Daily totals are automatically recalculated after adding a meal. All fat values are included in totals.
    """
    ing_list = _parse_ingredients(ingredients)  # Normalize ingredients to list[str]
    doc = add_meal(  # Insert meal with nutrition fields
        name=name,
        calories=calories,
        ingredients=ing_list,
        date=date,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        healthy_fat_g=healthy_fat_g,
        unhealthy_fat_g=unhealthy_fat_g,
    )
    return doc  # Return inserted meal


@app.tool()  # Expose as MCP tool
def meals_list(date: Optional[str] = None):  # List meals for date
    """List ALL meals for a given date (default: today, UTC).
    
    CRITICAL: Always call this BEFORE inserting, updating, or deleting meals to check what exists in the database.
    This helps avoid duplicates and ensures accurate calculations.
    
    Returns: List of all meal documents for the date, sorted by name.
    Each meal includes: name, calories, ingredients, protein_g, carbs_g, fat_g, healthy_fat_g, unhealthy_fat_g.
    All meals MUST have fat_g, healthy_fat_g, and unhealthy_fat_g values - verify these are present.
    """
    docs = list_meals_by_date(date)  # Query meals
    for d in docs:  # Normalize ObjectIds
        if "_id" in d:
            d["_id"] = str(d["_id"])  # Convert to string for JSON
    return docs  # Return list of meals


@app.tool()  # Expose as MCP tool
def meal_delete(name: str, date: Optional[str] = None):  # Delete meals by name/date
    """Delete meals by name for a given date (default: today).
    
    This deletes ALL meals with the given name for the specified date.
    Daily totals are automatically recalculated after deletion.
    
    Returns: dict with deleted_count (number of meals deleted).
    """
    deleted = delete_meal_by_name(name=name, date=date)  # Perform deletion
    return {"deleted_count": deleted}  # Return count


@app.tool()  # Expose as MCP tool
def total_get(date: Optional[str] = None):  # Get daily totals
    """Get daily total calories and nutrition for a date (default: today).
    
    Returns the automatically calculated totals from the daily_totals collection.
    Structure: date, total_calories, total_protein_g, total_carbs_g, total_fat_g, 
    total_healthy_fat_g, total_unhealthy_fat_g, meal_names (list).
    
    CRITICAL: Always call this to verify totals match expected calculations.
    When displaying results, ALWAYS show all three fat values: total_fat_g, total_healthy_fat_g, total_unhealthy_fat_g.
    If totals are wrong, check meals_list() to see if there are duplicate meals or incorrect entries.
    """
    doc = get_daily_total(date)  # Read total from DB
    if doc and "_id" in doc:  # Normalize ObjectId
        doc["_id"] = str(doc["_id"])  # Convert to string
    return doc  # Return totals document (or None)


@app.tool()  # Expose as MCP tool
def total_delete(date: Optional[str] = None):  # Delete daily total
    """Delete the daily total entry for a given date (default: today).
    
    FULL PERMISSION: You have permission to delete daily totals.
    This is useful when you need to clear incorrect totals before recalculating.
    
    Note: Daily totals are automatically recalculated when meals are added/deleted/updated.
    You may need to delete the total if it's incorrect due to duplicate meal entries.
    
    Returns: dict with deleted_count (0 or 1).
    """
    deleted = delete_daily_total(date)  # Delete daily total
    return {"deleted_count": deleted}  # Return count


@app.tool()  # Expose as MCP tool
def meals_delete_all(date: Optional[str] = None):  # Delete all meals for a date
    """Delete ALL meals for a given date (default: today).
    
    FULL PERMISSION: You have permission to delete all meals for a date.
    This is useful when you need to clear all data for a date and start fresh.
    
    WARNING: This will delete ALL meals for the date, not just one meal.
    Daily totals will be automatically recalculated (will be 0 after deletion).
    
    Returns: dict with deleted_count (number of meals deleted).
    """
    deleted = delete_all_meals_by_date(date)  # Delete all meals
    return {"deleted_count": deleted}  # Return count


@app.tool()  # Expose as MCP tool
def meal_update(  # Update meal tool
    name: str,
    date: Optional[str] = None,
    calories: Optional[float] = None,
    ingredients: Optional[Union[str, List[str]]] = None,
    protein_g: Optional[float] = None,
    carbs_g: Optional[float] = None,
    fat_g: Optional[float] = None,
    healthy_fat_g: Optional[float] = None,
    unhealthy_fat_g: Optional[float] = None,
):
    """Update an existing meal. Only provided fields will be updated.
    
    FULL PERMISSION: You have permission to update any meal.
    
    IMPORTANT: Always call meals_list() first to verify the meal exists before updating.
    Only fields you provide will be updated - other fields remain unchanged.
    
    REQUIRED: When updating fat information, always provide fat_g, healthy_fat_g, and unhealthy_fat_g together.
    Ensure fat_g equals healthy_fat_g + unhealthy_fat_g.
    
    Args:
      name: meal name (required - identifies which meal to update)
      date: YYYY-MM-DD (optional, defaults to today UTC)
      calories: numeric calories (optional)
      ingredients: list or comma-separated string (optional)
      protein_g: protein grams (optional)
      carbs_g: carbohydrate grams (optional)
      fat_g: REQUIRED when updating fats - total fat grams (must equal healthy_fat_g + unhealthy_fat_g)
      healthy_fat_g: REQUIRED when updating fats - healthy fat grams (e.g., from avocado, nuts, olive oil)
      unhealthy_fat_g: REQUIRED when updating fats - unhealthy fat grams (e.g., saturated/trans fats)
    
    Returns: updated meal document, or None if meal not found.
    
    Note: Daily totals are automatically recalculated after updating a meal. All fat values are included in totals.
    """
    ing_list = None
    if ingredients is not None:
        ing_list = _parse_ingredients(ingredients)  # Normalize ingredients
    
    doc = update_meal(  # Update meal
        name=name,
        date=date,
        calories=calories,
        ingredients=ing_list,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        healthy_fat_g=healthy_fat_g,
        unhealthy_fat_g=unhealthy_fat_g,
    )
    return doc  # Return updated meal or None


if __name__ == "__main__":  # If executed directly
    # Run as stdio MCP server
    app.run()  # Start serving tools over stdin/stdout
