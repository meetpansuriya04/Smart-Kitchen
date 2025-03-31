from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from collections import defaultdict
from datetime import datetime, timedelta
import re
import json
import random

app = Flask(__name__)
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

historical_sales = [
    {"date": "2023-01-26", "event": "Republic Day", "sales": 45, "profit": 2200},
    {"date": "2023-08-15", "event": "Independence Day", "sales": 52, "profit": 2550},
    {"date": "2023-10-24", "event": "Diwali", "sales": 68, "profit": 3300},
    {"date": "2023-12-25", "event": "Christmas", "sales": 38, "profit": 1850},
    # Add more festival dates and typical sales patterns
]

# Inventory and recipe data
inventory = defaultdict(dict)
recipes = {
    "Bombay Vadapav (Oil)": {
        "ingredients": {
            "pav": 2, 
            "potato": 1,
            "besan": 0.3,
            "oil": 0.2,
        
        },
        "profit_margin": 35
    },
    "Jain Vadapav (Oil)": {
        "ingredients": {
            "pav": 2,
            "raw_banana": 1,
            "besan": 0.3,
            "oil": 0.2,
        
        },
        "profit_margin": 40
    },
    "Dabeli (Butter)": {
        "ingredients": {
            "pav": 2,
            "potato": 1,
            "butter": 0.3,
            "pomegranate": 0.1
        },
        "profit_margin": 45
    },
    "Cheese Muskabun": {
        "ingredients": {
            "maida": 0.5,
            "cheese": 0.4,
            "yeast": 0.1,
            "milk": 0.2
        },
        "profit_margin": 50
    },
    "Cheese Bhel": {
        "ingredients": {
            "puffed_rice": 0.5,
            "sev": 0.2,
            "cheese": 0.3,
            "chaat_masala": 0.1
        },
        "profit_margin": 30
    },
    "Garlic Cheese Baked Roll": {
        "ingredients": {
            "tortilla": 1,
            "cheese": 0.4,
            "garlic_butter": 0.3,
            "cabbage": 0.2
        },
        "profit_margin": 55
    }
}

item_prices = {
    "pav": 2.0, "potato": 1.5, "besan": 4.0, "oil": 2.5,
    "chutney": 3.0, "raw_banana": 2.0, "jain_chutney": 3.5,
    "dabeli_masala": 5.0, "butter": 4.5, "pomegranate": 3.0,
    "maida": 2.0, "cheese": 6.0, "yeast": 1.0, "milk": 0.5,
    "puffed_rice": 1.2, "sev": 2.0, "chaat_masala": 4.0,
    "tortilla": 3.0, "garlic_butter": 5.0, "cabbage": 1.0
}

sales_history = []
def normalize_item_name(name):
    # Enhanced normalization map
    conversion_map = {
        # Brand handling
        'rajdhani maida': 'maida',
        'ashirvaad atta': 'atta',
        'amul cheese': 'cheese',
        
        # Physical forms
        'glass of milk': 'milk',
        'milk bottle': 'milk',
        'milk packet': 'milk',
        'cheese slice': 'cheese',
        'cheese block': 'cheese',
        'grated cheese': 'cheese',
        
        # Common alternatives
        'wheat flour': 'maida',
        'plain flour': 'maida',
        'bread rolls': 'pav',
        'burger buns': 'pav',
        
        # Plural forms
        'potatoes': 'potato',
        'tomatoes': 'tomato',
        'chillies': 'chilli'
    }

    # Clean the input name
    name = re.sub(r'[^a-z\s]', '', name.lower().strip())
    
    # Check direct matches first
    if name in conversion_map:
        return conversion_map[name]
    
    # Check partial matches
    for key in conversion_map:
        if key in name:
            return conversion_map[key]
    
    # Remove brand names and adjectives
    base_names = ['milk', 'cheese', 'maida', 'pav', 'potato']  # Add all inventory items
    for base in base_names:
        if base in name:
            return base
    
    # Final cleanup
    return re.sub(r's$', '', name).strip()

def predict_high_demand_dates():
    try:
        # Analyze historical patterns
        response = model.generate_content(f"""
        Analyze sales patterns and predict high-demand periods based on:
        {historical_sales}
        {sales_history}

        Consider:
        1. Annual festival patterns
        2. Weekly sales trends
        3. Recent growth rates
        4. Seasonal ingredient availability

        Respond in JSON format:
        {{
            "high_demand_dates": [
                {{
                    "date": "YYYY-MM-DD", 
                    "reason": "Festival/Pattern",
                    "expected_sales": "X-Y% increase",
                    "recommended_stock": "Inventory items to focus"
                }}
            ]
        }}
        """)
        return json.loads(response.text.replace('```json', '').replace('```', ''))
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/demand-predictions')
def get_demand_predictions():
    return jsonify(predict_high_demand_dates())

def calculate_spoilage(time_str):
    time_str = time_str.lower()
    numbers = re.findall(r'\d+', time_str)
    num = int(numbers[0]) if numbers else 7
    
    if 'hour' in time_str: return datetime.now() + timedelta(hours=num)
    if 'day' in time_str: return datetime.now() + timedelta(days=num)
    return datetime.now() + timedelta(days=7)

def get_expiry_status(spoilage_date):
    days_remaining = (spoilage_date - datetime.now()).days
    if days_remaining < 0: return "spoiled"
    if days_remaining <= 2: return "expiring"
    if days_remaining <= 5: return "near-expiry"
    return "fresh"

def get_priority_recipes():
    priority_list = []
    for name, details in recipes.items():
        recipe_possible = True
        cost = 0
        spoilage_priority = 0
        
        for ing, qty in details['ingredients'].items():
            inv_item = inventory.get(ing)
            if not inv_item or inv_item['quantity'] < qty:
                recipe_possible = False
                break
                
            days_left = (inv_item['spoilage_date'] - datetime.now()).days
            if days_left < 2: spoilage_priority += 3
            elif days_left < 5: spoilage_priority += 2
                
            cost += inv_item['price'] * qty
        
        if recipe_possible:
            priority_list.append({
                "name": name,
                "profit": details['profit_margin'] - cost,
                "spoilage_priority": spoilage_priority,
                "cost": cost
            })
    
    return sorted(priority_list, key=lambda x: (-x['spoilage_priority'], -x['profit']))

def calculate_waste_stats():
    at_risk = 0
    potential_loss = 0.0
    
    for item, details in inventory.items():
        if details['spoilage_date'] < datetime.now():
            potential_loss += details['price'] * details['quantity']
        elif (details['spoilage_date'] - datetime.now()).days < 3:
            at_risk += 1
            potential_loss += details['price'] * details['quantity'] * 0.5
    
    return {
        "items_at_risk": at_risk,
        "potential_savings": potential_loss * 0.7
    }

def get_expiry_alerts():
    alerts = []
    for name, details in inventory.items():
        status = get_expiry_status(details['spoilage_date'])
        if status != "fresh":
            alerts.append({
                "name": name,
                "status": status,
                "days_left": (details['spoilage_date'] - datetime.now()).days,
                "quantity": details['quantity']
            })
    return sorted(alerts, key=lambda x: x['days_left'])

def generate_stock_suggestions():
    try:
        if not sales_history:
            return {
                "items": [],
                "waste_reduction_tips": ["Process 5-10 sales to enable AI-powered suggestions"]
            }

        sales_data = "\n".join([
            f"{sale['timestamp']} - Sold {sale['recipe']} (Profit: ₹{sale['profit']:.2f})"
            for sale in sales_history[-50:]
        ])
        
        current_stock = "\n".join([
            f"{item}: {details['quantity']} units (expires {details['spoilage_date'].strftime('%Y-%m-%d')})"
            for item, details in inventory.items()
        ])

        response = model.generate_content(f"""
        Analyze kitchen inventory and sales to create:
        1. Stock levels sorted by expiration urgency
        2. Specific waste reduction actions with exact numbers

        Requirements:
        - For items expiring within 2 days: HIGH priority
        - Mention exact quantities and matching recipes
        - Calculate possible servings from expiring stock

        Sales Data:
        {sales_data}

        Current Stock:
        {current_stock}

        Required JSON format:
        {{
            "items": [
                {{
                    "name": "item_name",
                    "current_stock": number,
                    "suggested_stock": number,
                    "reorder_urgency": "high/medium/low",
                    "reason": "Specific plan with numbers and recipes"
                }}
            ],
            "waste_reduction_tips": [
                "Use X [ITEM] expiring [DATE] in [RECIPE] (Y needed/serving)",
                "Convert Z [ITEM] to [RECIPE] today"
            ]
        }}
        """)

        cleaned = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"error": f"AI analysis failed: {str(e)}"}

@app.route('/stacked-sales')
def get_stacked_sales():
    try:
        # Aggregate sales by date and recipe
        sales_data = defaultdict(lambda: defaultdict(float))
        
        for sale in sales_history + historical_sales:
            date = sale.get('date') or sale['timestamp'].split('T')[0]
            recipe = sale['recipe']
            sales_data[date][recipe] += sale['profit']

        # Convert to frontend format
        dates = sorted(sales_data.keys())
        recipes = list({recipe for day in sales_data.values() for recipe in day.keys()})
        
        return jsonify({
            "dates": dates,
            "recipes": recipes,
            "data": sales_data
        })
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify(error="No image uploaded"), 400
    
    try:
        img_file = request.files['image']
        if img_file.filename == '': return jsonify(error="No image selected"), 400

        img = Image.open(img_file.stream)
        img.verify()
        img_file.stream.seek(0)
        image_bytes = img_file.read()
        
        # Remove format restriction
        mime_type = f'image/{img_file.filename.split(".")[-1].lower()}'
        
        # Original code continues without format check
        response = model.generate_content([{
            'role': 'user',
            'parts': [
                "Analyze this kitchen inventory image. List items with quantities in format '- item (quantity)'",
                {'mime_type': mime_type, 'data': image_bytes}
            ]
        }])

        items = []
        for line in response.text.split('\n'):
            if line.startswith('- '):
                parts = line[2:].split('(')
                if len(parts) == 2:
                    raw_name = normalize_item_name(parts[0].strip().lower())
                    qty_part = parts[1].split(')')[0].strip().lower()
                    qty = 3 if qty_part in ['uncountable', 'several'] else \
                          int(re.sub(r'\D', '', qty_part or '1'))
                    
                    items.append({
                        "name": raw_name,
                        "quantity": qty,
                        "fridge_time": "5 days",
                        "room_temp_time": "2 days"
                    })

        for item in items:
            name = item['name']
            spoilage_date = calculate_spoilage(item['fridge_time'])
            
            if name in inventory:
                inventory[name]['quantity'] += item['quantity']
                if spoilage_date < inventory[name]['spoilage_date']:
                    inventory[name]['spoilage_date'] = spoilage_date
            else:
                inventory[name] = {
                    "quantity": item['quantity'],
                    "spoilage_date": spoilage_date,
                    "price": item_prices.get(name, 0)
                }

        return jsonify({
            "items": items,
            "inventory": inventory,
            "recipes": get_priority_recipes(),
            "waste_stats": calculate_waste_stats(),
            "expiry_alerts": get_expiry_alerts()
        })

    except Exception as e:
        return jsonify(error=f"Processing error: {str(e)}"), 500

@app.route('/sell-recipe', methods=['POST'])
def sell_recipe():
    recipe_name = request.json['recipe']
    if (recipe := recipes.get(recipe_name)) is None:
        return jsonify(error="Recipe not found"), 404

    if any(inventory.get(ing, {}).get('quantity', 0) < qty for ing, qty in recipe['ingredients'].items()):
        return jsonify(error="Insufficient ingredients"), 400

    for ing, qty in recipe['ingredients'].items():
        inventory[ing]['quantity'] = max(0, inventory[ing]['quantity'] - qty)

    cost = sum(item_prices.get(ing, 0) * qty for ing, qty in recipe['ingredients'].items())
    sales_history.append({
        "recipe": recipe_name,
        "timestamp": datetime.now().isoformat(),
        "profit": recipe['profit_margin'] - cost
    })

    # Calculate updated stats after sale
    waste_stats = calculate_waste_stats()
    priority_recipes = get_priority_recipes()
    expiry_alerts = get_expiry_alerts()

    return jsonify({
        "message": f"Sold {recipe_name}",
        "inventory": inventory,
        "recipes": priority_recipes,
        "waste_stats": waste_stats,
        "expiry_alerts": expiry_alerts
    })

@app.route('/stock-suggestions', methods=['GET'])
def get_stock_suggestions():
    try:
        suggestions = generate_stock_suggestions()
        if 'error' in suggestions:
            return jsonify(suggestions), 500
        return jsonify(suggestions)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/sales-report', methods=['GET'])
def get_sales_report():
    try:
        # Group sales by recipe
        recipe_summary = defaultdict(lambda: {
            'quantity': 0,
            'total_profit': 0,
            'total_revenue': 0,
            'total_cost': 0,  # Add this line
            'average_profit': 0,
            'cost_per_unit': 0  # Add this line
        })
        
        # Include both historical and current sales
        all_sales = sales_history + historical_sales
        
        for sale in all_sales:
            recipe = sale.get('recipe', '')
            profit = sale.get('profit', 0)
            
            if recipe:
                recipe_data = recipes.get(recipe, {})
                cost = sum(item_prices.get(ing, 0) * qty 
                          for ing, qty in recipe_data.get('ingredients', {}).items())
                revenue = profit + cost
                
                recipe_summary[recipe]['quantity'] += 1
                recipe_summary[recipe]['total_profit'] += profit
                recipe_summary[recipe]['total_revenue'] += revenue
                recipe_summary[recipe]['total_cost'] += cost  # Add this line
        
        # Calculate averages and prepare final report
        total_profit = 0
        total_revenue = 0
        total_cost = 0  # Add this line
        total_sales = 0
        report_items = []
        
        for recipe, stats in recipe_summary.items():
            if stats['quantity'] > 0:
                stats['average_profit'] = stats['total_profit'] / stats['quantity']
                stats['cost_per_unit'] = stats['total_cost'] / stats['quantity']  # Add this line
                total_profit += stats['total_profit']
                total_revenue += stats['total_revenue']
                total_cost += stats['total_cost']  # Add this line
                total_sales += stats['quantity']
                
                report_items.append({
                    'recipe': recipe,
                    'quantity': stats['quantity'],
                    'total_revenue': round(stats['total_revenue'], 2),
                    'total_profit': round(stats['total_profit'], 2),
                    'total_cost': round(stats['total_cost'], 2),  # Add this line
                    'cost_per_unit': round(stats['cost_per_unit'], 2),  # Add this line
                    'average_profit': round(stats['average_profit'], 2)
                })
        
        # Sort by total profit descending
        report_items.sort(key=lambda x: x['total_profit'], reverse=True)
        
        return jsonify({
            'items': report_items,
            'summary': {
                'total_sales': total_sales,
                'total_revenue': round(total_revenue, 2),
                'total_profit': round(total_profit, 2),
                'total_cost': round(total_cost, 2),  # Add this line
                'average_cost': round(total_cost / total_sales, 2) if total_sales > 0 else 0,  # Add this line
                'average_order_value': round(total_revenue / total_sales, 2) if total_sales > 0 else 0,
                'profit_margin': round((total_profit / total_revenue * 100), 2) if total_revenue > 0 else 0
            }
        })
        
    except Exception as e:
        return jsonify(error=f"Failed to generate sales report: {str(e)}"), 500

@app.route('/demand-prediction', methods=['GET'])
def get_demand_prediction():
    try:
        # Generate 12 months of historical data
        months = []
        sales_data = {}
        
        for recipe in recipes.keys():
            sales_data[recipe] = []
            
        current_date = datetime.now()
        for i in range(12):
            date = (current_date - timedelta(days=30*i)).strftime('%Y-%m')
            months.append(date)
            
            for recipe in recipes.keys():
                # Generate realistic seasonal variations
                base_demand = random.randint(30, 50)
                seasonal_factor = 1.0
                
                # Summer months (March-June)
                if date.endswith(('03', '04', '05', '06')):
                    seasonal_factor = 1.5 if 'Ice' in recipe else 0.8
                
                # Winter months (November-February)
                elif date.endswith(('11', '12', '01', '02')):
                    seasonal_factor = 0.7 if 'Ice' in recipe else 1.3
                
                # Festival months
                if date.endswith(('10', '11')):  # Diwali season
                    seasonal_factor *= 1.4
                
                demand = int(base_demand * seasonal_factor)
                sales_data[recipe].append(demand)
        
        # Calculate predictions for next month
        predictions = {}
        for recipe in recipes.keys():
            avg_demand = sum(sales_data[recipe][-3:]) / 3  # Last 3 months average
            growth_rate = 1.1  # 10% growth assumption
            predictions[recipe] = int(avg_demand * growth_rate)
        
        return jsonify({
            'historical': {
                'months': months[::-1],  # Reverse to show oldest first
                'data': sales_data
            },
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify(error=f"Failed to generate demand prediction: {str(e)}"), 500

@app.route('/loss-profit-stats', methods=['GET'])
def get_loss_profit_stats():
    try:
        # Get the number of months from query parameter, default to 6
        months = int(request.args.get('months', 6))
        
        # Generate sample data for demonstration
        current_date = datetime.now()
        monthly_trends = []
        
        for i in range(months):
            month_date = current_date - timedelta(days=30*i)
            month_str = month_date.strftime('%Y-%m')
            monthly_trends.append({
                "month": month_str,
                "waste_cost": random.uniform(500, 2000)  # Sample data
            })

        # Sort monthly trends chronologically
        monthly_trends.sort(key=lambda x: x['month'])

        # Generate top waste items
        waste_items = [
            {"item": "Cheese", "cost": random.uniform(200, 400)},
            {"item": "Bread", "cost": random.uniform(150, 300)},
            {"item": "Vegetables", "cost": random.uniform(100, 250)},
            {"item": "Sauces", "cost": random.uniform(50, 150)},
            {"item": "Spices", "cost": random.uniform(25, 75)}
        ]
        
        # Sort waste items by cost
        waste_items.sort(key=lambda x: x['cost'], reverse=True)

        # Calculate totals
        total_waste = sum(item['waste_cost'] for item in monthly_trends)
        potential_savings = total_waste * 0.4  # Example: 40% could be saved
        prevented_loss = total_waste * 0.25    # Example: 25% was prevented

        return jsonify({
            "total_waste_cost": total_waste,
            "potential_savings": potential_savings,
            "prevented_loss": prevented_loss,
            "monthly_trends": monthly_trends,
            "top_waste_items": waste_items
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this endpoint at the bottom of app.py
@app.route('/waste-heatmap', methods=['GET'])
def get_waste_heatmap():
    try:
        # Example waste data by station (you can adjust this dynamically)
        waste_data = {
            "Prep Station": random.uniform(200, 600),
            "Cooking": random.uniform(400, 1000),
            "Storage": random.uniform(150, 450),
            "Packaging": random.uniform(100, 300),
            "Cleaning": random.uniform(50, 200)
        }

        # Convert to format suitable for heatmap
        heatmap_data = [{"station": k, "waste": v} for k, v in waste_data.items()]

        return jsonify(heatmap_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/inventory/add', methods=['POST'])
def add_inventory_item():
    try:
        data = request.json
        name = normalize_item_name(data['name'])
        quantity = float(data['quantity'])
        spoilage_date = datetime.fromisoformat(data['spoilage_date'])
        price = float(data.get('price', item_prices.get(name, 0)))

        inventory[name] = {
            "quantity": quantity,
            "spoilage_date": spoilage_date,
            "price": price
        }

        return jsonify({
            "message": f"Added {name} to inventory",
            "inventory": inventory,
            "waste_stats": calculate_waste_stats(),
            "expiry_alerts": get_expiry_alerts()
        })
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.route('/inventory/update', methods=['PUT'])
def update_inventory_item():
    try:
        data = request.json
        old_name = normalize_item_name(data['name'])
        new_name = normalize_item_name(data.get('new_name', old_name))
        
        if old_name not in inventory:
            return jsonify(error="Item not found"), 404

        # If name is being changed
        if new_name != old_name:
            # Create new entry with updated name
            inventory[new_name] = inventory[old_name]
            # Delete old entry
            del inventory[old_name]
        
        # Update other fields
        if 'quantity' in data:
            inventory[new_name]['quantity'] = float(data['quantity'])
        if 'spoilage_date' in data:
            inventory[new_name]['spoilage_date'] = datetime.fromisoformat(data['spoilage_date'])
        if 'price' in data:
            inventory[new_name]['price'] = float(data['price'])

        return jsonify({
            "message": f"Updated {old_name} to {new_name} in inventory",
            "inventory": inventory,
            "waste_stats": calculate_waste_stats(),
            "expiry_alerts": get_expiry_alerts()
        })
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.route('/inventory/delete', methods=['DELETE'])
def delete_inventory_item():
    try:
        data = request.json
        name = normalize_item_name(data['name'])
        
        if name not in inventory:
            return jsonify(error="Item not found"), 404

        del inventory[name]

        return jsonify({
            "message": f"Deleted {name} from inventory",
            "inventory": inventory,
            "waste_stats": calculate_waste_stats(),
            "expiry_alerts": get_expiry_alerts()
        })
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)
