from flask import Flask,render_template,request,redirect,url_for
# from sklearn.neighbors import NearpestNeighbors 
import pandas as pd
import numpy as np
from generate_meal import generate_meal_plan,find_similar_product,find_meal_by_nutritional_need
app = Flask(__name__)



@app.route('/')
def home():
        return render_template('about.html')

@app.route('/generate')
def generate():
        return render_template("generate_meal.html")
# ====================================
# to generate meal 
# ===================================
@app.route('/generate_meal' , methods=['POST'])
def generate_meal():
        # option_of_generate = int(request.form['option_of_generate'])
        # dont forget the height validataion
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = int(request.form['height'])
        height = height / 100
        weight = int(request.form['weight'])
        activity = int(request.form['activity'])
        target = int(request.form['target']) 
        Number_of_Meals = int(request.form['Number_of_Meals']) 
        # Food_Preferences = request.form['Food_Preferences'] 
        sensitive = request.form['sensitive'] 
        # ----------------
        # print(option_of_generate , type(option_of_generate))
        print(gender , type(gender))
        print(age , type(age))
        print(height , type(height))
        print(weight , type(weight))
        print(activity , type(activity))
        print(target , type(target))
        print(Number_of_Meals , type(Number_of_Meals))
        # print(Food_Preferences , type(Food_Preferences))
        print(sensitive , type(sensitive))
        # ----------------------
        
        if Number_of_Meals == 2:
                meal_plan, total_macros_per_meal, total_macros = generate_meal_plan(age, gender, height, weight, activity, target, Number_of_Meals, sensitive)
                print(meal_plan[0][0]['name'])
                return render_template('meal_2.html',meal_plan = meal_plan,total_macros_per_meal=total_macros_per_meal,total_macros=total_macros)
        elif Number_of_Meals == 3:
                meal_plan, total_macros_per_meal, total_macros = generate_meal_plan(age, gender, height, weight, activity, target, Number_of_Meals, sensitive)
                return render_template('meal_3.html',meal_plan = meal_plan,total_macros_per_meal=total_macros_per_meal,total_macros=total_macros)
        elif Number_of_Meals == 4:
                meal_plan, total_macros_per_meal, total_macros = generate_meal_plan(age, gender, height, weight, activity, target, Number_of_Meals, sensitive)
                return render_template('meal_4.html',meal_plan = meal_plan,total_macros_per_meal=total_macros_per_meal,total_macros=total_macros)
        return render_template('index.html')

# ====================
# Find Similar Items
# ==================
@app.route('/Find_Similar_Items',methods = ['GET', 'POST'])
def Find_Similar_Items():
        if request.method == 'GET':
                return render_template('index.html')
        else:
                similar_food = request.form['similar_food']
                # print(similar_food['product_name'])
                similar_foodd = find_similar_product(similar_food)
                # print(similar_foodd['similar_products'][0]['name'])
                if not similar_foodd:
                        print('kiro')
                        return render_template('index.html',ms="the product not found")
                else:
                        return render_template('similar_food.html',similar_foodd=similar_foodd)
        
# ====================
# Choose a nutritional
# ==================
@app.route('/Choose_nutritional',methods = ['GET', 'POST'])
def Choose_nutritional():
        if request.method == 'GET':
                return render_template('indix.html',t=1)
        else:
                options = request.form['options']
                selected_meals=find_meal_by_nutritional_need(options)
                # print(selected_meals[0]['Estimated_Weight'])
                return render_template('Choose_nutritionall.html',selected_meals=selected_meals)




# @app.route('/generatee',methods=['POST'])
# def generatee():
# # Load the dataset (adjust the file path as needed)
#         df = pd.read_csv("C:/Users/Kiro/OneDrive - Arab Academy for Science and Technology/Desktop/AI_project/nutrition.csv")

# # Keep only relevant columns
#         df = df[['name', 'calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']]

# # Remove any rows with missing values
#         df = df.dropna()


# # Function to clean and normalize nutrient values
#         def normalize_value(value):
#                 value = str(value).lower().strip()
#                 if 'g' in value:
#                         return float(value.replace('g', '').strip())
#                 elif 'mg' in value:
#                         return float(value.replace('mg', '').strip()) / 1000  # Convert mg to g
#                 elif 'mcg' in value:
#                         return float(value.replace('mcg', '').strip()) / 1_000_000  # Convert mcg to g
#                 else:
#                         return float(value)  # Assume the value is already in grams


# # Normalize numeric columns
#         df['calories'] = pd.to_numeric(df['calories'], errors='coerce').fillna(0)
#         df['protein'] = df['protein'].apply(normalize_value)
#         df['carbohydrate'] = df['carbohydrate'].apply(normalize_value)
#         df['total_fat'] = df['total_fat'].apply(normalize_value)
#         df['fiber'] = df['fiber'].apply(normalize_value)
#         df['sugars'] = df['sugars'].apply(normalize_value)
#         df['saturated_fat'] = df['saturated_fat'].apply(normalize_value)


# # Function to find similar products using ML
#         # def find_similar_product(product_name, n=5):
#         # # Assuming df is your DataFrame and it contains the necessary columns
#         # # Check if the product exists in the dataset
#         #    product_row = df[df['name'].str.lower() == product_name.lower().strip()]
#         #    if product_row.empty:
#         #              print("Product not found in the dataset.")
#         #              return

#         #    # Extract nutrient columns for comparison
#         #    nutrient_columns = ['calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']
#         #    X = df[nutrient_columns]

#         #    # Initialize the Nearest Neighbors model with the feature names
#         #    knn = NearpestNeighbors(n_neighbors=n + 1, metric='euclidean')  # Include the product itself
#         #    knn.fit(X)

#         #    # Find neighbors for the given product
#         #    product_index = product_row.index[0]
#         #    distances, indices = knn.kneighbors([X.iloc[product_index]])

#         #    # Exclude the product itself and return similar products
#         #    similar_indices = indices[0][1:]
#         #    similar_products = df.iloc[similar_indices]
#         #    print(f"Similar products to '{product_name}':")
#         #    print(similar_products[
#         #                        ['name', 'calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']])

# # Function to validate user input
#         def validate_input(prompt, min_val, max_val, input_type=float):
#                 while True:
#                         try:
#                                 value = input_type(input(prompt))
#                                 if min_val <= value <= max_val:
#                                                 return value
#                                 else:
#                                                 print(f"Value must be between {min_val} and {max_val}.")
#                         except ValueError:
#                                         print(f"Invalid input. Please enter a {input_type._name_}.")


# # Function to calculate BMI
#         def calculate_bmi(weight, height):
#                 return weight / (height ** 2)


# # Function to calculate caloric needs using Harris-Benedict equation
#         def calculate_caloric_needs(age, weight, height, gender, activity_level):
#                 if gender == 1:  # Male
#                                 bmr = 88.362 + (13.397 * weight) + (4.799 * height * 100) - (5.677 * age)
#                 else:  # Female
#                                 bmr = 447.593 + (9.247 * weight) + (3.098 * height * 100) - (4.330 * age)

#                 activity_factors = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725}
#                 return bmr * activity_factors.get(activity_level, 1.2)


# # Function to calculate target caloric needs
#         def calculate_target_calories(caloric_needs, target):
#                 factors = {1: 1.3, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.7}
#                 return caloric_needs * factors.get(target, 1.0)


# # Function to calculate macronutrient targets
#         def calculate_macronutrient_targets(target_calories):
#                 protein_target = target_calories * 0.25 / 4  # 25% of calories from protein (4 kcal/g)
#                 carb_target = target_calories * 0.50 / 4  # 50% of calories from carbs (4 kcal/g)
#                 fat_target = target_calories * 0.25 / 9  # 25% of calories from fat (9 kcal/g)
#                 return protein_target, carb_target, fat_target


# # Function to enforce realistic nutrient ranges
#         def is_within_bounds(calories, protein, carbs, fats, fiber, sugars, saturated_fat,
#                                         target_calories, protein_target, carb_target, fat_target):
#                 return (
#                         0.9 * target_calories <= calories <= 1.1 * target_calories and
#                         0.9 * protein_target <= protein <= 1.1 * protein_target and
#                         0.9 * carb_target <= carbs <= 1.1 * carb_target and
#                         0.9 * fat_target <= fats <= 1.1 * fat_target and
#                         fiber >= 5 and  # Minimum fiber per meal (adjust as needed)
#                         sugars <= 25 and  # Maximum sugars per meal (adjust as needed)
#                         saturated_fat <= 10  # Maximum saturated fat per meal (adjust as needed)
#         )

#         bmi = 1
#         caloric_needs = 1
#         target_calories = 1
#         calories_per_meal = 1
#         protein_per_meal = 1
# # Main function
#         def main():
#                 print("Choose an option:")
#                 print("1. Generate a meal plan")
#                 print("2. Find a similar product")
#                 # choice = validate_input("Enter your choice (1 or 2): ", 1, 2, int)
#                 choice = request.form['choice']
#                 choice = int(choice)

#                 if choice == 1:
#                                 # Collect user data for meal plan generation
#                                 # age = validate_input("Enter your age: ", 12, 70, int)
#                                 age = request.form['age']
#                                 age = int(age)
#                                 # gender = validate_input("Enter your gender (1 for Male, 2 for Female): ", 1, 2, int)
#                                 gender = request.form['gender']
#                                 gender = int(gender)
#                                 # height = validate_input("Enter your height in meters (e.g., 1.75): ", 1.0, 2.5, float)
#                                 height = request.form['height']
#                                 height = float(height)
#                                 # weight = validate_input("Enter your weight in kg: ", 30, 200, float)
#                                 weight = request.form['weight']
#                                 weight = float(weight)
#                                 # activity_level = validate_input(
#                                 #                "Enter your activity level (1 for Sedentary, 2 for Light, 3 for Moderate, 4 for Very Active): ", 1, 4, int)
#                                 activity_level = request.form['activity_level']
#                                 activity_level = int(activity_level)
#                                 # target = validate_input(
#                                 #                "Enter your target (1 for Hard Gain, 2 for Gain, 3 for Maintain, 4 for Loss, 5 for Hard Loss): ", 1, 5, int)
#                                 target = request.form['target']
#                                 target = int(target)
#                                 # num_meals = validate_input("Enter the number of meals per day (e.g., 3 or 5): ", 1, 6, int)
#                                 num_meals = request.form['num_meals']
#                                 num_meals = int(num_meals)
#                                 if num_meals >= 6:
#                                         num_meals = 6
#                 # food_preferences = input(
#                 #         "Enter your food preferences (comma-separated, or leave blank for no preferences): ").split(',')
#                                 food_preferences = request.form['food_preferences']
#                                 # print(food_preferences)
#                                 food_preferences = ['']
#                                 generate_meal_plan(age, gender, height, weight, activity_level, target, food_preferences, num_meals)

#                 elif choice == 2:
#                         product_name = input("Enter the product name to find similar items: ")
#                 # find_similar_product(product_name)
# # Function to generate a meal plan
#         def generate_meal_plan(age, gender, height, weight, activity_level, target, food_preferences, num_meals):
#         # Calculate BMI
#                 bmi = calculate_bmi(weight, height)

#         # Calculate caloric needs
#                 caloric_needs = calculate_caloric_needs(age, weight, height, gender, activity_level)

#         # Calculate target calories
#                 target_calories = calculate_target_calories(caloric_needs, target)
#                 calories_per_meal = target_calories / num_meals

#         # Calculate macronutrient targets
#                 protein_target, carb_target, fat_target = calculate_macronutrient_targets(target_calories)
#                 protein_per_meal = protein_target / num_meals
#                 carb_per_meal = carb_target / num_meals
#                 fat_per_meal = fat_target / num_meals

#                 print(f"\nYour BMI is: {bmi:.2f}")
#                 print(f"Your daily caloric needs are: {caloric_needs:.2f} kcal")
#                 print(f"Your target daily calories are: {target_calories:.2f} kcal")
#                 print(f"Each meal should provide approximately: {calories_per_meal:.2f} kcal")
#                 print(
#                 f"Macronutrient targets per meal: Protein: {protein_per_meal:.2f} g, Carbohydrates: {carb_per_meal:.2f} g, Fat: {fat_per_meal:.2f} g")

#         # If no preferences provided, use all foods
#                 if food_preferences == ['']:
#                         food_preferences = df['name'].str.lower().tolist()

#         # Filter food items by preferences
#                 preferred_food = df[
#                         df['name'].str.lower().str.strip().apply(lambda x: any(item.strip() in x for item in food_preferences))]

#                 if preferred_food.empty:
#                         print("No matching food items found. Using all available items.")
#                         preferred_food = df

#         # Pre-filter foods close to the macronutrient targets
#                 preferred_food = preferred_food[
#                 (preferred_food['calories'] <= 1.2 * calories_per_meal) &
#                 (preferred_food['protein'] <= 1.2 * protein_per_meal) &
#                 (preferred_food['carbohydrate'] <= 1.2 * carb_per_meal) &
#                 (preferred_food['total_fat'] <= 1.2 * fat_per_meal)
#                 ]

#                 if preferred_food.empty:
#                         print("No suitable foods found based on the criteria.")
#                         return

#         # Generate meals
#                 total_calories = 0
#                 total_protein = 0
#                 total_carbs = 0
#                 total_fats = 0

#                 for i in range(num_meals):
#                         while True:
#                                 sampled_food = preferred_food.sample(n=min(3, len(preferred_food)))
#                                 meal_calories = sampled_food['calories'].sum()
#                                 meal_protein = sampled_food['protein'].sum()
#                                 meal_carbs = sampled_food['carbohydrate'].sum()
#                                 meal_fats = sampled_food['total_fat'].sum()
#                                 meal_fiber = sampled_food['fiber'].sum()
#                                 meal_sugars = sampled_food['sugars'].sum()
#                                 meal_saturated_fat = sampled_food['saturated_fat'].sum()

#                                 if is_within_bounds(
#                                         meal_calories, meal_protein, meal_carbs, meal_fats,
#                                         meal_fiber, meal_sugars, meal_saturated_fat,
#                                         calories_per_meal, protein_per_meal, carb_per_meal, fat_per_meal
#                                 ):
#                                         break

#                 total_calories += meal_calories
#                 total_protein += meal_protein
#                 total_carbs += meal_carbs
#                 total_fats += meal_fats

#                 print(f"\nMeal {i + 1}:")
#                 print(sampled_food[
#                                 ['name', 'calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']])

#         # Display daily totals
#                 print("\nTotal for the day:")
#                 print(f"Calories: {total_calories:.2f} kcal")
#                 print(f"Protein: {total_protein:.2f} g")
#                 print(f"Carbohydrates: {total_carbs:.2f} g")
#                 print(f"Total Fat: {total_fats:.2f} g")
#                 print("kirols fawzy kamel ")



#         main()
#         print("kirolos fawyz ")
#         # return render_template('result.html',bmi=bmi,caloric_needs = caloric_needs ,
#         #                         target_calories =target_calories, calories_per_meal=calories_per_meal,
#         #                         protein_per_meal=protein_per_meal)
#         return render_template('/templates/hamada.html')
if __name__ == "__main__":
        app.run(debug=True)