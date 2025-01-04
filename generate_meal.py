from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import heapq
import random

# Load the dataset (adjust the file path as needed)
df = pd.read_csv('Cleaned_Data_Final.csv')

# Keep only relevant columns
df = df[['name','serving_size' ,'calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat', 'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids', 'cholesterol', 'lactose']]

all_diets_df = pd.read_csv('All_Diets.csv')

imputer = KNNImputer(n_neighbors=5)
df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

# Normalize data for KNN
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])
df_scaled['name'] = df['name']

# Function to check for allergies
def check_allergies(product, allergies):
    if 'nuts' in allergies:
        if (
            product['monounsaturated_fatty_acids'] > 0 and
            product['polyunsaturated_fatty_acids'] > 0 and
            product['protein'] > 0 and
            product['fiber'] > 0 and
            product['total_fat'] > 0 and
            product['cholesterol'] == 0
        ):
            return True

    if 'lactose' in allergies and product['lactose'] > 0:
        return True

    if 'soy' in allergies:
        soy_keywords = ["soy", "tofu", "soybean", "edamame"]
        if any(keyword in product['name'].lower() for keyword in soy_keywords):
            return True

    if 'gluten' in allergies:
        gluten_keywords = ["wheat", "barley", "rye", "oat", "gluten"]
        if any(keyword in product['name'].lower() for keyword in gluten_keywords):
            return True

    if 'vegan' in allergies:
        if product['cholesterol'] > 0:
            return True

    return False


# Function to filter out allergenic products
def filter_allergenic_products(df, allergies):
    return df[~df.apply(lambda x: check_allergies(x, allergies), axis=1)]

# KNN to find similar products using ML
def find_similar_product(product_name, n=5):
    product_name = product_name.strip().lower()  
    
    matching_products = df[df['name'].str.lower().str.contains(product_name, na=False)]
    
    if matching_products.empty:
        print("Product not found in the dataset.")
        return {}

    product_row = matching_products.iloc[0]
    product_index = product_row.name  # Index of the matched product

    nutrient_columns = ['calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']
    
    if not all(col in df_scaled.columns for col in nutrient_columns):
        print("Nutrient columns are missing in the scaled dataset.")
        return {}

    X = df_scaled[nutrient_columns]

    knn = NearestNeighbors(n_neighbors=n + 1, metric='euclidean')
    knn.fit(X.values)

    product_features = X.loc[product_index].values.reshape(1, -1)
    
    distances, indices = knn.kneighbors(product_features)

    similar_indices = indices[0][1:]
    similar_products = df.iloc[similar_indices]

    similar_products_list = similar_products[['name', 'calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']].to_dict('records')
    
    print(f"Similar products to '{product_name}':")
    print(similar_products_list)
    
    return {"product_name": product_name, "similar_products": similar_products_list}



def validate_input(prompt, min_val, max_val, input_type=float):
    while True:
        try:
            value = input_type(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print(f"Invalid input. Please enter a {input_type.__name__}.")

# Function to enforce realistic nutrient ranges
def is_within_bounds(calories, protein, carbs, fats, fiber, sugars, saturated_fat,
                     target_calories, protein_target, carb_target, fat_target, num_meals):
    # Ensure target values are distributed across meals
    meal_index = list(range(num_meals))  # Create indices for each meal

    for i in meal_index:
        target_calories_i = target_calories[i] if isinstance(target_calories, list) else target_calories
        protein_target_i = protein_target[i] if isinstance(protein_target, list) else protein_target
        carb_target_i = carb_target[i] if isinstance(carb_target, list) else carb_target
        fat_target_i = fat_target[i] if isinstance(fat_target, list) else fat_target

        if not (
            0.9 * target_calories_i <= calories <= 1.05 * target_calories_i and
            0.9 * protein_target_i <= protein <= 1.05 * protein_target_i and
            0.9 * carb_target_i <= carbs <= 1.05 * carb_target_i and
            0.9 * fat_target_i <= fats <= 1.05 * fat_target_i #and
            # fiber >= 5 and  # Minimum fiber per meal (adjust as needed)
            # sugars <= 25 and  # Maximum sugars per meal (adjust as needed)
            # saturated_fat <= 10  # Maximum saturated fat per meal (adjust as needed)
        ):
            return False

    return True

def calculate_portion_size_factor(scale_calories, scale_protein, scale_carbs, scale_fats, bmi, gender, num_meals):
    # Validate and normalize scale factors
    scale_factors = [scale_calories, scale_protein, scale_carbs, scale_fats]
    valid_factors = [factor for factor in scale_factors if factor > 0]

    if not valid_factors:
        return 1.0  # Default to 1.0 if no valid factors

    # Compute weighted scaling factor with more emphasis on calories and protein
    weighted_factor = (0.5 * scale_calories + 0.3 * scale_protein + 0.1 * scale_carbs + 0.1 * scale_fats)

    # Adjust for BMI: Higher BMI scales portions up
    bmi_adjustment = 1.3 if bmi >= 25 else (1.1 if 18.5 <= bmi < 25 else 1)

    # Adjust for gender: Males generally require more portions
    gender_adjustment = 1.2 if gender == 'male' else 1.0

    # Adjust for number of meals
    meal_adjustment = 1 + (4 - num_meals) * 0.2

    # Combine all factors
    portion_size_factor = weighted_factor * bmi_adjustment * gender_adjustment * meal_adjustment

    # Ensure the factor stays within a logical range
    portion_size_factor = max(0.8, min(portion_size_factor, 3.0))

    return portion_size_factor



def generate_meal_plan(age, gender, height, weight, activity_level, target, num_meals, allergies):
    def calculate_bmi(weight, height):
        return weight / (height ** 2)

    def calculate_caloric_needs(age, weight, height, gender, activity_level):
        if gender == 1:  # Male
            bmr = 88.362 + (13.397 * weight) + (4.799 * height * 100) - (5.677 * age)
        else:  # Female
            bmr = 447.593 + (9.247 * weight) + (3.098 * height * 100) - (4.330 * age)

        activity_factors = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725}
        return bmr * activity_factors.get(activity_level, 1.2)

    def calculate_target_calories(caloric_needs, target):
        factors = {1: 1.3, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.7}
        return caloric_needs * factors.get(target, 1.0)

    def calculate_macronutrient_targets(target_calories):
        protein_target = target_calories * 0.25 / 4
        carb_target = target_calories * 0.50 / 4
        fat_target = target_calories * 0.25 / 9
        return protein_target, carb_target, fat_target

    bmi = calculate_bmi(weight, height)
    caloric_needs = calculate_caloric_needs(age, weight, height, gender, activity_level)
    target_calories = calculate_target_calories(caloric_needs, target)
    protein_target, carb_target, fat_target = calculate_macronutrient_targets(target_calories)

    calories_per_meal = [target_calories / num_meals] * num_meals
    protein_per_meal = [protein_target / num_meals] * num_meals
    carb_per_meal = [carb_target / num_meals] * num_meals
    fat_per_meal = [fat_target / num_meals] * num_meals

    print(f"\nYour BMI is: {bmi:.2f}")
    print(f"Your daily caloric needs are: {caloric_needs:.2f} kcal")
    print(f"Your target daily calories are: {target_calories:.2f} kcal")

    filtered_food = filter_allergenic_products(df, allergies)

    if filtered_food.empty:
        print("No suitable food items found due to allergies.")
        return []

    meal_plan = []
    total_macros_per_meal = []

    for i in range(num_meals):
        while True:
            sampled_food = filtered_food.sample(min(3, len(filtered_food)))
            sampled_calories = sampled_food['calories'].sum()
            sampled_protein = sampled_food['protein'].sum()
            sampled_carbs = sampled_food['carbohydrate'].sum()
            sampled_fats = sampled_food['total_fat'].sum()
            sampled_fiber = sampled_food['fiber'].sum()
            sampled_sugars = sampled_food['sugars'].sum()
            sampled_saturated_fat = sampled_food['saturated_fat'].sum()

            scale_calories = sampled_calories / calories_per_meal[i] if calories_per_meal[i] > 0 else 1.0
            scale_protein = sampled_protein / protein_per_meal[i] if protein_per_meal[i] > 0 else 1.0
            scale_carbs = sampled_carbs / carb_per_meal[i] if carb_per_meal[i] > 0 else 1.0
            scale_fats = sampled_fats / fat_per_meal[i] if fat_per_meal[i] > 0 else 1.0

            portion_size_factor = calculate_portion_size_factor(
                scale_calories,
                scale_protein,
                scale_carbs,
                scale_fats,
                bmi, gender, num_meals
            )

            sampled_food['calories'] *= portion_size_factor
            sampled_food['protein'] *= portion_size_factor
            sampled_food['carbohydrate'] *= portion_size_factor
            sampled_food['total_fat'] *= portion_size_factor
            sampled_food['fiber'] *= portion_size_factor
            sampled_food['sugars'] *= portion_size_factor
            sampled_food['saturated_fat'] *= portion_size_factor

            if is_within_bounds(
                    sampled_calories * portion_size_factor,
                    sampled_protein * portion_size_factor,
                    sampled_carbs * portion_size_factor,
                    sampled_fats * portion_size_factor,
                    sampled_fiber * portion_size_factor,
                    sampled_sugars * portion_size_factor,
                    sampled_saturated_fat * portion_size_factor,
                    calories_per_meal[i], protein_per_meal[i], carb_per_meal[i], fat_per_meal[i], num_meals
            ):
                adjusted_food = sampled_food.copy()
                for column in ['serving_size', 'calories', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars', 'saturated_fat']:
                    # adjusted_food[column] *= portion_size_factor
                    adjusted_food[column] = (adjusted_food[column] * portion_size_factor).round(2)

                meal_plan.append(adjusted_food)

                # Calculate macronutrients for this meal
                # meal_macros = {
                #     'calories': adjusted_food['calories'].sum(),
                #     'protein': adjusted_food['protein'].sum(),
                #     'carbohydrate': adjusted_food['carbohydrate'].sum(),
                #     'total_fat': adjusted_food['total_fat'].sum()
                # }
                meal_macros = {
                    'calories': round(adjusted_food['calories'].sum(), 2),
                    'protein': round(adjusted_food['protein'].sum(), 2),
                    'carbohydrate': round(adjusted_food['carbohydrate'].sum(), 2),
                    'total_fat': round(adjusted_food['total_fat'].sum(), 2)
                    }


                total_macros_per_meal.append(meal_macros)
                break

    # Format the meal plan for better readability
    formatted_meal_plan = [meal.to_dict('records') for meal in meal_plan]

    # Calculate total macros for all meals
    # total_macros = {
    #     'calories': sum(m['calories'] for m in total_macros_per_meal),
    #     'protein': sum(m['protein'] for m in total_macros_per_meal),
    #     'carbohydrate': sum(m['carbohydrate'] for m in total_macros_per_meal),
    #     'total_fat': sum(m['total_fat'] for m in total_macros_per_meal)
    # }
    total_macros = {
        'calories': round(sum(m['calories'] for m in total_macros_per_meal), 2),
        'protein': round(sum(m['protein'] for m in total_macros_per_meal), 2),
        'carbohydrate': round(sum(m['carbohydrate'] for m in total_macros_per_meal), 2),
        'total_fat': round(sum(m['total_fat'] for m in total_macros_per_meal), 2)
    }

    print(f"Formatted Meal Plan: {formatted_meal_plan}")
    print(f"Total Macros per Meal: {total_macros_per_meal}")
    print(f"Total Macros for All Meals: {total_macros}")
    return formatted_meal_plan, total_macros_per_meal, total_macros
def find_meal_by_nutritional_need(criteria):
    priority_queue = []

    if criteria == 'high_calories':
        for index, row in all_diets_df.iterrows():
            heapq.heappush(priority_queue, (-row['calories'], index))
    elif criteria == 'high_protein':
        for index, row in all_diets_df.iterrows():
            heapq.heappush(priority_queue, (-row['protein'], index))
    elif criteria == 'high_carbs':
        for index, row in all_diets_df.iterrows():
            heapq.heappush(priority_queue, (-row['carbohydrate'], index))
    elif criteria == 'balanced':
        for index, row in all_diets_df.iterrows():
            balance_score = abs(row['protein'] - 15) + abs(row['carbohydrate'] - 40) + abs(row['total_fat'] - 10)
            heapq.heappush(priority_queue, (balance_score, index))
    else:
        print("Invalid criteria selected.")
        return {}

    # Retrieve top 5 meals based on the criteria
    selected_meals = []
    for _ in range(5):
        if priority_queue:
            _, index = heapq.heappop(priority_queue)
            selected_meals.append(all_diets_df.iloc[index].to_dict())

    # Return the selected meals as a dictionary
    if not selected_meals:
        print("No meals found matching the criteria.")
        return {}
    print("Selected meals based on the criteria:")
    print(selected_meals)
    return selected_meals

def main():
    print("Choose an option:")
    print("1. Generate a meal plan")
    print("2. Find a similar product")
    print("3. Find a meal by nutritional need")
    choice = validate_input("Enter your choice (1, 2, or 3): ", 1, 3, int)

    if choice == 1:
        age = validate_input("Enter your age: ", 12, 70, int)
        gender = validate_input("Enter your gender (1 for Male, 2 for Female): ", 1, 2, int)
        height = validate_input("Enter your height in cm (e.g., 175): ", 100, 250, float)
        height = height / 100
        weight = validate_input("Enter your weight in kg: ", 30, 200, float)
        activity_level = validate_input(
            "Enter your activity level (1 for Sedentary, 2 for Light, 3 for Moderate, 4 for Very Active): ", 1, 4, int)
        target = validate_input(
            "Enter your target (1 for Hard Gain, 2 for Gain, 3 for Maintain, 4 for Loss, 5 for Hard Loss): ", 1, 5, int)

        num_meals = validate_input("Enter the number of meals per day (e.g., 3 for traditional meals): ", 2, 4, int)
        if num_meals != 3:
            print("Note: If you choose a number other than 3 (2 or 4), your macronutrients will be divided equally among all meals.")

        allergies = input("Enter your allergies (comma-separated, or leave blank for none): ").split(',')

        generate_meal_plan(age, gender, height, weight, activity_level, target, num_meals, allergies)

    elif choice == 2:
        product_name = input("Enter the product name to find similar items: ")
        find_similar_product(product_name)

    elif choice == 3:
        print("Choose a nutritional need:")
        print("1. High Calories")
        print("2. High Protein")
        print("3. High Carbs")
        print("4. Balanced")
        criteria_choice = validate_input("Enter your choice (1-4): ", 1, 4, int)
        criteria_map = {1: 'high_calories', 2: 'high_protein', 3: 'high_carbs', 4: 'balanced'}
        find_meal_by_nutritional_need(criteria_map[criteria_choice])

# main()

