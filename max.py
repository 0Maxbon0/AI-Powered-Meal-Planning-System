import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 1. تحميل البيانات
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 50, 45, 60, 32],
    'Height': [170, 165, 180, 175, 160, 172, 168, 185],
    'Current_Weight': [80, 90, 100, 85, 120, 110, 75, 95],
    'Target_Weight': [70, 80, 85, 75, 90, 85, 70, 80],
    'Days': [30, 60, 90, 45, 120, 80, 150, 100],
    'Activity_Level': [1, 2, 3, 2, 1, 2, 3, 2],  # 1: Sedentary, 2: Moderate, 3: Active
    'Diet_Type': [1, 2, 1, 3, 2, 4, 5, 6],  # 1: Keto, 2: Vegetarian, 3: Paleo, 4: Vegan, 5: Mediterranean, 6: Low Carb
    'Allergies': [0, 1, 0, 0, 1, 0, 0, 1],  # 0: None, 1: Gluten
    'Chronic_Diseases': [0, 1, 0, 0, 1, 0, 0, 1],  # 0: None, 1: Diabetes
    'Daily_Meals': [3, 4, 3, 4, 3, 4, 3, 4],
    'Daily_Calorie_Deficit': [500, 400, 600, 450, 350, 500, 400, 600]
})

# 2. تقسيم البيانات إلى مدخلات ومخرجات
X = data[['Age', 'Height', 'Current_Weight', 'Target_Weight', 'Days', 'Activity_Level', 'Diet_Type', 'Allergies', 'Chronic_Diseases']]
y_meals = data['Daily_Meals']  # المخرجات: عدد الوجبات
y_calories = data['Daily_Calorie_Deficit']  # المخرجات: السعرات الحرارية

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_meals_train, y_meals_test = train_test_split(X, y_meals, test_size=0.2, random_state=42)
_, _, y_calories_train, y_calories_test = train_test_split(X, y_calories, test_size=0.2, random_state=42)

# 3. تدريب نموذج XGBoost لتوقع عدد الوجبات اليومية
xgb_meals_model = XGBRegressor(random_state=42)
xgb_meals_model.fit(X_train, y_meals_train)

# 4. تدريب نموذج XGBoost لتوقع السعرات الحرارية اليومية
xgb_calories_model = XGBRegressor(random_state=42)
xgb_calories_model.fit(X_train, y_calories_train)

# 5. التقييم
y_meals_pred_xgb = xgb_meals_model.predict(X_test)
y_calories_pred_xgb = xgb_calories_model.predict(X_test)

print("Meals Model MAE:", mean_absolute_error(y_meals_test, y_meals_pred_xgb))
print("Calories Model MAE:", mean_absolute_error(y_calories_test, y_calories_pred_xgb))

# 6. إضافة المزيد من أنواع الدايت
diet_types = {
    1: "Keto",
    2: "Vegetarian",
    3: "Paleo",
    4: "Vegan",
    5: "Mediterranean",
    6: "Low Carb"
}

# 7. توقع جديد بناءً على مدخلات المستخدم
def get_user_input():
    print("Please enter the following details:")
    age = int(input("Age: "))
    height = int(input("Height (cm): "))
    current_weight = int(input("Current Weight (kg): "))
    target_weight = int(input("Target Weight (kg): "))
    days = int(input("Number of Days for the Diet: "))
    activity_level = int(input("Activity Level (1: Sedentary, 2: Moderate, 3: Active): "))
    diet_type = int(input("Diet Type (1: Keto, 2: Vegetarian, 3: Paleo, 4: Vegan, 5: Mediterranean, 6: Low Carb): "))
    allergies = int(input("Any Allergies (0: None, 1: Gluten): "))
    chronic_diseases = int(input("Any Chronic Diseases (0: None, 1: Diabetes): "))
    
    user_input = pd.DataFrame({
        'Age': [age],
        'Height': [height],
        'Current_Weight': [current_weight],
        'Target_Weight': [target_weight],
        'Days': [days],
        'Activity_Level': [activity_level],
        'Diet_Type': [diet_type],
        'Allergies': [allergies],
        'Chronic_Diseases': [chronic_diseases]
    })
    
    return user_input

# 8. الحصول على المدخلات من المستخدم
user_input = get_user_input()

# 9. التنبؤ بنوع الدايت
diet_type = user_input['Diet_Type'][0]
predicted_meals = xgb_meals_model.predict(user_input)
predicted_calories = xgb_calories_model.predict(user_input)

# 10. تغيير عدد الوجبات والسعرات الحرارية بناءً على الأيام
daily_meals = []
daily_calories = []

for day in range(user_input['Days'][0]):
    # تعديل عدد الوجبات بشكل صحيح (عدد صحيح)
    daily_meals.append(int(round(predicted_meals[0] + np.random.randint(-1, 2))))  # التغيير يوميًا بشكل صحيح
    daily_calories.append(int(predicted_calories[0] + np.random.randint(-50, 51)))  # التغيير في السعرات الحرارية

# 11. عرض النتائج
print(f"Predicted Diet Type: {diet_types[diet_type]}")
print(f"Predicted Daily Meals (varied over {user_input['Days'][0]} days): {daily_meals}")
print(f"Predicted Daily Calorie Deficit (varied over {user_input['Days'][0]} days): {daily_calories}")

# 12. عرض نوع الخوارزمية المستخدمة
print("Algorithm used: XGBoost Regressor")
