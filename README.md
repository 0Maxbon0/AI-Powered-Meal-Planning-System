# AI-Powered Meal Planning System

This repository contains the codebase for an AI-driven meal planning application that generates customized meal plans, accommodates dietary restrictions, and provides nutritional recommendations.

---

## Features

- **Custom Meal Plans**: Generate personalized meal plans for 2-4 meals per day based on user data such as age, weight, height, and activity level.
- **Allergy and Dietary Filtering**: Excludes meals based on allergies (e.g., nuts, lactose, gluten) and dietary choices (e.g., vegan).
- **Food Recommendations**: Suggests alternative food items using K-Nearest Neighbors (KNN).
- **Nutritional Optimization**: Provides plans tailored to specific dietary needs (e.g., high-protein, low-carb, balanced).

---

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap
- **Machine Learning**:
  - KNN for food similarity analysis.
  - XGBoost for calorie and meal frequency predictions.
- **Data**:
  - `Cleaned_Data_Final.csv`: Nutritional dataset for various food items.
  - `All_Diets.csv`: Pre-defined dietary categories.

---

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/meal-planner.git
   cd meal-planner
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python meal_planner.py
   ```

4. **Access the Web App**:
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Repository Structure

```
📂 meal-planner
├── generate_meal.py       # Core logic for meal generation and filtering
├── max.py                 # Machine learning models for meal prediction
├── meal_planner.py        # Flask web application
├── Cleaned_Data_Final.csv # Cleaned nutritional dataset
├── All_Diets.csv          # Dietary dataset
├── templates/             # HTML templates for the web interface
├── static/                # Static files (CSS, JS, images)
└── requirements.txt       # Python dependencies
```

---

## Future Enhancements

- Integration with external APIs for real-time nutritional data.
- Advanced filtering for more dietary preferences.

---

## Contributors

- **Your Name** - Developer
- **Collaborator Name** - Data Scientist
- **Collaborator Name** - UI/UX Designer

---

