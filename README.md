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
  - Greedy Search Algorithm for meal finder.
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
   git clone https://github.com/0Maxbon0/AI-Powered-Meal-Planning-System.git
   cd AI-Powered-Meal-Planning-System
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
📂 AI-Powered-Meal-Planning-System
├── generate_meal.py       # Core logic for meal generation and filtering
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
- Web app version.

---

## Contributors

The development of this project was made possible through the collaborative efforts of a dedicated team with diverse expertise:

- **Maxim Mamdouh Salib** - Lead AI Developer, responsible for machine learning models and core logic implementation.
- **Abdullah Salah El Sayed** - Algorithm Specialist, focused on optimizing recommendation systems and computational efficiency.
- **Kerollos Fawzy Kamel** - Backend Engineer, developed and integrated Flask-based backend functionalities.
- **Kerollos Nabil Worthy** - UI/UX Designer, ensured a user-friendly and aesthetically pleasing interface.
- **Flora Osama Shukry** - UI/UX Designer & Data Analyst, contributed to interface design and dataset preparation.
- **Kholoud Ashraf Ibrahim** - UI/UX Designer & Data Analyst, involved in interface design and data curation.
- **Enjy Bushra Tawfik** - UI/UX Designer & Data Analyst, supported interface design and data refinement.

