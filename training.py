from pycaret.regression import *
import pandas as pd

data = pd.read_csv('housing.csv')
print(data.columns.tolist())

reg1 = setup(
    data,
    target='median_house_value',
    session_id=123,
    numeric_features=[
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income'
    ],
    categorical_features=[
        'ocean_proximity'
    ]
)

best_model = compare_models()

save_model(best_model, 'prediction_house_price_model')

print("Model trained and saved successfully!")

create_api(best_model, 'house_price_api')