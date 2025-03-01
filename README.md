# IT3385-MLOPS-wileen-roanne
Project Description: 
MLOPS streamlines the process of deploying Machine Learning models to production and then maintaining and monitoring them. In this project, we will be covering two types of machine leanring models.
1. Wheat type classification model: To predict wheat type based on its attributes on its kernel. Type 1- Kama, Type 2 - Rosa and Type 3 - Canadian.
2. Car price regression model: To predict used care in India based on its characteristics.

For deployment locally, run the respective requirements.txt file, then proceed to run the py file. 
- for wheat prediction: python wileenAPP.py
- for car price prediction: python src/roanne_carapp.py

## Wheat Type Classification Model 
Description: This classification model predicts what type of seed based on it's kernel attribute. There are a total of 3 seeds. Type 1- Kama, Type 2 - Rosa and Type 3 - Canadian.
File Structure
wileen
- artifacts
  - seed_pipline.pkl
  - seed_type_classification.pkl
- config
  - wheat.yaml
- data
  - 03_Wheat_Seeds.csv
- notebook
  - wileenFinalModelling.ipynb
- templates
  - wheat.html
- requirements.txt
- wileenAPP.py

Deployed model on render. Link: https://mlopsindividual.onrender.com/ 

## Car Price Regression Model 
Description: This regression model predicts the price of used cars in india based on its characteristics (For example, brand, model, engine seats etc)  
File Structure
roanne
- artifacts
  - used_car_price_model.joblib
  - used_car_price_model.pkl
- config
  - car.yaml
- src
  - roanne_carapp.py
- data
  - 02_Used_Car_Prices.xlsx
- notebook
  - roanneFinalModelling.ipynb
- templates
  - roanne_car.html
- requirements.txt

Deployed model on render. Link: https://mlops-car.onrender.com/predict


