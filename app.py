import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib

from category import catagories

# 2. Create the app object
app = FastAPI()
model = joblib.load('model_joblib')




# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_category(data:catagories):
    data = data.dict()
    category=data['category']

   
    prediction = model.wv.most_similar([[category]])
    
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)