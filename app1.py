import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

origins = ["http://127.0.0.1:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    joblib_in = open("model_fit.joblib", "rb")
    model_fit = joblib.load(joblib_in)
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model_fit = None

templates = Jinja2Templates(directory="templates")

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "results": ""})
    # return RedirectResponse(url='/predict', status_code=303)

@app.post('/predict', response_class=HTMLResponse)
async def predict_lungcancer(request: Request, AGE: int = Form(...), SMOKING: int = Form(...), YELLOW_FINGERS: int = Form(...), ANXIETY: int = Form(...), 
                            PEER_PRESSURE: int = Form(...), CHRONIC_DISEASE: int = Form(...), WHEEZING: int = Form(...), ALCOHOL_CONSUMING: int = Form(...), 
                            COUGHING: int = Form(...), SHORTNESS_OF_BREATH: int = Form(...), SWALLOWING_DIFFICULTY: int = Form(...), CHEST_PAIN: int = Form(...), 
                            GENDER_NEW: int = Form(...)):

    data = {
        'AGE': AGE,
        'SMOKING': SMOKING,
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ANXIETY': ANXIETY,
        'PEER_PRESSURE': PEER_PRESSURE,
        'CHRONIC_DISEASE': CHRONIC_DISEASE,
        'WHEEZING': WHEEZING,
        'ALCOHOL_CONSUMING': ALCOHOL_CONSUMING,
        'COUGHING': COUGHING,
        'SHORTNESS_OF_BREATH': SHORTNESS_OF_BREATH,
        'SWALLOWING_DIFFICULTY': SWALLOWING_DIFFICULTY,
        'CHEST_PAIN': CHEST_PAIN,
        'GENDER_NEW': GENDER_NEW
    }

    prediction_value = model_fit.predict([list(data.values())])

    prediction = "You have LOW RISK of Lung Cancer" if prediction_value[0] < 0.5 else "You have HIGH RISK of Lung Cancer"

    return templates.TemplateResponse("home.html", {"request": request, "results": prediction})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app1:app --reload
