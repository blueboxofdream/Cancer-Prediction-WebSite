from pydantic import BaseModel
class LungCancer(BaseModel):
    AGE : int
    SMOKING : int
    YELLOW_FINGERS : int
    ANXIETY : int
    PEER_PRESSURE : int
    CHRONIC_DISEASE : int
    # FATIGUE : int
    # ALLERGY : int
    WHEEZING : int
    ALCOHOL_CONSUMING : int
    COUGHING : int
    SHORTNESS_OF_BREATH : int
    SWALLOWING_DIFFICULTY : int
    CHEST_PAIN : int
    GENDER_NEW : int