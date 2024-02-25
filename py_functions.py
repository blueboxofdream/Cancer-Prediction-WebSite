import pandas as pd

def check_user_exist (email, cnxn):
    query= 'select from USERINFO where email="+"*"+str(email)+"*"'
    df = pd.read_sql (query, cnxn)
    return df.shape[0]

def signup_data(username,email,password):
    query= "INSERT INTO USERINFO"\
    "VALUES ("+"'"+str(username)+"'"+",'"+str(email)+"','"+str(password)+"'"+")"
    print(query)
    # df= pd.read_sql (query,cnxn)
    return query