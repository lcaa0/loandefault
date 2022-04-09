#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        purchases = request.form.get("purchases")
        suppcard = request.form.get("suppcard")
        print(purchases, suppcard)
        model1 = joblib.load("CART")
        loan = float(loan)
        age = float(age)
        imcome= float(income)
        pred1 = clfmodel.predict([[loan,income,age]])
        model1= joblib.load("clfmodel")
        pred2 = rfcmodel.predict([[loan,income,age]])
        model2= joblib.load("rfcmodel")                          
        pred3 = gbcmodel.predict([[purchases,suppcard]])
        model3= joblib.load("gbcmodel")
        return(render_template("index.html", result1=pred1, result2=pred2, result3=pred3))
    else:
        return(render_template("index.html", result1="2", result2="2", result3='2'))

if __name__ == "__main__":
    app.run()


# In[ ]:




