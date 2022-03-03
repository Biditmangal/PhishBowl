from flask import Flask,render_template,request

import URLFeatureExtraction
import pickle
import pandas as pd 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/getURL',methods=['GET','POST'])
def getURL():
    if request.method == 'POST':
        url = request.form['url']
        print(url)
        features_url=[]
        features_url.append(URLFeatureExtraction.featureExtraction(url))
        feature_names = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']
        data_to_pass_in_the_model = pd.DataFrame(features_url, columns= feature_names)
        print(data_to_pass_in_the_model)
        XGBoost = pickle.load(open("XGBoostClassifier.pickle", "rb"))
        predicted_value = XGBoost.predict(data_to_pass_in_the_model)
        #print(predicted_value)
        if predicted_value == 0:    
            value = "Legitimate"
            return render_template("home.html",error=value)
        else:
            value = "Phishing"
            return render_template("home.html",error=value)
if __name__ == "__main__":
    app.run(debug=True)