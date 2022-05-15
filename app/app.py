import re
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np 
import joblib
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

model = joblib.load('spam_model.pkl')
cv = joblib.load('vector.pkl')

@app.route('/',methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    
    msg = request.form.get('message')
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = msg)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()

    output = model.predict(temp)
    if output == 0:
      result = "This Message is Not a SPAM Message."
    else:
      result = "This Message is a SPAM Message." 
    return render_template('index.html', result=result,message=msg)      

  else:
    return render_template('index.html')  


if __name__ == '__main__':
    app.run(debug=True)