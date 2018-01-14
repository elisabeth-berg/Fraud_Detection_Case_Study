from flask import Flask, render_template, request, jsonify
import requests
from pymongo import MongoClient


client = MongoClient('mongodb://localhost:27017/')
db = client.events_fraud
events = db.events
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # get the data from mongod
    # [{user_id: ..., prediced_prob: ..., ...}]
    data_from_mongo = events.find().sort("largest", -1).limit(10)
    return render_template('fraud.html', event_data=data_from_mongo)



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
