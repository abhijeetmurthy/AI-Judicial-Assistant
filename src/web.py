
import spacy
nlp = spacy.load('en_core_web_sm')
from collections import defaultdict
from gensim.models import KeyedVectors
from prepare_data import *
from model import label_generator
from training import *
import os
import json
from flask import Flask,request,jsonify,render_template

data_path="data/"
train_file="train.jsonl"
test_file="doc.jsonl"
label_file="Labels.jsonl"
save_path="saved/"
vocab_thresh=2
embed_dim=128
epochs=50
batch_size=5
device='cpu'
lr=1e-3
l2reg=5e-4
lr_patience=5
lr_factor=0.5
model_path = 'saved\model.pt'



traindev_data = build_dataset(data_path + train_file)
label_data = build_dataset(data_path + label_file)
num_docs = len(traindev_data)
num_sents = len(sum([doc['text'] for doc in traindev_data], []))
word_freq = defaultdict(int)
sent_label_freq = defaultdict(int)
doc_label_freq = defaultdict(int)
calc_frequencies(traindev_data, word_freq, sent_label_freq, doc_label_freq)
calc_frequencies(label_data, word_freq)
label_vocab = create_label_vocab(label_data)
vocab = create_vocab(word_freq)
ptemb_matrix = None
numericalize_dataset(traindev_data, vocab, label_vocab)
numericalize_dataset(label_data, vocab, label_vocab)
sent_label_wts = torch.from_numpy(calc_label_weights(label_vocab, sent_label_freq, num_sents))
doc_label_wts = torch.from_numpy(calc_label_weights(label_vocab, doc_label_freq, num_docs))
charges = prepare_charges(label_data)
model = label_generator(len(vocab), embed_dim, len(label_vocab), charges['charge_text'], charges['sent_lens'], charges['doc_lens'], device, sent_label_wts, doc_label_wts, ptemb_matrix)
model.load_state_dict(torch.load(model_path))

# data = build_dataset(data_path + test_file)
def build_dataset_from_api(doc):
    l=[]
    doc['text'] = list(map(lambda x: tokenize_text(x), doc['text']))
    l.append(doc)
    return l

app = Flask(__name__)

@app.route("/model.html", methods=["POST", "GET"])
def inference():
    if request.method == "POST":
        ip = request.form["area"]
        doc=nlp(ip)
        sentences = list(doc.sents)
        sentences=[i.text for i in sentences]
        input_json=({"text":sentences})
        data=build_dataset_from_api(input_json)
        numericalize_dataset(data, vocab, label_vocab)
        a=infer(model, data, label_vocab, batch_size=5, device='cpu')
        print(a[0])
        op="The topics violated are as follows:"
        for i in range(len(a[0])):
            op=op+("\n")
            op=op+str(i+1)+"-"+str(a[0][i])
        print(op)
        return render_template("model.html",content_input=ip,content_output=op)
    else:
        return render_template("model.html",content="")

@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")


@app.route("/landing.html", methods=["GET"])
def landing():
    return render_template("landing.html")

@app.route("/metrics.html", methods=["GET"])
def metrics():
    return render_template("metrics.html")

@app.route("/label.html", methods=["GET"])
def label():
    return render_template("label.html")

@app.route("/about_team.html", methods=["GET"])
def about_team():
    return render_template("about_team.html")



if __name__ == '__main__':

    #set it to True if running in command promt(for debugging)
    app.run(host="localhost",port='8081',debug=True)


# @app.route('/return_inference', methods=['POST'])
# def return_inference():
#     input_json= request.get_json()
#     input_json={"text": ["The relevant and necessary facts to dispose of this petition are: The respondent was working as a Road Transport Inspector in the Regional Office of the Road Transport Corporation, Bhopal and is a public servant as such.", "A complaint for the check period 25.9.1982 to 27.3.1993 was filed stating that he had acquired the property in excess of the known source of his income.", "During the investigation properties and assets belonging to his mother-in- law, father, brother and nephew were shown as assets of the respondent.", "The assets of his wife, who is an income-tax payer and a self earning member, were also connected with the assets of the respondent.", "While submitting charge sheet several important documents, which were collected during the course of investigation, were withheld.", "According to the respondent the said documents supported him.", "If those documents were considered even prima facie there was no scope to frame charges against him."], "doc_labels": ["criminal conspiracy", "cheating"]}
#     data=build_dataset_from_api(input_json)
#     numericalize_dataset(data, vocab, label_vocab)
#     a=infer(model, data, label_vocab, batch_size=5, device='cpu')
#     return_json= {'prediction':a[0]}
#     return_json = json.dumps(return_json)
#     return return_json

# @app.route('/return_label_f1', methods=['GET'])
# def return_metrics():
#     with open('saved\metrics.json') as f:
#         data = json.load(f)
#     return_json= {'F1_scores_for_each_label':data['label-F1']}
#     return_json = json.dumps(return_json)
#     return return_json

# @app.route('/return_f1', methods=['GET'])
# def return_metrics_f1():
#     with open('saved\metrics.json') as f:
#         data = json.load(f)
#     return_json= {'F1':data['macro-F1']}
#     return_json = json.dumps(return_json)
#     return return_json
