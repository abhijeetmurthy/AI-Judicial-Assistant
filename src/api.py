from collections import defaultdict
from gensim.models import KeyedVectors
from prepare_data import *
from model import label_generator
from training import *
import os
import json

data_path="data/"
train_file="Train-Sent.jsonl"
test_file="Doc.jsonl"
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



traindev_data = build_dataset_from_jsonl(data_path + train_file)
label_data = build_dataset_from_jsonl(data_path + label_file)
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




# data = build_dataset_from_jsonl(data_path + test_file)
def build_dataset_from_api(doc):
    l=[]
    doc['text'] = list(map(lambda x: tokenize_text(x), doc['text']))
    l.append(doc)
    return l

from flask import Flask,request,jsonify  # Flask == 1.0.2
app = Flask(__name__)
HTTP_BAD_REQUEST = 400

@app.route('/', methods=['POST'])
def index():
    input_json= request.get_json()
    print(input_json)
    random_number = random.randint(1, 1000)
    return render_template('template.html', random_number=random_number)


@app.route('/return_inference', methods=['POST'])
def return_inference():
    input_json= request.get_json()
    input_json={"factid": "2000_M_266", "text": ["The relevant and necessary facts to dispose of this petition are: The respondent was working as a Road Transport Inspector in the Regional Office of the Road Transport Corporation, Bhopal and is a public servant as such.", "A complaint for the check period 25.9.1982 to 27.3.1993 was filed stating that he had acquired the property in excess of the known source of his income.", "During the investigation properties and assets belonging to his mother-in- law, father, brother and nephew were shown as assets of the respondent.", "The assets of his wife, who is an income-tax payer and a self earning member, were also connected with the assets of the respondent.", "While submitting charge sheet several important documents, which were collected during the course of investigation, were withheld.", "According to the respondent the said documents supported him.", "If those documents were considered even prima facie there was no scope to frame charges against him."], "doc_labels": ["criminal conspiracy", "cheating"]}
    data=build_dataset_from_api(input_json)
    numericalize_dataset(data, vocab, label_vocab)
    a=infer(model, data, label_vocab, batch_size=5, device='cpu')
    return_json= {'prediction':a[0]}
    return_json = json.dumps(return_json)
    return return_json
            

    

@app.route('/return_label_f1', methods=['GET'])
def return_metrics():
    with open('saved\metrics.json') as f:
        data = json.load(f)
    return_json= {'F1_scores_for_each_label':data['label-F1']}
    return_json = json.dumps(return_json)
    return return_json

@app.route('/return_f1', methods=['GET'])
def return_metrics_f1():
    with open('saved\metrics.json') as f:
        data = json.load(f)
    return_json= {'F1':data['macro-F1']}
    return_json = json.dumps(return_json)
    return return_json



if __name__ == '__main__':

    #set it to True if running in command promt(for debugging)
    app.run(host="localhost",port='8081')