from collections import defaultdict
from gensim.models import KeyedVectors
from prepare_data import *
from model import label_generator
from training import *



#defining all global variables:
data_path="data/"
train_file="train.jsonl"
test_file="test.jsonl"
label_file="Labels.jsonl"
save_path="saved/"
#vocab threshold 
vocab_thresh=2
#embedding dimension
embed_dim=128
#number of epochs -tried 100
epochs=50
#batch size 
batch_size=5
#laptop without cuda support hence cpu
device='cpu'
#learning rate and the patience 
lr=1e-3
l2reg=5e-4
lr_patience=5
lr_factor=0.5




print("Loading and tokenizing fact descriptions...")
traindev_data = build_dataset(data_path + train_file)
test_data = build_dataset(data_path + test_file)


print("Loading and tokenizing charge descriptions...")
label_data = build_dataset(data_path + label_file)
num_docs = len(traindev_data)
num_sents = len(sum([doc['text'] for doc in traindev_data], []))



print("Creating vocab...")
word_freq = defaultdict(int)
sent_label_freq = defaultdict(int)
doc_label_freq = defaultdict(int)
calc_frequencies(traindev_data, word_freq, sent_label_freq, doc_label_freq)
calc_frequencies(label_data, word_freq)
label_vocab = create_label_vocab(label_data)
vocab = create_vocab(word_freq)



print("Numericalizing all data...")
numericalize_dataset(traindev_data, vocab, label_vocab)
numericalize_dataset(test_data, vocab, label_vocab)
numericalize_dataset(label_data, vocab, label_vocab)
sent_label_wts = torch.from_numpy(calc_label_weights(label_vocab, sent_label_freq, num_sents))
doc_label_wts = torch.from_numpy(calc_label_weights(label_vocab, doc_label_freq, num_docs))
charges = prepare_charges(label_data)
model = label_generator(len(vocab), embed_dim, len(label_vocab), charges['charge_text'], charges['sent_lens'], charges['doc_lens'], device, sent_label_wts, doc_label_wts, None)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2reg)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=lr_patience, factor=lr_factor, verbose=True)
metrics, model = train(model, traindev_data, test_data, optimizer,lr_scheduler=scheduler, num_epochs=epochs, batch_size=batch_size, device=device)
with open(save_path + "metrics.json", 'w') as fw:
	json.dump(metrics, fw)
# torch.save(model, save_path + "model.pt")
torch.save(model.state_dict(), save_path + "model.pt")
