from tqdm import tqdm
import pickle
def build_dataset_from_jsonl(data_file):
	data = []
	with open(data_file) as fr:
		for line in tqdm(fr):
			doc = json.loads(line)

			doc['text'] = list(map(lambda x: tokenize_text(x), doc['text']))
			data.append(doc)
	return data

def tokenize_text(text):
	parsed_text = parser(text)
	cleaned = [tok.lemma_.lower() if tok.ent_type == 0 else '[' + tok.ent_type_ + ']'\
		for tok in parsed_text if not any([tok.is_punct, tok.is_stop, tok.is_digit, len(tok) == 0])]
	cleaned = [group[0] for group in groupby(cleaned)]
	return cleaned

file = open("rcv1_raw_text.p",'rb')
object_file = pickle.load(file)
print(object_file)
file.close()