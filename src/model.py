import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as U

class BiRNN(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.rnn = nn.GRU(embed_dim, embed_dim // 2, batch_first = True, bidirectional = True)
        
        self.embed_dim = embed_dim
        self.device = device
        
    def forward(self, sequence_in, lens): 
        batch_size = sequence_in.shape[0]
        hidden = torch.randn(2, batch_size, self.embed_dim // 2)
        
        packed_sequence_in = U.pack_padded_sequence(sequence_in, lens, batch_first = True, enforce_sorted = False)
        

        packed_sequence_out, sequence_rep = self.rnn(packed_sequence_in, hidden)
        
        sequence_out = U.pad_packed_sequence(packed_sequence_out, batch_first = True)[0]
        sequence_rep = sequence_rep.permute(1,0,2).contiguous().view(batch_size, -1)
        
        return sequence_out, sequence_rep
    
class SeqAttn(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.context = nn.Parameter(torch.rand((embed_dim, 1)))
        
        self.device = device
    
    def forward(self, sequence, lens, dcontext=None): 
        batch_size = sequence.shape[0]
        max_len = sequence.shape[1]
 
        mask = (torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)).float()
        
        sequence = torch.tanh(self.fc(sequence.view(-1, self.embed_dim)))

        sequence = sequence.view(batch_size, -1, self.embed_dim)
        sequence = sequence * mask.unsqueeze(2)

        context = self.context.expand(batch_size, self.embed_dim, 1) if dcontext is None else dcontext.unsqueeze(2)

        scores = torch.bmm(sequence, context).squeeze(2)
        probs = F.softmax(scores, dim = 1).unsqueeze(2)
        

        sequence_wt = torch.sum(sequence * probs, dim = 1)
        return sequence_wt
    
class RNNEncoder(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.rnn_layer = BiRNN(embed_dim, device)
        self.attn_layer = SeqAttn(embed_dim, device)
        
    def forward(self, sequence, lens, context=None):
        encoded, _ = self.rnn_layer(sequence, lens)
        weighted = self.attn_layer(encoded, lens, context) 
        return weighted
    
class HAN(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.sent_encoder = RNNEncoder(embed_dim, device)
        self.doc_encoder = RNNEncoder(embed_dim, device)
        
    def forward(self, sequence, sent_lens, doc_lens, sent_context=None, doc_context=None):
        
        encoded_sents = self.sent_encoder(sequence, sent_lens, sent_context)
        
        docs = U.pad_sequence(torch.split(encoded_sents, doc_lens.tolist()), batch_first=True)
        encoded_docs = self.doc_encoder(docs, doc_lens, doc_context)
        
        return encoded_sents, encoded_docs
    
class TFEncoder(nn.Module):
    def __init__(self, embed_dim, device, nhead=16, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.tf_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.embed_dim, nhead=nhead), num_layers)
        self.attn_layer = SeqAttn(embed_dim, device)
        
    def forward(self, sequence, lens, context=None): 
        encoded = self.tf_layer(sequence * math.sqrt(self.embed_dim)) 
        weighted = self.attn_layer(encoded, lens, context) 
        return weighted
    
class label_generator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, 
                 charge_text, charge_sent_lens, charge_doc_lens, 
                 device='cpu', sent_label_wts=None, doc_label_wts=None, pretrained=None, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        
        self.charge_text = charge_text
        self.charge_sent_lens = charge_sent_lens
        self.charge_doc_lens = charge_doc_lens
        
        self.device = device
        self.sent_label_wts = sent_label_wts if sent_label_wts is not None else torch.ones((self.num_labels,))
        self.doc_label_wts = doc_label_wts if doc_label_wts is not None else torch.ones((self.num_labels,))
        
        self.embedder = nn.Embedding(self.vocab_size, self.embed_dim)
        if pretrained is not None:
            self.embedder.weight.data.copy_(torch.from_numpy(pretrained))
            
        self.fact_encoder = HAN(self.embed_dim, self.device)
        self.charge_encoder = HAN(self.embed_dim, self.device)
        self.charge_aggregator = TFEncoder(self.embed_dim, self.device)

        self.sent_context = nn.Linear(self.embed_dim, self.embed_dim)
        self.doc_context = nn.Linear(self.embed_dim, self.embed_dim)
        self.charge_context = nn.Linear(self.embed_dim, self.embed_dim)

        self.classifier = nn.Linear(2 * self.embed_dim, self.num_labels)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, batch):
        facts = batch['fact_text'] 
        charges = self.charge_text
        
        num_fsents = batch['sent_lens'].size(0)
        num_fdocs = batch['doc_lens'].size(0)
        num_csents = self.charge_sent_lens.size(0)
        num_cdocs = self.charge_doc_lens.size(0)
        
        facts = self.embedder(facts) 
        charges = self.embedder(charges) 
        
        fact_sents, facts = self.fact_encoder(facts, batch['sent_lens'], batch['doc_lens'])

        # code for sentence level classifications
        sent_context = self.sent_context(fact_sents) 
        doc_context = self.doc_context(fact_sents) 
        charge_context = self.charge_context(fact_sents) 
        
        sent_charges = charges.repeat(num_fsents, 1, 1)
        sent_context = sent_context.repeat(1, num_csents).view(-1, self.embed_dim) 
        doc_context = doc_context.repeat(1, num_cdocs).view(-1, self.embed_dim) 
        
        _, sent_charges = self.charge_encoder(sent_charges, self.charge_sent_lens.repeat(num_fsents), self.charge_doc_lens.repeat(num_fsents), sent_context, doc_context) # [FS*CD, E]
        
        sent_charges = sent_charges.view(num_fsents, -1, self.embed_dim)
        charge_lens = torch.tensor([self.num_labels] * num_fsents, dtype=torch.long, device='cpu')
        sent_charges = self.charge_aggregator(sent_charges, charge_lens, charge_context) 
        
        sent_logits = self.dropout(self.classifier(torch.cat([fact_sents, sent_charges], dim=1)))
        
        # code for document level classifications
        sent_context = self.sent_context(facts) 
        doc_context = self.doc_context(facts)
        charge_context = self.charge_context(facts) 
        
        doc_charges = charges.repeat(num_fdocs, 1, 1) 
        sent_context = sent_context.repeat(1, num_csents).view(-1, self.embed_dim) 
        doc_context = doc_context.repeat(1, num_cdocs).view(-1, self.embed_dim) 
        
        _, doc_charges = self.charge_encoder(doc_charges, self.charge_sent_lens.repeat(num_fdocs), self.charge_doc_lens.repeat(num_fdocs), sent_context, doc_context) # [FD*CD, E]
        
        doc_charges = doc_charges.view(num_fdocs, -1, self.embed_dim) 
        charge_lens = torch.tensor([self.num_labels] * num_fdocs, dtype=torch.long, device='cpu')
        doc_charges = self.charge_aggregator(doc_charges, charge_lens, charge_context) 
        
        doc_logits = self.dropout(self.classifier(torch.cat([facts, doc_charges], dim=1)))
        
        # calculation of loss and prections
        loss = 0
        if 'sent_labels' in batch:
            loss += F.binary_cross_entropy_with_logits(sent_logits, batch['sent_labels'], pos_weight=self.sent_label_wts)
        if 'doc_labels' in batch:
            loss += F.binary_cross_entropy_with_logits(doc_logits, batch['doc_labels'], pos_weight=self.doc_label_wts)
            
        model_out = {'loss': loss}
        model_out['sent_preds'] = (torch.sigmoid(sent_logits) > 0.5).float().detach()
        model_out['doc_preds'] = (torch.sigmoid(doc_logits) > 0.5).float().detach()
        
        return model_out
