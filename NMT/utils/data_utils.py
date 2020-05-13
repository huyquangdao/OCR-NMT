# from tokenizers import (ByteLevelBPETokenizer,
#                             CharBPETokenizer,
#                             SentencePieceBPETokenizer,
#                             BertWordPieceTokenizer)
import torch
import spacy

def create_tokenizer(corpus_file_path, vocab_size):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(corpus_file_path,vocab_size, lower = True)
    tokenizer.add_special_tokens(['<SOS>','<PAD>','<EOS>'])
    return tokenizer    


def pad_to_max_length(token_ids, tokenizer, max_seq_length = 100):

    pad_ids = tokenizer.token_to_id('<PAD>')

    sos_ids = tokenizer.token_to_id('<SOS>')
    eos_ids = tokenizer.token_to_id('<EOS>')
    
    if len(token_ids) > max_seq_length - 2:
        while len(token_ids) > max_seq_length - 2:
            token_ids.pop()
    
    token_ids = [sos_ids] + token_ids + [eos_ids]
    
    length_to_pad = max_seq_length - len(token_ids)

    # to add <SOS> token and <EOS> token

    pad = [pad_ids] * length_to_pad

    token_ids = token_ids + pad

    return token_ids

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention


# def translate_sentence(sentence, src_field, trg_field, src_tokenizer, des_tokenizer,  model, device, max_len = 50):
    
#     model.eval()
        
#     src = src_tokenizer.encode(sentence)

#     tokens = [src.token_to_id('<SOS>')] + src.ids + [src.token_to_id('<EOS>')]
        
#     src_indexes = tokens

#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
#     src_mask = model.make_src_mask(src_tensor)
    
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)

#     trg_indexes = [des_tokenizer.token_to_id('<SOS>')]

#     for i in range(max_len):

#         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

#         trg_mask = model.make_trg_mask(trg_tensor)
        
#         with torch.no_grad():
#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
#         pred_token = output.argmax(2)[:,-1].item()
        
#         trg_indexes.append(pred_token)

#         if pred_token == des_tokenizer.token_to_id('<EOS>'):
#             break
    
#     trg_tokens = des_tokenizer.decode(trg_indexes)
    
#     return trg_tokens[1:], attention
