import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import nltk

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inv_labels_dict = {0: 'O', 1: 'PRD', 2: 'PRD', 3: 'POP', 4: 'POP',
                   5: 'HARM', 6: 'HARM', 7: 'POB', 8: 'POB', 9: 'IND',
                   10: 'IND', 11: 'TRG', 12: 'TRG', 13: 'HM1', 14: 'HM1',
                   15: 'HM2', 16: 'HM2', -100: '-100'}


def get_tokens(input_sentence):
    tokens = nltk.word_tokenize(input_sentence)
    # Add spetial tokens into tokens list
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    return tokens


def label_encoding_v2(encoding, labels):
    return [encoding[label] for label in labels]


class PipadiTransfomer():

    def __init__(self, model_name, tokenizer, max_len, batch_size):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.eval_ids = None
        self.eval_att_masks = None
        self.eval_dataloader = None

    def tokenize(self, input_sentence):
        # Add special tokens and initial tokenization
        input_tokens = get_tokens(input_sentence)

        # Invoke DistilmBERT tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer)

        # Map each token to BERT vocabulary index using mBERT tokenizer
        # Add padding (if necessary) to reach MAX_LEN number of tokens
        eval_inputs = tokenizer(input_tokens,
                                add_special_tokens=False,
                                max_length=self.max_len,
                                is_split_into_words=True)

        # Obtain model inputs: sub_tokens, ids and attention masks
        eval_subtokens = [tokenizer.convert_ids_to_tokens(ids) for ids in eval_inputs['input_ids']]
        eval_ids = eval_inputs['input_ids']
        eval_att_masks = eval_inputs['attention_mask']

        # Convert lists into torch tensors
        self.eval_ids = torch.tensor(eval_ids)
        self.eval_att_masks = torch.tensor(eval_att_masks)

        return eval_subtokens

    def predict(self, num_labels):
        # No fit as model is pre-trained
        model = DistilBertForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels)

        # Model in evaluation mode
        model.eval()

        # Tracking variables
        predictions = []

        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(self.eval_ids[None, :], attention_mask=self.eval_att_masks[None, :])

        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)

        # Post-process predictions
        predictions_v2 = [np.argmax(prediction, axis=2).flatten().tolist() for prediction in predictions]
        predictions_v2 = np.array([prediction for predictions in predictions_v2 for prediction in predictions])

        # Generate agg predictions
        predictions_v3 = label_encoding_v2(inv_labels_dict, predictions_v2)

        return predictions_v3
