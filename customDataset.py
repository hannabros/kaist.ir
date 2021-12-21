
import json
from collections import defaultdict
import torch
from tqdm import tqdm


class InputFeature():
    """A single set of features of data"""

    def __init__(self,
                 input_ids,
                 token_type_ids,
                 attention_mask,
                 start_positions,
                 end_positions):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions

def read_file(file_path, is_source=False, source_no="all"):
    with open(file_path) as f:
        data = json.load(f)

    qa_data = defaultdict(list)
    for doc in data['data']:
        if is_source:
            source = doc['source']
        else:
            source = "all"
        filtered = True if source == source_no else False
        if filtered:
            for paragraph in doc['paragraphs']:
                context = paragraph['context']
                for question_and_answers in paragraph['qas']:
                    is_impossible = question_and_answers['is_impossible'] if 'is_impossible' in question_and_answers else None
                    if not is_impossible:
                        question = question_and_answers['question']
                        answers = question_and_answers['answers']    
                        for answer in answers:
                            qa_data['context'].append(context)
                            qa_data['question'].append(question)
                            qa_data['answers'].append(answer)
                            if is_source:
                                qa_data['source'].append(source)
    return qa_data


def convert_to_features(qa_data, tokenizer, max_len, doc_stride):
    encodings = []
    for context, question in tqdm(zip(qa_data['context'], qa_data['question']), total=len(qa_data)):
        encoding = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        sample_mapping = encoding.pop("overflow_to_sample_mapping")
        offset_mapping = encoding.pop("offset_mapping")

        encoding["start_positions"] = []
        encoding["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = encoding["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = encoding.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = qa_data['answers'][sample_index]
            start_char = answers['answer_start']
            end_char = start_char + len(answers['text'])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                encoding["start_positions"].append(cls_index)
                encoding["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                encoding["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                encoding["end_positions"].append(token_end_index + 1)

        encodings.append(encoding)
        del encoding

    return [InputFeature(enc.input_ids,
                         enc.token_type_ids,
                         enc.attention_mask,
                         enc.start_positions,
                         enc.end_positions) for enc in encodings]

def convert_to_eval_features(qa_data, tokenizer, max_len, doc_stride):
    encodings = tokenizer(
        qa_data['question'],
        qa_data['context'],
        truncation="only_second",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    ) 

    sample_mapping = encodings.pop("overflow_to_sample_mapping")
    encodings["example_id"] = []

    for i in range(len(encodings["input_ids"])):
        sequence_ids = encodings.sequence_ids(i)
        context_index = 1

        sample_index = sample_mapping[i]
        encodings["example_id"].append(range(len(qa_data['question']))[sample_index])

        encodings["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(encodings["offset_mapping"][i])
        ]

    return encodings