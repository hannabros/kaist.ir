
import os, sys
import yaml
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm
import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from configMgr import config_parser, parser
from customDataset import read_file, convert_to_features, convert_to_eval_features
from customBertModel import CustomizedBertForQuestionAnswering

import log
logger = log.setup_custom_logger('root')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def validation(valid_loader, model, device):
    model.eval()
    pred_start_logits, pred_end_logits = [], []
    with torch.no_grad():
        val_losses, acc = [], []
        valid_example_ct = 0
        for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            valid_example_ct += batch.input_ids.shape[0]
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            start_positions = batch.start_positions.to(device)
            end_positions = batch.end_positions.to(device)

            outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions)
            
            loss = outputs[0].item()
            val_losses.append(loss)
            start_pred = torch.argmax(outputs[1], dim=1)
            pred_start_logits.append(outputs[1])
            end_pred = torch.argmax(outputs[2], dim=1)
            pred_end_logits.append(outputs[2])
            acc.append(((start_pred == start_positions).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_positions).sum()/len(end_pred)).item())
        avg_loss = sum(val_losses)/len(val_losses) 
        avg_acc = sum(acc)/len(acc)

        print(f"Accuracy of the model on the {valid_example_ct} " +
            f"test samples: {100 * avg_acc}% with valid loss: {avg_loss}")
        
    return pred_start_logits, pred_end_logits

def postprocess_qa_predictions(examples, features, all_start_logits, all_end_logits, n_best_size = 20, max_answer_length = 30):
    example_id_to_index = {k: i for i, k in enumerate(examples.index)}
    features_per_example = defaultdict(list)
    for i, e_id in enumerate(features['example_id']):
        features_per_example[example_id_to_index[e_id]].append(i)

    # The dictionaries we have to fill.
    predictions = OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, (r_idx, example) in enumerate(tqdm(examples.iterrows())):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features["offset_mapping"][feature_index]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits).tolist()[-1 : -n_best_size - 1 : -1]
            end_indexes = np.argsort(end_logits).tolist()[-1 : -n_best_size - 1 : -1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        predictions[r_idx] = best_answer["text"]

    return predictions

def main():
    args, args_text = _parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    if args.method == 'contrastive':
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    else:
        model = CustomizedBertForQuestionAnswering.from_pretrained(args.model_name)
    model = model.to(device)

    if args.valid_squad or args.valid_both:
        cached_valid_squad_file = f"{args.valid_squad_path}.pkl"
        if Path(cached_valid_squad_file).is_file():
            valid_squad_features = pickle.load(Path(cached_valid_squad_file).open('rb'))
            logger.info(f"Reading {len(valid_squad_features)} hub files")

        else:
            qa_data = read_file(args.valid_squad_path)
            valid_squad_features = convert_to_eval_features(qa_data, tokenizer, args.max_len, args.doc_stride)
            with open(cached_valid_squad_file, 'wb') as f:
                pickle.dump(valid_squad_features, f)

    if args.valid_hub or args.valid_both:
        if args.filter_source:
            logger.info(f"Filtering on {args.source_no}")
            cached_valid_hub_file = f"{args.valid_hub_path}_src{args.source_no}.pkl"
            if Path(cached_valid_hub_file).is_file():
                valid_hub_features = pickle.load(Path(cached_valid_hub_file).open('rb'))
                logger.info(f"Reading {len(valid_hub_features)} hub files")
            else:
                qa_data = read_file(args.valid_hub_path, is_source=True, source_no=args.source_no)
                valid_hub_features = convert_to_eval_features(qa_data, tokenizer, args.max_len, args.doc_stride)
                with open(cached_valid_hub_file, 'wb') as f:
                    pickle.dump(valid_hub_features, f)
        else:
            cached_valid_hub_file = f"{args.valid_hub_path}.pkl"
            if Path(cached_valid_hub_file).is_file():
                valid_hub_features = pickle.load(Path(cached_valid_hub_file).open('rb'))
                logger.info(f"Reading {len(valid_hub_features)} hub files")
            else:
                qa_data = read_file(args.valid_hub_path)
                valid_hub_features = convert_to_eval_features(qa_data, tokenizer, args.max_len, args.doc_stride)
                with open(cached_valid_hub_file, 'wb') as f:
                    pickle.dump(valid_hub_features, f)


if __name__=="__main__":
    main()