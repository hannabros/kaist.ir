
import os, sys
import yaml
from pathlib import Path
import pickle
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from configMgr import config_parser, parser
from customBertModel import CustomizedBertForQuestionAnswering
from customDataset import read_file, convert_to_features
from utils import WarmupLinearSchedule, save_model
from collate import collate_wrapper

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

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

def train_log(loss, example_ct, epoch, lr):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss, "lr": lr}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}" + f" with lr: {lr}")

def train_batch(step, batch, model, optimizer, scheduler, device, args):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    start_positions = batch.start_positions.to(device)
    end_positions = batch.end_positions.to(device)
    if args.method == 'contrastive':
        input_type = batch.input_type.to(device)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions,
                        input_type=input_type)
    else:
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
    loss = outputs[0]

    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    loss.backward()

    if (step + 1) % args.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return loss, lr

def validation(valid_loader, model, device):
    model.eval()
    with torch.no_grad():
        val_losses, acc = [], []
        valid_example_ct = 0
        for _, batch in enumerate(valid_loader):
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
            end_pred = torch.argmax(outputs[2], dim=1)
            acc.append(((start_pred == start_positions).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_positions).sum()/len(end_pred)).item())
        avg_loss = sum(val_losses)/len(val_losses) 
        avg_acc = sum(acc)/len(acc)

        print(f"Accuracy of the model on the {valid_example_ct} " +
            f"test samples: {100 * avg_acc}% with valid loss: {avg_loss}")
        
        wandb.log({"valid_loss": avg_loss, "accuracy": avg_acc})
    return avg_loss, avg_acc

def main():
    args, args_text = _parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            logger.warning("You've requested to log metrics to wandb but package not found. "
                           "Metrics not being logged to wandb, try `pip install wandb`")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    
    # Prepare Train Dataset
    logger.info("Preparing train dataset ...")
    if args.train_squad or args.train_both:
        cached_train_squad_file = f"{args.train_squad_path}.pkl"
        if Path(cached_train_squad_file).is_file():
            train_squad_features = pickle.load(Path(cached_train_squad_file).open('rb'))
            logger.info(f"Reading {len(train_squad_features)} hub files")

        else:
            qa_data = read_file(args.train_squad_path)
            train_squad_features = convert_to_features(
                qa_data, tokenizer, args.max_len, args.doc_stride)
            with open(cached_train_squad_file, 'wb') as f:
                pickle.dump(train_squad_features, f)

    if args.train_hub or args.train_both:
        if args.filter_source:
            logger.info(f"Filtering on {args.source_no}")
            cached_train_hub_file = f"{args.train_hub_path}_src{args.source_no}.pkl"
            if Path(cached_train_hub_file).is_file():
                train_hub_features = pickle.load(Path(cached_train_hub_file).open('rb'))
                logger.info(f"Reading {len(train_hub_features)} hub files")
            else:
                qa_data = read_file(args.train_hub_path, is_source=True, source_no=args.source_no)
                train_hub_features = convert_to_features(qa_data, tokenizer, args.max_len, args.doc_stride)
                with open(cached_train_hub_file, 'wb') as f:
                    pickle.dump(train_hub_features, f)
        else:
            cached_train_hub_file = f"{args.train_hub_path}.pkl"
            if Path(cached_train_hub_file).is_file():
                train_hub_features = pickle.load(Path(cached_train_hub_file).open('rb'))
                logger.info(f"Reading {len(train_hub_features)} hub files")
            else:
                qa_data = read_file(args.train_hub_path)
                train_hub_features = convert_to_features(qa_data, tokenizer, args.max_len, args.doc_stride)
                with open(cached_train_hub_file, 'wb') as f:
                    pickle.dump(train_hub_features, f)

    if args.train_both:
        train_features = train_squad_features + train_hub_features
        torch_input_ids = [torch.tensor(f.input_ids, dtype=torch.long) for f in train_features]
        all_input_ids = torch.cat([ii for ii in torch_input_ids], dim=0)
        torch_token_type_ids = [torch.tensor(f.token_type_ids, dtype=torch.long) for f in train_features]
        all_token_type_ids = torch.cat([tti for tti in torch_token_type_ids], dim=0)
        torch_attention_mask = [torch.tensor(f.attention_mask, dtype=torch.long) for f in train_features]
        all_attention_mask = torch.cat([am for am in torch_attention_mask], dim=0)
        torch_start_positions = [torch.tensor(f.start_positions, dtype=torch.long) for f in train_features]
        all_start_positions = torch.cat([sp for sp in torch_start_positions], dim=0)
        torch_end_positions = [torch.tensor(f.end_positions, dtype=torch.long) for f in train_features]
        all_end_positions = torch.cat([ep for ep in torch_end_positions], dim=0)
        squad_size = sum([len(f.input_ids) for f in train_squad_features])
        hub_size = sum([len(f.input_ids) for f in train_hub_features])
        input_type = torch.tensor([0] * squad_size + [1] * hub_size, dtype=torch.long)
        train_dataset = TensorDataset(all_input_ids,
                                      all_token_type_ids,
                                      all_attention_mask,
                                      all_start_positions,
                                      all_end_positions,
                                      input_type)
    else:
        train_features = train_squad_features
        torch_input_ids = [torch.tensor(f.input_ids, dtype=torch.long) for f in train_features]
        all_input_ids = torch.cat([ii for ii in torch_input_ids], dim=0)
        torch_token_type_ids = [torch.tensor(f.token_type_ids, dtype=torch.long) for f in train_features]
        all_token_type_ids = torch.cat([tti for tti in torch_token_type_ids], dim=0)
        torch_attention_mask = [torch.tensor(f.attention_mask, dtype=torch.long) for f in train_features]
        all_attention_mask = torch.cat([am for am in torch_attention_mask], dim=0)
        torch_start_positions = [torch.tensor(f.start_positions, dtype=torch.long) for f in train_features]
        all_start_positions = torch.cat([sp for sp in torch_start_positions], dim=0)
        torch_end_positions = [torch.tensor(f.end_positions, dtype=torch.long) for f in train_features]
        all_end_positions = torch.cat([ep for ep in torch_end_positions], dim=0)
        train_dataset = TensorDataset(all_input_ids,
                                      all_token_type_ids,
                                      all_attention_mask,
                                      all_start_positions,
                                      all_end_positions)
    
    # Prepare Valid Dataset
    logger.info("Preparing valid dataset ...")
    cached_valid_file = f"{args.valid_path}.pkl"
    if Path(cached_valid_file).is_file():
        valid_features = pickle.load(Path(cached_valid_file).open('rb'))
        logger.info(f"Reading {len(valid_features)} valid files")
    else:
        qa_data = read_file(args.valid_path)
        valid_features = convert_to_features(qa_data, tokenizer, args.max_len, args.doc_stride)
        with open(cached_valid_file, 'wb') as f:
            pickle.dump(valid_features, f)
    # valid_features = valid_features
    torch_input_ids = [torch.tensor(f.input_ids, dtype=torch.long) for f in valid_features]
    valid_input_ids = torch.cat([ii for ii in torch_input_ids], dim=0)
    torch_token_type_ids = [torch.tensor(f.token_type_ids, dtype=torch.long) for f in valid_features]
    valid_token_type_ids = torch.cat([tti for tti in torch_token_type_ids], dim=0)
    torch_attention_mask = [torch.tensor(f.attention_mask, dtype=torch.long) for f in valid_features]
    valid_attention_mask = torch.cat([am for am in torch_attention_mask], dim=0)
    torch_start_positions = [torch.tensor(f.start_positions, dtype=torch.long) for f in valid_features]
    valid_start_positions = torch.cat([sp for sp in torch_start_positions], dim=0)
    torch_end_positions = [torch.tensor(f.end_positions, dtype=torch.long) for f in valid_features]
    valid_end_positions = torch.cat([ep for ep in torch_end_positions], dim=0)
    valid_dataset = TensorDataset(valid_input_ids,
                                  valid_token_type_ids,
                                  valid_attention_mask,
                                  valid_start_positions,
                                  valid_end_positions)
    
    # Prepare Train & Valid Loader
    logger.info("Preparing train & valid loader ...")
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              collate_fn=collate_wrapper)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              collate_fn=collate_wrapper)
    
    if args.train_both:
        model = CustomizedBertForQuestionAnswering.from_pretrained(args.model_name)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    model = model.to(device)

    # Train
    criterion = nn.CrossEntropyLoss()
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=t_total*args.warmup_proportion, t_total=t_total)

    with wandb.init(project="qa_contrastive", config=args):
        wandb.watch(model, criterion, log="all", log_freq=10)

        example_ct = 0  # number of examples seen
        batch_ct = 0
        best_acc, best_loss = 0, 10.0
        
        for epoch in tqdm(range(args.epochs)):
            for step, batch in enumerate(train_loader):
                loss, lr = train_batch(step, batch, model, optimizer, scheduler, device, args)
                example_ct +=  batch.input_ids.shape[0]
                batch_ct += 1

                if ((batch_ct + 1) % args.log_interval) == 0:
                    train_log(loss, example_ct, epoch, lr)

            avg_loss, avg_acc = validation(valid_loader, model, device)
            
            # Save the model
            if args.metric == 'loss':
                if best_loss > avg_loss:
                    best_loss = avg_loss
                    logger.info(f'best loss changed to {best_loss}')
                    save_model(model, tokenizer, args.model_path)
            elif args.metric == 'accuracy':
                if best_acc < avg_acc:
                    best_acc = avg_acc
                    logger.info(f'best accuracy changed to {best_acc}')
                    save_model(model, tokenizer, args.model_path)
                
            model.train()


if __name__ == "__main__":
    main()
