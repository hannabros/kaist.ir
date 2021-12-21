import argparse

COLOR_LABEL_MAPPING = {
    'default': '0.5',
    'question': 'cyan',
    'context': 'orange',
    'answer': 'red'
}

MARKER_LABEL_MAPPING = {
    'default': 'o',
    'question': 'o',
    'context': 'o',
    'answer': 'd'
}

config_parser = parser = argparse.ArgumentParser(description='Configuration', add_help=False)

parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='YAML Configuration')

# Train & Path
parser.add_argument("--train_both", action='store_true', help="Whether to train on both train data")
parser.add_argument("--train_squad", action='store_true', help="Whether to only train on squad data")
parser.add_argument("--train_hub", action='store_true', help="Whether to only train on squad data")
parser.add_argument("--train_squad_path", type=str, default=None, help="Path of squad file to train")
parser.add_argument("--train_hub_path", type=str, default=None, help="Path of hub file to train")
parser.add_argument("--valid_path", type=str, default=None, help="Path of file to valid. Usually squad dev file.")
parser.add_argument("--model_path", type=str, default=None, help="Path to save model")

# Preprocessing
parser.add_argument('--filter_source', action='store_true', help='Filter source')
parser.add_argument("--source_no", type=int, default=1, help="Number of source")

# Epoch/Batch/LR
parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")

# Hypterparameters
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The maximum norm for backward gradients.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
parser.add_argument('--beta', type=float, default=0.01, help="Scale for contrastive loss")
parser.add_argument('--sigma', type=float, default=0.01, help="Noise scale for feature smoothing")
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

# Validation
parser.add_argument("--metric", type=str, default="loss", help="Metric to save model")

# Tokenizer & Model
parser.add_argument("--method", type=str, default='ce', help='Method to train model (contrastive | ce)')
parser.add_argument("--model_name", type=str, default='klue/bert-base', help='Name or path of pretrained model')
parser.add_argument("--do_lower_case", action='store_true', help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument('--max_len', type=int, default=384, help="Maximum length")
parser.add_argument('--doc_stride', type=int, default=128, help="Length of doc stride")

# Wandb & Logging
parser.add_argument('--log_wandb', action='store_true', help='log training and validation metrics to wandb')
parser.add_argument('--experiment', default='', type=str, metavar='NAME', help='name of train experiment, name of sub-folder for output')
parser.add_argument('--log_interval', type=int, default=10, help='Log train loss every n steps')


