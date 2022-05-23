import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Parser")

parser.add_argument("--verbose",'-v',action='store_true',
                    help='Increase verbosity')

parser.add_argument("--model",type=str,default="i3d",help="Model used as feature extractor")
parser.add_argument("--modality",type=str,default="Flow",help="Modality of model used as feature extractor")
parser.add_argument("--shift",type=str,default="D1-D1",help="The shift used for the feature extractor")
parser.add_argument("--batch_size",type=int,default=200,help="The batch size used for the training")
parser.add_argument("--weight_decay",type=float,default=1e-5,help="The weight decay used for the training")
parser.add_argument("--early",action="store_true",help="Early stopping for training training")
parser.add_argument("--epochs",type=int,default=600,help="The number of epochs used for the training")
parser.add_argument("--transpose_input",action="store_true",help="Defines if the output of the dataset must be transposed before going to the model. Necessary for TRM wh i3d for example")
parser.add_argument("--temporal_aggregator",type=str,default="AvgPooling")
parser.add_argument("--learning_rate",type=float,default=0.01,help="The learning rate used for the training")

