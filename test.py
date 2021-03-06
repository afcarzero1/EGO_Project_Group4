import os.path
from datetime import datetime

import torch.nn.parallel
import torch.optim
from utils.loaders import VideoDataset
from train import validate, get_model_spec
from utils.args import parser
import warnings  # Default Python package
import utils  # Own package
from pathlib import Path
import pandas as pd  # Python package for handling dataframes
import numpy as np  # Python package for handling matrixes
import os  # Default Python package

# Deactivate warnings
warnings.filterwarnings("ignore")

# Initialize weights for reproducing results
np.random.seed(13696641)
torch.manual_seed(13696641)

# Define models to test
N_MODELS = 9  # number of models to test
# Parse arguments from command line (described in the README)
args = parser.parse_args()

KD_from_flow = (args.egomo > 0) or \
               (args.egomo_cossim > 0) or \
               (args.egomo_feat_patch > 0) or \
               (args.egomo_w > 0)  # add here others KD version
if KD_from_flow:
    print("KD FROM FLOW IS ON")


# Main script to execute
def main():
    """
    This is the main function to be executed for the testing the models.
    """
    # Check the modality to test. This parameter can assume three values (RGB,Flow and Event) depending
    # on the modality to check
    modalities = args.modality

    # This is the path of the model to test
    assert args.resume_from is not None, 'You must specify the path of the experiment you want to test. Do it in' \
                                         '--resume_from argument. Example : --resume_from example/path/*'
    # Function for taking the path of the files. # In case of changing machine internal paths must be modified (check todo)
    utils.utils.take_path(args)
    source_domains, target_domains, _ = utils.utils.set_domain_shift(args)
    '''
        Verbs (the list of possible predicitons for the action recognition):
        0 - take (get)
        1 - put-down (put/place)
        2 - open
        3 - close
        4 - wash (clean)
        5 - cut
        6 - stir (mix)
        7 - pour
        Domains:
        D1 - P08
        D2 - P01
        D3 - P22
    '''

    num_class = 8
    # Select the device to run. Select cuda only if available.
    device = "cpu"#FIXME :torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GENERAL]Using {device} device")
    '''
            Network: feature extractor & domain discriminator
    '''
    arch = {m: args.base_arch[modalities.index(m)] for m in modalities} # List of backbones
    model = {m: args.model[modalities.index(m)] for m in modalities} # Model to be used (i3d or tsn)
    model_spec = {m: get_model_spec(model[m], args, m) for m in modalities}
    num_frames_per_clip_test = {m: model_spec[m]["num_frames_per_clip_test"] for m in modalities}
    dense_sampling_test = {m: args.dense_sampling_test[modalities.index(m)] for m in modalities}
    # Instantiation of the model. Instantiate one for each modality indicated in the arguments.
    model_template = {m: model_spec[m]["net"](num_class, model_spec[m]["segments_test"], m,
                                              base_model=arch[m],
                                              args=args,
                                              **model_spec[m]["kwargs"])
                      for m in modalities}
    image_tmpl, _, val_transform = utils.utils.data_preprocessing(model_template, modalities, args.flow_prefix, args)
    # Instantiate the data loader using the image template obtained and the validation transform.
    val_loader = torch.utils.data.DataLoader(
        VideoDataset(pd.read_pickle(args.val_list),
                     args.modality,
                     image_tmpl,
                     num_frames_per_clip=num_frames_per_clip_test,
                     dense_sampling=dense_sampling_test,
                     fixed_offset=True,
                     visual_path=args.visual_path,
                     flow_path=args.flow_pwc_path if args.pwc else args.flow_path,
                     event_path=args.event_path,
                     num_clips=args.num_clips_test,
                     mode='test',
                     transform=val_transform,
                     args=args),
        batch_size=args.batch_size // 4, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.resume_from is not None:
        path = args.resume_from
        if args.last:
            path = list(sorted(Path(args.resume_from).iterdir(),
                               key=lambda date: datetime.strptime(
                                   os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S")))[-1]
        else:
            path = Path(args.resume_from)
        print('\nRESTORE FROM \'{}\'\n'.format(path))
        # list all files in chronological order (1st is most recent, last is less recent)
        TASK_NAME = 'action-classifier' # todo: put again action-classifier
        # List of paths to the models to load (.pth files)
        saved_models : [Path]= [x
                        for x in
                        reversed(sorted(Path(path).iterdir(), key=lambda x: os.path.basename(x).split('.')[0][-1]))
                        if TASK_NAME in x.name and ('1000' not in x.name and '2000' not in x.name and
                                                    '3000' not in x.name and '4000' not in x.name and
                                                    #'5000' not in x.name and '6000' not in x.name and
                                                    '7000' not in x.name and '8000' not in x.name and
                                                    '9000' not in x.name)][:N_MODELS * len(modalities)] #todo : take 5000 out
        model_list = []
        print('Restoring ---> {}'.format(TASK_NAME))
        for i, model_path in enumerate(saved_models):
            if i % len(modalities) == 0:
                model = {}
            m : str = model_path.name.split('_')[-3].upper() #todo : remove upper ,put -2 again (why I changed this?, name of model seem wrong)
            if m == "Flow" and KD_from_flow:
                continue
            model_template = model_spec[m]["net"](num_class, model_spec[m]["segments_test"], m,
                                                  base_model=arch[m],
                                                  args=args,
                                                  **model_spec[m]["kwargs"])
            # Now we create the model template using DataParallel. I this way the application is parallelized at the
            # module level by splitting the input across the devices.
            # As explained in documentation, in froward pass the module is replicated in each device and in backward pass
            # gradients from each replica are summed up into the original model.
            # Read in the documentation the possible problems that come from this usage. In particular any update on the
            # running module during forward will be lost.
            model_template = torch.nn.DataParallel(model_template).to(device) # Create the model template
            model[m], _, _, _, _, _, _ = utils.utils.load_checkpoint(str(model_path), model_template, optimizer=None) # Load checkpoint
            print('[{}] Mode: {} OK!\tFile = \'{}\''.format(i // 2 + 1, m, model_path.name))
            if (i + 1) % len(modalities) == 0:
                model_list.append(model)
    print('Weights successfully restored!')

    print("---------- START TESTING -----------")

    log_filename = args.shift + '_' + '-'.join(modalities) + '-' + str(args.num_clips_test) + 'clips.txt'
    #Call to the validation function. Used for testing the model.
    test_results = validate(val_loader, None, model_list, device, args.num_clips_test)

    os.makedirs(os.path.join('TEST_RESULTS', args.name), exist_ok=True)
    with open(os.path.join('TEST_RESULTS', args.name, log_filename), 'w') as f:
        message = ("Testing Results:\n"
                   "Verb Prec@1\t{:.3f}%\nVerb Prec@5\t{:.3f}%\n"
                   "Prec@1\t{:.3f}%\nPrec@5\t{:.3f}%\n"
                   "Class accuracies\n{}"
                   ).format(test_results['verb_top1'],
                            test_results['verb_top5'],
                            test_results['top1'],
                            test_results['top5'],
                            test_results['class_accuracies']
                            )
        f.write(message)


if __name__ == "__main__":
    main()
