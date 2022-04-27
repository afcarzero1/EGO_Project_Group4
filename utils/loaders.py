import math

import pandas as pd

from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import librosa
import os.path
import numpy as np
from numpy.random import randint
import pickle

class VideoDataset(data.Dataset):
    """
    Class for implementing the video dataset.

    Attributes
    ----------
    list_file : pd.DataFrame
        Pandas DataFrame with annotations for the video dataset
    num_clips : int
        Number of clips to analyze
    video_list : [EpicVideoRecord]
        It is a list of EpicVideoRecords. Each of the elements contains information about a clip such as the kitchen
        participant, start frame, etc...
    :todo finish the documentation

    """
    def __init__(self, list_file, modality, image_tmpl,
                 num_frames_per_clip, dense_sampling,
                 sample_offset=0, num_clips=1,
                 fixed_offset=False,
                 visual_path=None, flow_path=None, event_path=None,
                 mode='train', transform=None, args=None):
        """
        Constructor for video dataset class.
        """

        # Get list of user environment variables.
        self.load_cineca_data = os.environ["HOME"].split("/")[-1] == "abottin1"
        self.sync = args.sync #Synchronization parameter todo: understand where it must be used
        self.num_frames = num_frames_per_clip # Number of frames in a clip
        self.num_clips = num_clips
        self.resampling_rate= args.resampling_rate
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        self.audio_path = args.audio_path
        # self.audio_path = pickle.load(open(self.audio_path, 'rb')) # todo: uncomment this line

        self.modalities = modality  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.args = args
        self.factor = {"RGB": 1, "Spec":1, "Flow": 2, "Event": self.args.rgb4e}

        self.stride = {"RGB": 2, "Spec":1, "Flow": 1, "Event": 1}
        self.flow_path = flow_path
        self.visual_path = visual_path
        self.event_path = event_path
        self.list_file : pd.DataFrame = list_file  # all files paths taken from .pkl file

        # data related
        self.num_clips = num_clips
        self.image_tmpl = image_tmpl  # filename format. (ex. 'img_{:010d}.jpg' for RGB)
        self.transform = transform  # pipeline of transforms
        # List of EpicVideoRecord Instances
        self.video_list = [EpicVideoRecord(tup, self.args.rgb4e) for tup in self.list_file.iterrows()]

    def _sample_train(self, record, modality='RGB'):
        if self.dense_sampling[modality]:
            '''
            TO BE COMPLETED!
            '''

        else:
            '''
            TO BE COMPLETED!
            '''

        return indices

    def _get_train_indices(self, record):
        '''
        TO BE COMPLETED!
        '''
        return segment_indices

    def _get_val_indices(self, record, modality) -> [int]:
        r""" This is a private function for getting the indices for validation.
        Arguments:
            record (EpicVideoRecord) : This is the record for which we want to get the validation indices.
            modality (str) : This is the modality of the experiment.
        Returns:
             indices ([int]) : List of indices.
        FIXME : Complete this class. Now it does dummy stuff
        """
        indices: [] = []
        starting_frame: int = record.start_frame
        end_frame: int = record.end_frame
        num_frames: int = self.num_frames[modality] # Rettrieve the number of frames to be sampled from the given modality.
        dense_sampling: bool = self.dense_sampling # Check whether sampling is dense or not
        if modality == 'RGB':
            # Check if the sampling is dense
            if dense_sampling:
                # In case of dense sampling take a random point in the clip and take num_frames consecutive frames.
                indices_start = np.random.randint(starting_frame,end_frame-num_frames)
                indices = [idx for idx in range(indices_start,indices_start+num_frames)]
            else:
                # In case of not dense sampling take uniform distributed frames from the clip.
                indices = np.random.randint(starting_frame,end_frame,num_frames).tolist()
        elif modality == 'Flow':
            #todo : write the code for this
            indices = None

        return indices

    def __getitem__(self, index):
        """
        This is the function to implement. It gives an item from the dataset
        :param index This is the index from which we want to get the data.
        """
        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        record = self.video_list[index]

        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)

        else:  # val or test case
            segment_indices = {}
            for m in self.modalities:
                segment_indices[m] = self._get_val_indices(record, m)


        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        return frames, label

    '''
                Audio-related
    '''

    def _log_specgram(self, audio, window_size=10, step_size=5, eps=1e-6):
        '''
        TO BE COMPLETED!
        '''

    def _extract_sound_feature(self, record, idx):
        '''
        TO BE COMPLETED!
        '''

        return self._log_specgram(samples)


    def get(self, modality, record, indices):
        r""" Function for getting a set of records.
        Arguments:
            modality (str) : It is the modality used in the loader. It may be RGB or Flow or Event.
            record (EpicVideoRecord) : It is the EpicVideorecord
            indices ([int]) : List of indices of the videos to get FIXME: Is this correct?
        Returns:
            FIXME : What does this return?
        """
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            frame = self._load_data(modality, record, p)
            images.extend(frame)

        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        r""" Private function for loading the data
        Arguments:
            modality (str) : It is the modality of the experiment
            record (EpicVideoRecord) : The number of the record to retrieve
            idx (int) : The index of the frame # FIXME : Verify this is really the index of the frame

        """

        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame + idx
            img = Image.open(os.path.join(self.visual_path, record.untrimmed_video_name,
                                          self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')
            return [img]
        elif modality == 'Flow':
            idx_untrimmed = (record.start_frame // 2) + idx
            #print(idx_untrimmed)
            try:
                x_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                self.image_tmpl[modality].format('x', idx_untrimmed))).convert('L')
                y_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                self.image_tmpl[modality].format('y', idx_untrimmed))).convert('L')
            except FileNotFoundError:
                for i in range(0, 3):
                    found = True
                    try:
                        x_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                        self.image_tmpl[modality].format('x',
                                                                                         idx_untrimmed + i))).convert(
                            'L')
                        y_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                        self.image_tmpl[modality].format('y',
                                                                                         idx_untrimmed + i))).convert(
                            'L')
                    except FileNotFoundError:
                        found = False

                    if found:
                        break
            return [x_img, y_img]
        elif modality == 'Event':
            idx_untrimmed = (record.start_frame // self.args.rgb4e) + idx

            try:
                img_npy = np.load(os.path.join(self.event_path, record.untrimmed_video_name,
                                               self.image_tmpl[modality].format(idx_untrimmed))).astype(
                    np.float32)
            except ValueError:
                img_npy = np.load(os.path.join(self.event_path, record.untrimmed_video_name,
                                               self.image_tmpl[modality].format(
                                                   record.num_frames["Event"]))).astype(
                    np.float32)
            return np.stack([img_npy], axis=0)
        else:
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]

    def __len__(self):
        return len(self.video_list)
