from pathlib import Path
import argparse

"""
This is a script for transforming EPIC dataset into symlinks format.
"""

if __name__ == '__main__':
    # Instantiate parser and read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path, help='Directory of epic dataset')
    parser.add_argument('symlinks_dir', type=Path, help='Directory to save symlinks for EPIC')

    print("[GENERAL] Starting application")


    args = parser.parse_args()
    # Case in which argument for symlinks folder does not exist.
    if not args.symlinks_dir.exists():
        args.symlinks_dir.mkdir(parents=True)
    # Take directory for data
    data_dir: Path = args.data_dir
    print("[INFO] Parent directory:", data_dir)

    # Match any directory with the name P??. '?' is any character in that position. '*'
    # matches zero or more characters in a segment of a name.
    # Remember glob uses standard unix expansion rules (not regexp)
    participant_pattern = r'P??'
    # Transform the structure of the directory for all participants. Function .glob iterates over the subtree that
    # matches the participant_pattern

    # An example of the directory structure is:
    # └── P08
    # ├── flow_frames
    # │        ├── P08_09
    # │        │       ├── u
    # │        │       └── v
    # │        ├── P08_10
    # │        │       ├── u
    # │        │       └── v
    # │        ├── P08_14
    # │        │       ├── u
    # │        │       └── v
    # │        ├── P08_15
    # │        │       ├── u
    # │        │       └── v
    # │        ├── P08_16
    # │        │       ├── u
    # │        │       └── v
    # │        └── P08_17
    # │            ├── u
    # │            └── v
    # └── rgb_frames
    # ├── P08_09
    # ├── P08_10
    # ├── P08_14
    # ├── P08_15
    # ├── P08_16
    # └── P08_17

for participant_dir in data_dir.glob(participant_pattern):
        # Inside each directory we require to have directories for rgb frames and flow frames.
        print("[GENERAL] Participant directory:", participant_dir)
        for modality in ['rgb_frames', 'flow_frames']:
            if modality == 'rgb_frames':
                video_id_pattern = 'P??_*??/'
            else:
                video_id_pattern = 'P??_*??/*/'

            frames_dir = participant_dir / modality
            print("[GENERAL] Frames directory:", frames_dir)
            # Search inside the frames' directory for folders with the corresponding pattern
            for source_file in frames_dir.glob(video_id_pattern):
                # Take the rgb frames
                if modality == 'rgb_frames':
                    video = str(source_file).split('/')[-1] # modification: take single element of list
                else:
                    video, _ = str(source_file).split('/')[-2:]
                print(video)
                link_path = args.symlinks_dir / video

                print(f"[OUTPUT] Links path is {link_path}")
                if not link_path.exists():
                    link_path.mkdir(parents=True)

                for i, _ in enumerate(source_file.iterdir()):
                    # Define name of frame in output directory
                    f = 'frame_{:010d}.jpg'.format(i + 1)
                    source = source_file / f

                    if modality == 'rgb_frames':
                        link = link_path / 'img_{:010d}.jpg'.format(i)
                    else:
                        # Case of flow frames
                        if source_file.name == 'u':
                            link = link_path / 'x_{:010d}.jpg'.format(i)

                        else:
                            # Case of file v
                            link = link_path / 'y_{:010d}.jpg'.format(i)

                    if link.exists():
                        link.unlink()
                    link.symlink_to(source)
