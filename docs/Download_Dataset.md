# Downloading the EuRoC Dataset

This guide explains how to download the EuRoC MAV dataset required for testing the stereo visual-inertial odometry system.

## EuRoC Dataset Overview

The EuRoC MAV (European Robotics Challenge Micro Aerial Vehicle) dataset contains stereo camera images, IMU data, and ground truth trajectory data from a micro aerial vehicle flying in indoor environments. It includes 11 sequences total with different difficulty levels.

## Download Options

### Option 1: Using the Provided Script (Recommended)

The repository includes a convenience script to download all EuRoC MAV datasets automatically.

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/of/dataset
```

This will download all 11 sequences into the specified directory structure:
```
/path/of/dataset/
├── EuRoC/
│   ├── MH_01_easy/
│   ├── MH_02_easy/
│   ├── MH_03_medium/
│   ├── MH_04_difficult/
│   ├── MH_05_difficult/
│   ├── V1_01_easy/
│   ├── V1_02_medium/
│   ├── V1_03_difficult/
│   ├── V2_01_easy/
│   ├── V2_02_medium/
│   └── V2_03_difficult/
```

### Option 2: Manual Download

You can manually download specific sequences from the [EuRoC dataset website](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

1. Visit the EuRoC dataset page
2. Download the desired sequences in ASL Dataset Format
3. Extract the files into the appropriate directory structure

## Dataset Structure

Each sequence contains the following files:
```
MH_01_easy/
├── mav0/
│   ├── cam0/           # Left camera images
│   │   ├── data/
│   │   └── data.csv
│   ├── cam1/           # Right camera images
│   │   ├── data/
│   │   └── data.csv
│   ├── imu0/           # IMU data
│   │   └── data.csv
│   ├── leica0/         # Ground truth poses
│   │   └── data.csv
│   └── state_groundtruth_estimate0/
│       └── data.csv
```

## Sequence Recommendations

## Storage Requirements

- **Single sequence**: ~1-3 GB
- **All sequences**: ~20-25 GB

Make sure you have sufficient disk space before downloading all sequences.

## Troubleshooting

### Download Issues
- Check your internet connection
- Ensure sufficient disk space
- Verify the target directory has write permissions

### Script Permissions
```bash
chmod +x script/download_euroc.sh
```

## Next Steps

After downloading the dataset:
1. Follow the [Installation Guide](Install.md) to build the project
2. See the [Running Examples](Running_Example.md) to test with the downloaded data
