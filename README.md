# Safe-Speak Hackathon Solutions

## Solution Explanation

I used 3 main approaches for my solution:

- SEMAA architecture (AASIST improvement presented on ASVspoof2024 workshop) + Rawboost Augmentation
- SSL feature extractor + AASIST + Rawboost augmentation
  - SSL: wav2vec-XLSR
  - SSL: DPHubert: Distilled Hubert model
  - SSL: DPHubert with weighted sum feature aggregation
- Rawformer architecture
  
Number of parameters:
- SEMAA: 341k
- wav2vec + AASIST: ~300M
- DPHubert + AASIST: ~23M
- Rawformer: 297k

Achieved Metrics:

|       Model       | Augmentation | EER on test |
|:-----------------:|:------------:|-------------|
|       SEMAA       |   Rawboost   |             |
|       SEMAA       |    NO Aug    |             |
|  wav2vec + AASIST |   Rawboost   |             |
| DPHubert + AASIST |   Rawboost   |             |
|     Rawformer     |    NO Aug    |             |

- Note that Rawboost augmentation is EXTREMELY slow to train with: it slowed SEMAA training up to 4 times, that's why model without augmentation was able to train better.

## Loading Data:

All data, that I used can be found [HERE](https://www.asvspoof.org/index2021.html).

Load data and keys and organize into folders, so the file structure looks like this: (my database folder was named `asvspoof`)

```
/path/to/asvspoof/
└── ASVspoof2021_LA_eval/
    └── keys/
    └── flac/
└── ASVspoof2021_DF_eval/
    └── flac/
└── 2019_LA/
    └── ASVspoof2019_LA_cm_protocols
    └── ASVspoof2019_LA_train\
        └── flac/
    ...
    ...
```

- In `2019_LA` is all the data for train, dev and eval for ASVspoof2019 data. 
- Note, that all keys for 2021 LA and DF are kept inside ASVspoof2021_LA_eval, even for DF subset!

### What data I used to get my best scores:

Firstly I used ASVspoof2021 LA validation subset to train my models.
Then I also tried using ASVspoof2021 DF validation dataset, because it has a lot more attacks and is 3 times bigger.

Of course, you can also run my code to train with 2019 ASVspoof dataset as it was initially suggested for 2021 challenge,
you will just need to change some parameters for your runs.

## Experiments Reproduction

Clone repo:

```
git clone https://github.com/Alphonsce/AntiSpoof-Hackathon.git
```

Whole project is divided into 3 main parts for each of the archutectures provided:
- `spoof5/Baseline-AASIST` for SEMAA model
- `SSL_Anti-spoofing` for SSL + AASIST
- `rawformer_asv_spoof` for Rawformer

## SEMAA (AASIST Improvement from ASVspoof5) Architecture + Rawboost:

Code is in the `spoof5/Baseline-AASIST`

```
cd spoof5/Baseline-AASIST
```

Create env and install packages:

```
conda create -n aasist
```

```
pip install -r 
```

## SSL + AASIST:

Code is in the `SSL_Anti-spoofing`

```
cd SSL_Anti-spoofing
```

## Rawformer architecture:

Code is in the `rawformer_asv_spoof`

```
cd spoof5/Baseline-AASIST
```

Install packages:
```

```