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

- By default all scripts make models train on 2021 LA eval set, but you can find more bash scripts
for runs I did and see how to run train on 2021 DF or 2019 LA (mostly, you just need to change paths : ) )

## SEMAA (AASIST Improvement from ASVspoof5) Architecture + Rawboost:

Code is in the `spoof5/Baseline-AASIST`

```
cd spoof5/Baseline-AASIST
```

Create env and install packages:

```
conda create -n semaa python=3.9
conda activate semaa
pip install -r requirements.fixed.txt
```

### Train on 2021 ASVspoof LA dataset:
- Path to database: LA or DF can be setup in model config, e.g.: `spoof5/Baseline-AASIST/config/SEMAA.conf`

- All scripts, that I used to run my experiments can be found in `spoof5/Baseline-AASIST/scripts`.
  

- Script example:
```
python ./main.py \
    --config ./config/SEMAA_2021.conf \
    --comment semma_rawboost \
    --use_rawboost \
    --algo 4 
```

### Create submission:
- Script can be seen in `spoof5/Baseline-AASIST/scripts/submit.sh`
- Example for submission creation:

```
ckpt_path=exp_result/SEMAA_2021_ep100_bs24_NO_AUG/weights/epoch_20_0.001.pth

python submit.py \
    --architecture SEMAA \
    --ckpt_path $ckpt_path \
    --config_path ./config/SEMAA_2021.conf \
    --device cuda \
    --output_file semaa2021_NO_AUG_20_ep.csv \
    # --need_sigmoid
```

## SSL + AASIST:

- This architecture supports two SSL models:
  - wav2vec-xlsr, initial pretrain I used is from [HERE](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr)
  - Distil Prune Hubert, initial pretrain I used is from [HERE](https://huggingface.co/pyf98/DPHuBERT/blob/main/DPHuBERT-sp0.75.pth)

- I also made DPHubert support not only last-layer feature usage of SSL transformer model,
but weighted-sum method, that was proposed in [paper](https://arxiv.org/abs/2210.01273).

Code is in the `SSL_Anti-spoofing`

```
cd SSL_Anti-spoofing
```

Create env and install packages:
```
conda create -n ssl_aasist python=3.7
conda activate ssl_aasist
pip install -r requirements.fixed.txt
```

### Train on ASVspoof:
- Every script I used is in the `SSL_Anti-spoofing/scripts`

Additional train parameters:

- you can change `--ssl_backbone` parameter to `dp_hubert` to train with DPHubert
  - For DPHubert you can also use weighted sum feature aggregation, you can it with `--ssl_behaviour` parameter, change it to `weighted-sum`

- you can freeze weights of SSL model, just pass `--freeze_ssl` parameter
- You can also use SEMAA architecture as classifier, not only AASIST, for it add argument: `--use_semaa`
- Example:
```
python ./main_SSL_LA.py \
    --batch_size 32 \
    --comment wav2vec_2021_DF \
    --algo 5 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021 \
    --weight_decay 3e-5
```

- little crutch by me here, to train on 2021 DF, make: `--train_year 2021_DF`, sorry for this one, did not have much time...

### Submit:

- Script can be found in `SSL_Anti-spoofing/scripts/submit.sh`
- Example:
```
ckpt_path="models/model_LA_weighted_CCE_100_24_1e-05_2021_wav2vec_unfreeze/last.pth"

python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_10_epochs_la.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer
```

## Rawformer architecture:

Code is in the `rawformer_asv_spoof`

```
cd rawformer_asv_spoof
```

- Initial repository required building docker image, so I did this and I recommend
you to do so to run training of this model

- To Run just the inference of Rawboost model it is enough to use conda env created for SEMAA

Build docker:
```
docker build -t rawformer .
```

Run docker:
```
DATA_PATH=/path/to/your/data

docker run -it --rm \
    --network=host --shm-size=10g \
    --gpus "all" \
    -p 8888:8888 \
    -v $PWD:/app \
    -v $DATA_PATH:/dataset \
    rawformer
```

### Train on ASVspoof:
- all the paths are set up here in `rawformer_asv_spoof/config.py` file and I made it again a bit crutchy to work with 2021, but it's ok...
  
- Example for train:
  - You can change rawboost augmentation strategy with `--algo`, but actually I could not train it with rawboost...
```
python ./main.py \
    --algo 0 \
    --comment 2021_train
```

### Submit:
```
ckpt_path=checkpoints/Rawformer-2021-Train-no-aug/2021_train_ep_19_rawboost_algo_0_allow_aug_False.pth

python submit.py \
    --architecture Rawformer \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file rawformer_19_ep_2021.csv
```

---

## Links to Related Projects:

### SEMAA:
- paper: [LINK](https://www.isca-archive.org/asvspoof_2024/xia24_asvspoof.pdf)
- code: [LINK](https://github.com/SherlockEvans/SEMAA-for-ASVspoof5)
  
### SSL + AASIST:
- paper: [LINK](https://arxiv.org/abs/2202.12233)
- code: [LINK](https://github.com/TakHemlata/SSL_Anti-spoofing)

### Rawformer:
- paper: [LINK](https://ieeexplore.ieee.org/document/10096278)
- code (unofficial implementation): [LINK](https://github.com/rst0070/Rawformer-implementation-anti-spoofing)