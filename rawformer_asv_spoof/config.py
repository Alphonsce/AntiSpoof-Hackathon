class SysConfig:
    """ """

    def __init__(self):

        self.wandb_disabled = False
        self.wandb_project = "ASVSpoof-Rawformer"
        self.wandb_name = "Rawformer-2021-Train-no-aug"
        self.wandb_entity = "jurujin"
        self.wandb_key = "b117cc2bdbcbc127dc0a49d6d94cc6f49a6ef821"
        self.wandb_notes = "lr=8*1e-4, ts_hidden=660, rand_seed=1024, pre-emphasis=0.97"

        # self.path_label_asv_spoof_2019_la_train     = '/dataset/2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
        # self.path_asv_spoof_2019_la_train           = '/dataset/2019_LA/ASVspoof2019_LA_train/flac'
        self.path_label_asv_spoof_2019_la_train = (
            "/dataset/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
        )
        self.path_asv_spoof_2019_la_train = "/dataset/ASVspoof2021_LA_eval/flac"

        self.path_label_asv_spoof_2019_la_dev = "/dataset/2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        self.path_asv_spoof_2019_la_dev = "/dataset/2019_LA/ASVspoof2019_LA_dev/flac"

        self.path_label_asv_spoof_2021_la_eval = (
            "/dataset/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
        )
        self.path_asv_spoof_2021_la_eval = "/dataset/ASVspoof2021_LA_eval/flac"

        self.num_workers = 4

        self.ckpt_save_dir = f"./checkpoints/{self.wandb_name}/"
        self.ckpt_load_path = None


class ExpConfig:

    def __init__(self):
        self.eval_every_n_epochs = 1

        self.random_seed = 1024

        self.pre_emphasis = 0.97

        self.sample_rate = 16000
        self.train_duration_sec = 4
        self.test_duration_sec = 4

        self.batch_size_train = 32
        self.batch_size_test = 32
        self.embedding_size = 64
        self.max_epoch = 300

        self.lr = 1 * 1e-4
        self.lr_min = (
            1e-6  # this could not work because i turned off scheduler in some cases
        )

        self.transformer_hidden = 660

        self.allow_data_augmentation = False
        # self.data_augmentation          = ['ACN', 'HPF', 'LPF', 'GAN'] # Augmentations besides RawBoost
        self.data_augmentation = ["ACN"]
