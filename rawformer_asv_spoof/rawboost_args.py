import argparse

def create_rawboost_args():
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")

    ##===================================================Rawboost data augmentation parameters======================================================================#

    parser.add_argument(
        "--algo",
        type=int,
        default=0,
        help="Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]",
    )

    # LnL_convolutive_noise parameters
    parser.add_argument(
        "--nBands",
        type=int,
        default=5,
        help="number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]",
    )
    parser.add_argument(
        "--minF",
        type=int,
        default=20,
        help="minimum centre frequency [Hz] of notch filter.[default=20] ",
    )
    parser.add_argument(
        "--maxF",
        type=int,
        default=8000,
        help="maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]",
    )
    parser.add_argument(
        "--minBW",
        type=int,
        default=100,
        help="minimum width [Hz] of filter.[default=100] ",
    )
    parser.add_argument(
        "--maxBW",
        type=int,
        default=1000,
        help="maximum width [Hz] of filter.[default=1000] ",
    )
    parser.add_argument(
        "--minCoeff",
        type=int,
        default=10,
        help="minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]",
    )
    parser.add_argument(
        "--maxCoeff",
        type=int,
        default=100,
        help="maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]",
    )
    parser.add_argument(
        "--minG",
        type=int,
        default=0,
        help="minimum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--maxG",
        type=int,
        default=0,
        help="maximum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--minBiasLinNonLin",
        type=int,
        default=5,
        help=" minimum gain difference between linear and non-linear components.[default=5]",
    )
    parser.add_argument(
        "--maxBiasLinNonLin",
        type=int,
        default=20,
        help=" maximum gain difference between linear and non-linear components.[default=20]",
    )
    parser.add_argument(
        "--N_f",
        type=int,
        default=5,
        help="order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]",
    )

    # ISD_additive_noise parameters
    parser.add_argument(
        "--P",
        type=int,
        default=10,
        help="Maximum number of uniformly distributed samples in [%].[defaul=10]",
    )
    parser.add_argument(
        "--g_sd", type=int, default=2, help="gain parameters > 0. [default=2]"
    )

    # SSI_additive_noise parameters
    parser.add_argument(
        "--SNRmin",
        type=int,
        default=10,
        help="Minimum SNR value for coloured additive noise.[defaul=10]",
    )
    parser.add_argument(
        "--SNRmax",
        type=int,
        default=40,
        help="Maximum SNR value for coloured additive noise.[defaul=40]",
    )

    args = parser.parse_args()
    return args
