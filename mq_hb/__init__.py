from mq_hb.mq_random_search import mqRandomSearch
from mq_hb.mq_bo import mqBO
from mq_hb.mq_sh import mqSuccessiveHalving
from mq_hb.mq_hb import mqHyperband
from mq_hb.mq_bohb import mqBOHB
from mq_hb.mq_bohb_v0 import mqBOHB_v0
from mq_hb.mq_bohb_v2 import mqBOHB_v2
from mq_hb.mq_mfes import mqMFES
from mq_hb.mq_mfes_v4 import mqMFES_v4
from mq_hb.async_mq_random import async_mqRandomSearch
from mq_hb.async_mq_bo import async_mqBO
from mq_hb.async_mq_ea import async_mqEA
from mq_hb.async_mq_sh import async_mqSuccessiveHalving
from mq_hb.async_mq_sh_v0 import async_mqSuccessiveHalving_v0
from mq_hb.async_mq_sh_v2 import async_mqSuccessiveHalving_v2
from mq_hb.async_mq_sh_v3 import async_mqSuccessiveHalving_v3
from mq_hb.async_mq_hb import async_mqHyperband
from mq_hb.async_mq_hb_v0 import async_mqHyperband_v0
from mq_hb.async_mq_hb_v2 import async_mqHyperband_v2
from mq_hb.async_mq_hb_v3 import async_mqHyperband_v3
from mq_hb.async_mq_bohb import async_mqBOHB
from mq_hb.async_mq_bohb_v0 import async_mqBOHB_v0
from mq_hb.async_mq_bohb_v2 import async_mqBOHB_v2
from mq_hb.async_mq_bohb_v3 import async_mqBOHB_v3
from mq_hb.async_mq_bosh import async_mqBOSH
from mq_hb.async_mq_mfes import async_mqMFES
from mq_hb.async_mq_mfes_v6 import async_mqMFES_v6
from mq_hb.async_mq_mfes_v12 import async_mqMFES_v12
from mq_hb.async_mq_mfes_v13 import async_mqMFES_v13
from mq_hb.async_mq_mfes_v14 import async_mqMFES_v14
from mq_hb.async_mq_mfes_v15 import async_mqMFES_v15
from mq_hb.async_mq_mfes_v16 import async_mqMFES_v16
from mq_hb.async_mq_mfes_s import async_mqMFES_s

from mq_hb.async_mq_sh_stopping import async_mqSuccessiveHalving_stopping
from mq_hb.async_mq_hb_stopping import async_mqHyperband_stopping
from mq_hb.async_mq_mfes_stopping import async_mqMFES_stopping
from mq_hb.async_mq_median_stopping import async_mqMedianStopping
from mq_hb.async_mq_median_stopping_mfes import async_mqMFES_MedianStopping
from mq_hb.async_mq_median_stopping_mfgp import async_mqMFGP_MedianStopping

from mq_hb.mq_mfes_v5 import mqMFES_v5
from mq_hb.mq_mfes_v6 import mqMFES_v6
from mq_hb.mq_mfes_v7 import mqMFES_v7
from mq_hb.mq_mfes_v8 import mqMFES_v8

# stopping variant (reporter)
stopping_mths = [
    async_mqSuccessiveHalving_stopping,
    async_mqHyperband_stopping,
    async_mqMFES_stopping,
    async_mqMedianStopping,
    async_mqMFES_MedianStopping,
    async_mqMFGP_MedianStopping,
]

mth_dict = dict(
    random=(mqRandomSearch, 'sync'),
    bo=(mqBO, 'sync'),
    sh=(mqSuccessiveHalving, 'sync'),
    hyperband=(mqHyperband, 'sync'),
    bohb=(mqBOHB, 'sync'),
    bohbv0=(mqBOHB_v0, 'sync'),
    bohbv2=(mqBOHB_v2, 'sync'),  # tpe
    mfes=(mqMFES, 'sync'),
    mfesv4=(mqMFES_v4, 'sync'),
    arandom=(async_mqRandomSearch, 'async'),
    abo=(async_mqBO, 'async'),
    aea=(async_mqEA, 'async'),  # Asynchronous Evolutionary Algorithm
    aeav2=(async_mqEA, 'async', dict(strategy='oldest')),  # Asynchronous Evolutionary Algorithm
    asha=(async_mqSuccessiveHalving, 'async'),       # delayed asha
    ashav0=(async_mqSuccessiveHalving_v0, 'async'),  # origin asha
    ashav2=(async_mqSuccessiveHalving_v2, 'async'),  # promotion cycle
    ahb=(async_mqHyperband, 'async'),       # hb + delayed asha
    ahbv0=(async_mqHyperband_v0, 'async'),  # hb + origin asha
    ahbv2=(async_mqHyperband_v2, 'async'),  # hb + promotion cycle
    abohb=(async_mqBOHB, 'async'),  # prf
    abohbv0=(async_mqBOHB_v0, 'async'),  # origin asha, prf
    abohbv2=(async_mqBOHB_v2, 'async'),  # tpe
    abohbv3=(async_mqBOHB_v3, 'async'),  # amazon multi-fidelity gp
    abosh=(async_mqBOSH, 'async'),
    amfesv6=(async_mqMFES_v6, 'async'),
    amfesv12=(async_mqMFES_v12, 'async'),
    amfesv13=(async_mqMFES_v13, 'async'),
    amfesv14=(async_mqMFES_v14, 'async'),
    amfesv15=(async_mqMFES_v15, 'async'),
    amfesv16=(async_mqMFES_v16, 'async'),

    # stopping variant
    ashav3=(async_mqSuccessiveHalving_v3, 'async'),  # stopping variant asha
    ahbv3=(async_mqHyperband_v3, 'async'),  # hb + stopping variant asha
    amfess1=(async_mqMFES_s, 'async', dict(use_weight_init=True,    # from v20
                                           weight_init_choosing='proportional',
                                           weight_method='rank_loss_p_norm',
                                           non_decreasing_weight=False,
                                           increasing_weight=True, )),
    amfess2=(async_mqMFES_s, 'async', dict(use_weight_init=False,   # from v19
                                           weight_method='rank_loss_p_norm',
                                           non_decreasing_weight=False,
                                           increasing_weight=True, )),

    # stopping variant (reporter)
    asha_stop=(async_mqSuccessiveHalving_stopping, 'async'),  # stopping variant asha
    ahb_stop=(async_mqHyperband_stopping, 'async'),  # hb + stopping variant asha
    amfesv20_stop=(async_mqMFES_stopping, 'async', dict(use_weight_init=True,    # from v20
                                                        weight_init_choosing='proportional',
                                                        weight_method='rank_loss_p_norm',
                                                        non_decreasing_weight=False,
                                                        increasing_weight=True, )),
    amfesv19_stop=(async_mqMFES_stopping, 'async', dict(use_weight_init=False,   # from v19
                                                        weight_method='rank_loss_p_norm',
                                                        non_decreasing_weight=False,
                                                        increasing_weight=True, )),
    # median stopping (reporter)
    ams=(async_mqMedianStopping, 'async', dict()),
    amfes_ms=(async_mqMFES_MedianStopping, 'async', dict(weight_method='rank_loss_p_norm',
                                                         non_decreasing_weight=False,
                                                         increasing_weight=True, )),
    amfgp_ms=(async_mqMFGP_MedianStopping, 'async', dict()),
    amfgpbt_ms=(async_mqMFGP_MedianStopping, 'async', dict(use_botorch_gp=True)),

    # exp version:
    mfesv5=(mqMFES_v5, 'sync'),
    mfesv6=(mqMFES_v6, 'sync'),
    mfesv7=(mqMFES_v7, 'sync'),
    mfesv8=(mqMFES_v8, 'sync'),
    amfesv32=(async_mqMFES, 'async', dict(use_weight_init=False,
                                          weight_method='rank_loss_prob',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfesv35=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='pow',
                                          weight_method='rank_loss_prob',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfese1=(async_mqMFES, 'async', dict(use_weight_init=True,
                                         weight_init_choosing='proportional',
                                         weight_method='rank_loss_p_norm',
                                         non_decreasing_weight=False,
                                         increasing_weight=True,
                                         test_random=True,
                                         test_original_asha=True, )),  # ahb with bracket selection
    amfese2=(async_mqMFES, 'async', dict(use_weight_init=True,
                                         weight_init_choosing='proportional',
                                         weight_method='rank_loss_p_norm',
                                         non_decreasing_weight=False,
                                         increasing_weight=True,
                                         test_bohb=True,
                                         acq_optimizer='random', )),  # abohb with bracket selection
    amfessh=(async_mqMFES, 'async', dict(use_weight_init=False,
                                         weight_method='rank_loss_p_norm',
                                         non_decreasing_weight=False,
                                         increasing_weight=True,
                                         test_sh=True, )),  # amfes + sh
    amfesv18=(async_mqMFES, 'async', dict(set_promotion_threshold=False,
                                          use_weight_init=True,
                                          weight_init_choosing='proportional',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfesv19=(async_mqMFES, 'async', dict(use_weight_init=False,
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfesv20=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='proportional',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfesv21=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='argmax',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfesv22=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='argmax2',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True, )),
    amfesgpv1=(async_mqMFES, 'async', dict(use_weight_init=True,
                                           weight_init_choosing='proportional',
                                           weight_method='rank_loss_p_norm',
                                           non_decreasing_weight=False,
                                           increasing_weight=True,
                                           surrogate_type='gp', )),
    amfesgpv2=(async_mqMFES, 'async', dict(use_weight_init=True,
                                           weight_init_choosing='argmax2',
                                           weight_method='rank_loss_p_norm',
                                           non_decreasing_weight=False,
                                           increasing_weight=True,
                                           surrogate_type='gp', )),
    amfesv25=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='proportional',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True,
                                          test_original_asha=True, )),    # test original asha + mfes
    amfesv26=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='proportional',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True,
                                          rand_prob=0.1, )),    # test rand prob 0.1
    amfesv27=(async_mqMFES, 'async', dict(use_weight_init=True,
                                          weight_init_choosing='proportional',
                                          weight_method='rank_loss_p_norm',
                                          non_decreasing_weight=False,
                                          increasing_weight=True,
                                          rand_prob=0.2, )),    # test rand prob 0.2
    # median imputation
    amfesm1=(async_mqMFES, 'async', dict(use_weight_init=True,
                                         weight_init_choosing='proportional',
                                         weight_method='rank_loss_p_norm',
                                         non_decreasing_weight=False,
                                         increasing_weight=True,
                                         median_imputation='top')),
    amfesm2=(async_mqMFES, 'async', dict(use_weight_init=True,
                                         weight_init_choosing='proportional',
                                         weight_method='rank_loss_p_norm',
                                         non_decreasing_weight=False,
                                         increasing_weight=True,
                                         median_imputation='corresponding')),
    amfesm3=(async_mqMFES, 'async', dict(use_weight_init=True,
                                         weight_init_choosing='proportional',
                                         weight_method='rank_loss_p_norm',
                                         non_decreasing_weight=False,
                                         increasing_weight=True,
                                         median_imputation='all')),
)
