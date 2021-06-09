from mq_hb.mq_random_search import mqRandomSearch
from mq_hb.mq_sh import mqSuccessiveHalving
from mq_hb.mq_hb import mqHyperband
from mq_hb.mq_bohb import mqBOHB
from mq_hb.mq_bohb_v0 import mqBOHB_v0
from mq_hb.mq_mfes import mqMFES
from mq_hb.mq_mfes_v4 import mqMFES_v4
from mq_hb.async_mq_sh import async_mqSuccessiveHalving
from mq_hb.async_mq_sh_v0 import async_mqSuccessiveHalving_v0
from mq_hb.async_mq_sh_v2 import async_mqSuccessiveHalving_v2
from mq_hb.async_mq_hb import async_mqHyperband
from mq_hb.async_mq_hb_v0 import async_mqHyperband_v0
from mq_hb.async_mq_hb_v2 import async_mqHyperband_v2
from mq_hb.async_mq_weight_hb import async_mqWeightHyperband
from mq_hb.async_mq_bohb import async_mqBOHB
from mq_hb.async_mq_bosh import async_mqBOSH
from mq_hb.async_mq_mfes_v3 import async_mqMFES_v3
from mq_hb.async_mq_mfes_v6 import async_mqMFES_v6
from mq_hb.async_mq_mfes_v12 import async_mqMFES_v12
from mq_hb.async_mq_mfes_v13 import async_mqMFES_v13
from mq_hb.async_mq_mfes_v14 import async_mqMFES_v14
from mq_hb.async_mq_mfes_v15 import async_mqMFES_v15
from mq_hb.async_mq_mfes_v16 import async_mqMFES_v16
from mq_hb.async_mq_mfes_v17 import async_mqMFES_v17
from mq_hb.async_mq_mfes_v18 import async_mqMFES_v18
from mq_hb.async_mq_mfes_v19 import async_mqMFES_v19
from mq_hb.async_mq_mfes_v20 import async_mqMFES_v20
from mq_hb.async_mq_mfes_v21 import async_mqMFES_v21
from mq_hb.async_mq_mfes_v22 import async_mqMFES_v22
from mq_hb.async_mq_mfes_v23 import async_mqMFES_v23
from mq_hb.async_mq_mfes_v24 import async_mqMFES_v24
from mq_hb.async_mq_mfes_v25 import async_mqMFES_v25
from mq_hb.async_mq_mfes_v26 import async_mqMFES_v26
from mq_hb.async_mq_mfes_v27 import async_mqMFES_v27

mth_dict = dict(
    random=(mqRandomSearch, 'sync'),
    sh=(mqSuccessiveHalving, 'sync'),
    hyperband=(mqHyperband, 'sync'),
    bohb=(mqBOHB, 'sync'),
    bohbv0=(mqBOHB_v0, 'sync'),
    mfes=(mqMFES, 'sync'),
    mfesv4=(mqMFES_v4, 'sync'),
    asha=(async_mqSuccessiveHalving, 'async'),
    ashav0=(async_mqSuccessiveHalving_v0, 'async'),
    ashav2=(async_mqSuccessiveHalving_v2, 'async'),
    ahb=(async_mqHyperband, 'async'),
    ahbv0=(async_mqHyperband_v0, 'async'),
    ahbv2=(async_mqHyperband_v2, 'async'),
    aweighthb=(async_mqWeightHyperband, 'async'),
    abohb=(async_mqBOHB, 'async'),
    abosh=(async_mqBOSH, 'async'),
    amfesv3=(async_mqMFES_v3, 'async'),
    amfesv6=(async_mqMFES_v6, 'async'),
    amfesv12=(async_mqMFES_v12, 'async'),
    amfesv13=(async_mqMFES_v13, 'async'),
    amfesv14=(async_mqMFES_v14, 'async'),
    amfesv15=(async_mqMFES_v15, 'async'),
    amfesv16=(async_mqMFES_v16, 'async'),
    amfesv17=(async_mqMFES_v17, 'async'),
    amfesv18=(async_mqMFES_v18, 'async'),
    amfesv19=(async_mqMFES_v19, 'async'),
    amfesv20=(async_mqMFES_v20, 'async'),
    amfesv21=(async_mqMFES_v21, 'async'),
    amfesv22=(async_mqMFES_v22, 'async'),
    amfesv23=(async_mqMFES_v23, 'async'),
    amfesv24=(async_mqMFES_v24, 'async'),
    amfesv25=(async_mqMFES_v25, 'async'),
    amfesv26=(async_mqMFES_v26, 'async'),
    amfesv27=(async_mqMFES_v27, 'async'),
)
