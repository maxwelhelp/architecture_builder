# Matrix Program Pseudocode

Source: `runs/v13_review_fixed_45ep_tuned/seq_analysis_epoch_045.json`

Epoch: `45`

## Metrics
- train: loss `0.4342905811725124` acc `0.8513104838709677`
- val: loss `1.393229221343994` acc `0.605`

## Learned Program

### Layer 0
- **L0.B0** role=`extract/temporal_spectral` gate=`0.9991322755813599`
  - categories: extract=0.974, compare=0.009, memory=0.008
  - ops: short_onset=0.458, delta=0.299, offset=0.205, bilinear=0.008, energy_count=0.005
  - смысл: короткие onset/атаки; изменения во времени; смещение/раннее-позднее отличие
  - task focus: global_loss_error=0.096, global_true_pred_distribution=0.093, confusion_row:yes=0.075
- **L0.B1** role=`extract/temporal_spectral` gate=`0.9991686344146729`
  - categories: extract=0.972, compare=0.009, memory=0.009
  - ops: short_onset=0.438, delta=0.316, offset=0.207, bilinear=0.009, energy_count=0.005
  - смысл: короткие onset/атаки; изменения во времени; смещение/раннее-позднее отличие
  - task focus: global_true_pred_distribution=0.100, global_loss_error=0.078, confusion_row:yes=0.064
- **L0.B2** role=`extract/temporal_spectral` gate=`0.9991893172264099`
  - categories: extract=0.973, memory=0.009, compare=0.009
  - ops: short_onset=0.457, delta=0.294, offset=0.210, bilinear=0.008, energy_count=0.005
  - смысл: короткие onset/атаки; изменения во времени; смещение/раннее-позднее отличие
  - task focus: global_loss_error=0.083, global_true_pred_distribution=0.082, top_confusion_pair_cell_0=0.081
- **L0.B3** role=`extract/temporal_spectral` gate=`0.9991440176963806`
  - categories: extract=0.975, compare=0.008, memory=0.008
  - ops: short_onset=0.455, delta=0.300, offset=0.208, bilinear=0.008, energy_count=0.005
  - смысл: короткие onset/атаки; изменения во времени; смещение/раннее-позднее отличие
  - task focus: global_true_pred_distribution=0.095, confusion_row:yes=0.088, global_loss_error=0.071

### Layer 1
- **L1.B0** role=`aggregate/global_summary` gate=`0.9940699934959412`
  - categories: repair=0.581, aggregate=0.229, suppress=0.054
  - ops: residual_refdelta=0.389, normalize=0.184, global_summary=0.111, ema_memory=0.076, block_compare=0.046
  - смысл: ремонт residual/delta; нормализация состояния; глобальная сводка
  - task focus: global_loss_error=0.068, global_true_pred_distribution=0.067, top_confusion_pair_cell_0=0.060
- **L1.B1** role=`aggregate/global_summary` gate=`0.9947237968444824`
  - categories: repair=0.475, aggregate=0.350, suppress=0.072
  - ops: residual_refdelta=0.325, normalize=0.169, global_summary=0.156, ema_memory=0.105, block_compare=0.062
  - смысл: ремонт residual/delta; нормализация состояния; глобальная сводка
  - task focus: global_true_pred_distribution=0.062, global_loss_error=0.048, class_pressure:up=0.047
- **L1.B2** role=`repair/normalize` gate=`0.9976429343223572`
  - categories: repair=0.726, aggregate=0.078, compare=0.067
  - ops: residual_refdelta=0.495, normalize=0.220, bilinear=0.057, global_summary=0.053, ema_memory=0.040
  - смысл: ремонт residual/delta; нормализация состояния; билинейное смешивание
  - task focus: top_confusion_pair_cell_0=0.082, global_true_pred_distribution=0.082, top_confusion_pair_cell_1=0.079
- **L1.B3** role=`aggregate/global_summary` gate=`0.9949203729629517`
  - categories: repair=0.566, aggregate=0.251, suppress=0.056
  - ops: residual_refdelta=0.387, normalize=0.178, global_summary=0.119, ema_memory=0.082, block_compare=0.047
  - смысл: ремонт residual/delta; нормализация состояния; глобальная сводка
  - task focus: global_true_pred_distribution=0.066, confusion_row:yes=0.055, global_loss_error=0.049

### Layer 2
- **L2.B0** role=`repair/normalize` gate=`0.9896517395973206`
  - categories: aggregate=0.415, repair=0.396, suppress=0.098
  - ops: residual_refdelta=0.238, ema_memory=0.182, normalize=0.150, global_summary=0.140, suppress=0.063
  - смысл: ремонт residual/delta; EMA-память; нормализация состояния
  - task focus: top_confusion_pair_cell_0=0.063, top_confusion_pair_cell_1=0.061, global_loss_error=0.056
- **L2.B1** role=`aggregate/global_summary` gate=`0.9904503226280212`
  - categories: aggregate=0.574, repair=0.251, suppress=0.115
  - ops: ema_memory=0.248, global_summary=0.175, residual_refdelta=0.155, normalize=0.130, energy_count=0.075
  - смысл: EMA-память; глобальная сводка; ремонт residual/delta
  - task focus: class_pressure:up=0.050, global_true_pred_distribution=0.046, class_pressure:left=0.044
- **L2.B2** role=`repair/normalize` gate=`0.9965859055519104`
  - categories: repair=0.734, memory=0.104, aggregate=0.082
  - ops: residual_refdelta=0.480, normalize=0.221, ema_memory=0.071, global_summary=0.051, small_refine=0.032
  - смысл: ремонт residual/delta; нормализация состояния; EMA-память
  - task focus: top_confusion_pair_cell_0=0.103, top_confusion_pair_cell_1=0.099, global_true_pred_distribution=0.073
- **L2.B3** role=`repair/normalize` gate=`0.9915843605995178`
  - categories: aggregate=0.415, repair=0.407, suppress=0.094
  - ops: residual_refdelta=0.251, ema_memory=0.196, normalize=0.152, global_summary=0.131, suppress=0.060
  - смысл: ремонт residual/delta; EMA-память; нормализация состояния
  - task focus: global_true_pred_distribution=0.051, top_confusion_pair_cell_0=0.044, confusion_row:yes=0.043

### Layer 3
- **L3.B0** role=`aggregate/global_summary` gate=`0.9853985905647278`
  - categories: aggregate=0.749, suppress=0.109, repair=0.088
  - ops: global_summary=0.269, ema_memory=0.206, energy_count=0.157, block_compare=0.097, suppress=0.071
  - смысл: глобальная сводка; EMA-память; учёт энергии
  - task focus: global_loss_error=0.045, global_true_pred_distribution=0.043, class_pressure:down=0.042
- **L3.B1** role=`aggregate/global_summary` gate=`0.9866343140602112`
  - categories: aggregate=0.839, suppress=0.096, repair=0.034
  - ops: global_summary=0.308, ema_memory=0.229, energy_count=0.160, block_compare=0.105, memory_write=0.072
  - смысл: глобальная сводка; EMA-память; учёт энергии
  - task focus: class_pressure:up=0.059, class_pressure:stop=0.053, class_pressure:right=0.053
- **L3.B2** role=`aggregate/global_summary` gate=`0.992777943611145`
  - categories: repair=0.585, aggregate=0.182, memory=0.135
  - ops: residual_refdelta=0.358, normalize=0.167, global_summary=0.134, ema_memory=0.093, suppress=0.074
  - смысл: ремонт residual/delta; нормализация состояния; глобальная сводка
  - task focus: top_confusion_pair_cell_0=0.083, global_true_pred_distribution=0.080, top_confusion_pair_cell_1=0.080
- **L3.B3** role=`aggregate/global_summary` gate=`0.9871551394462585`
  - categories: aggregate=0.763, suppress=0.106, repair=0.084
  - ops: global_summary=0.286, ema_memory=0.220, energy_count=0.140, block_compare=0.095, suppress=0.070
  - смысл: глобальная сводка; EMA-память; учёт энергии
  - task focus: class_pressure:up=0.046, confusion_row:up=0.042, global_true_pred_distribution=0.042

## High-level pattern

ранний слой извлекает onset/delta/time/freq признаки; средние блоки делают residual repair / normalize / small_refine; safe ops реально используются, поэтому блоки могут не писать разрушительный update; память участвует как EMA/read/write компонент.

## Practical read

Это soft program. Если top-ops стабильны на heldout и между эпохами, их можно переводить в hard/top-k программу.
