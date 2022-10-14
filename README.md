# Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection

This is the official implementation of our paper '[Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection](http://liyiming.tech/publications/)', accepted in NeurIPS 2022. This research project is developed based on Python 3 and Pytorch, created by [Yiming Li](http://liyiming.tech/) and [Yang Bai](https://scholar.google.com.sg/citations?user=wBH_Q1gAAAAJ&hl=zh-CN).

The codes for reproducing our main experiments will be released soon.


### For UBW-C
-- model structure files:

model_i.py for ImageNet; 
model.py for other datasets.

-- run scripts
```
python UBW-C.py $SOURCE_CLASS$ $TARGET_CLASS$ $POISON_NUM$ $DATASET$ 
```

## Reference
```
@inproceedings{li2022untargeted,
  title={Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection},
  author={Li, Yiming and Bai, Yang and Jiang, Yong and Yang, Yong and Xia, Shu-Tao and Li, Bo},
  booktitle={NeurIPS},
  year={2022}
}
```
