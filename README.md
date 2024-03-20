# Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection

This is the official implementation of our paper '[Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection](https://www.researchgate.net/publication/363766436_Untargeted_Backdoor_Watermark_Towards_Harmless_and_Stealthy_Dataset_Copyright_Protection)', accepted in NeurIPS 2022 (selected as **Oral** paper, TOP 2%). This research project is developed based on Python 3 and Pytorch, created by [Yiming Li](http://liyiming.tech/) and [Yang Bai](https://scholar.google.com.sg/citations?user=wBH_Q1gAAAAJ&hl=zh-CN).

- 2024/03/20: We fixed a typo in UBW-C's codes. This modification will not influence the results reported in our paper since the typo was introduced due to our post-camera-ready code reconstruction. We are deeply sorry for the potential inconveniences that our typos may cause you.
- 2022/12/31: I have updated the codes of UBW-P. I will polish the codes of UBW-C and the README.md ASAP.
- 2022/12/01: I am deeply sorry that I have recently suspended the update of this Repo, due to some personal issues such as job hunting and sickness. I will release the codes and update this repo as soon as possible. Please refer to our submitted [codes and ckpts](https://www.dropbox.com/sh/djm0zehxwrwxbae/AAB6E19WFkVY9RwtHxv2Enfba?dl=0) for some insights.


### For UBW-C
-- model structure files:

model_i.py for ImageNet; 
model.py for other datasets.

-- run scripts
```
python UBW-C.py $SOURCE_CLASS$ $TARGET_CLASS$ $POISON_NUM$ $DATASET$ 
```

### For Ownership Verification
Please refer to [DVBW](https://github.com/THUYimingLi/DVBW) for more details about how to implement a t-test (but you need to slightly change something due to the diffiences of two hypothesis tests).

## Reference
```
@inproceedings{li2022untargeted,
  title={Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection},
  author={Li, Yiming and Bai, Yang and Jiang, Yong and Yang, Yong and Xia, Shu-Tao and Li, Bo},
  booktitle={NeurIPS},
  year={2022}
}
```
