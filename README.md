# Probe Request Association against MAC Randomization

This repo implements the papers of [Efficient Association of Wi-Fi Probe Requests under MAC](https://ieeexplore.ieee.org/document/9488769) and [Self-Supervised Association of Wi-Fi Probe Requests Under MAC Address Randomization](https://ieeexplore.ieee.org/abstract/document/9888045). 

### Create environment 
```
pip install -r requirement.txt
```

### An example of associating probe requests
```python
python assoc_api.py
```

Association result
```
[*] 11 out of 11 packet groups are finalized. 
[*] number of ultimate packet groups: 11
num_pkts = 76
num_pred_devices = 11
num_true_devices = 10
homogeneity = 1.000, completeness = 0.789, v_measure_score = 0.882
average_id_switches = 0.000
average_segs = 1.100
purity = 1.000
```

More association details are shown in `./data/result/`

### Citation
```
@inproceedings{tan2021efficient,
  title={Efficient Association of Wi-Fi Probe Requests under MAC Address Randomization},
  author={Tan, Jiajie and Chan, S-H Gary},
  booktitle={IEEE INFOCOM 2021-IEEE Conference on Computer Communications},
  pages={1--10},
  year={2021},
  organization={IEEE}
}
@article{he2022self,
  title={Self-Supervised Association of Wi-Fi Probe Requests under MAC Address Randomization},
  author={He, Tianlang and Tan, Jiajie and Chan, S-H Gary},
  journal={IEEE Transactions on Mobile Computing},
  volume={22},
  number={12},
  pages={7044--7056},
  year={2022},
  publisher={IEEE}
}
```
