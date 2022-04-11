# CoLive

## Framework

![framework](https://s2.loli.net/2021/12/25/2vVCiamkjdrlnAc.png)



## ConvLSTM hybrid model

![ConvLSTM](https://s2.loli.net/2022/04/11/snrUBmIJf17gQTV.png)



## Environmental requirements

- PyTorch: 1.8
- Python: 3.7
- Other python packages: please refer to the following code



## Code Structure

- [Arguments.py](./Arguments.py): model configuration of all parameters
- [convlstm.py](./convlstm.py): ConvLSTM model structure
- [Convtrain.py](./Convtrain.py):  training and testing part of the code for hybrid model
- [get_frames.py](./get_frames.py): Extract video frames
- Visual test metrics: [plot.py](./plot.py), [plot_tiles.py](./plot_tiles.py), [plot_time.py](./plot_time.py), [plot_vs.py](./plot_vs.py)

