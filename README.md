# ChARM Trainer (and caller)

ChARM Trainer is a PyTorch framework for I/Q sample classification.
It has been used with both a residual network (ResNet) and a convolutional neural network (CNN).

ChARM Trainer learns to classify the I/Q samples and to label them with respect the originating wireless technology (e.g., LTE, WiFi).
It includes a _caller_ utility which can be executed with a small amount (20k samples) of data for classification a-posteriori.

## ChARM work

ChARM Trainer stems from a research project at the Northeastern University [1], if you use this code, please cite our work:

```
@inproceedings{Baldesi2022Charm,
  author = {Baldesi, Luca and Restuccia, Francesco and Melodia, Tommaso},
  booktitle = {{IEEE INFOCOM 2022 - IEEE Conference on Computer Communications}},
  title = {{ChARM: NextG Spectrum Sharing Through Data-Driven Real-Time O-RAN Dynamic Control}},
  year = {2022},
  month = may 
}
```

## ChARM dataset

ChARM Trainer has been released with a dataset for training, validation, and testing [2].
The dataset was collected using Xilinx and USRP software defined radios running the [OpenWiFi](https://github.com/open-sdr/openwifi) and [srsRAN](https://github.com/srsran/srsRAN) software stacks.

## References

1. L. Baldesi, F. Restuccia and T. Melodia. "ChARM: NextG Spectrum Sharing Through Data-Driven Real-Time O-RAN Dynamic Control", IEEE INFOCOM 2022 - IEEE Conference on Computer Communications, May 2022.
2. http://hdl.handle.net/2047/D20423481
