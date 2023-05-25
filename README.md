# WaveBench

Wavebench provides a comprehensive collection of datasets for training machine learning-based solvers for wave propagation PDEs.

# Dataset Description

The benchmark dataset includes two variants of wave propagation problems: **time-harmonic** and **time-varying** ones.

## Time-harmonic wave problems

In the time-harmonic wave problems, we aim to learn operator maps wavespeeds $c = c(\boldsymbol{x})$ with a spatial variable $\boldsymbol{x} \in \mathbb{R}^2$ to a pressure field $p = p(\boldsymbol{x}, \omega)$ with a prescribed frequency $\omega$. To this end, we create training dataset that consists of paired wavespeed and pressure field $\{(c_j, p_j)\}_{j}$. Below, we show samples of wavespeed $c$ and pressure field $p(\cdot, \omega)$ at different frequencies in our dataset.

<br />

<p align="center">
<img src = "./saved_figs/dataset_demo/wavebench_helmholtz_demo.png" width=80%>
</p>


## Time-varying wave problems

We consider two time-varying wave problems: the **reverse time continuation (RTC)** and the **Inverse Source (IS)** problem.


In **RTC**, our goal is to map the pressure field $p = p(\boldsymbol{x}, T)$ at the final time $T$ to the pressure field $p = p(\boldsymbol{x}, 0)$ at the initial time $0$. Traing samples of $p(\cdot, 0)$ and $p(\cdot, T)$ are shown below. Since the final pressure $p(\cdot, T)$ depends on both the initial pressure $p(\cdot, 0)$ and the wavespeed $c$, we create different versions of the dataset, each with a fixed wavespeed $c$ as a column of the below figure.

### Inverse Source (IS)

<p align="center">
<img src = "./saved_figs/dataset_demo/wavebench_rtc_demo.png" width=80%>
</p>

In IS, our goal is to predict the initial pressure $p(\cdot, 0)$ based on pressure measurements taken at certain boundary locations over a time span $[0, T]$. This dataset is closer to seismic imaging, where we can only measure the pressure field at the surface of the earth. We create different versions of the dataset, each with a fixed wavespeed $c$ as a column of the below figure.


<p align="center">
<img src = "./saved_figs/dataset_demo/wavebench_is_demo.png" width=80%>
</p>

## Usage

TODO

## Citation

If you use this benchmark dataset in your research, please cite our paper:

TODO

## License

TODO