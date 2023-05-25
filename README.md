# WaveBench

Welcome to WaveBench, a comprehensive collection of datasets designed for training machine learning-based solvers to wave propagation partial differential equations (PDEs).


# Dataset Description

The benchmark dataset contains two variants of wave propagation problems, **time-harmonic** and **time-varying**.


## Time-harmonic wave problems

In the time-harmonic wave problems, we aim to learn an operator that maps wavespeeds $c = c(\boldsymbol{x})$ to a pressure field $p = p(\boldsymbol{x}, \omega)$ at a prescribed frequency $\omega$. Here, $\boldsymbol{x}$ is a spatial coordinate in 2D.

The time-harmonic datasets consist of paired wavespeed and pressure fields $\{(c_j, p_j)\}_{j}$. The following illustrates samples of wavespeed $c$ and pressure field $p(\cdot, \omega)$ at different frequencies. 

<br />
<p align="center">
<img src = "./saved_figs/dataset_demo/wavebench_helmholtz_demo.png" width=80%>
</p>


## Time-varying wave problems

Time-varying wave problems comprise two categories: **reverse time continuation** and **inverse source** problems.

### Reverse time continuation (RTC)

The objective in **RTC** is to map the pressure field $p = p(\boldsymbol{x}, T)$ at the final time $T$ to the pressure field $p = p(\boldsymbol{x}, 0)$ at the initial time $0$. The figure below shows training samples of $p(\cdot, 0)$ and $p(\cdot, T)$. Various versions of the dataset are available, each simulated with a fixed wavespeed $c$ as shown in the top row. While the final pressure depends both on the initial pressure and the wavespeed, the mapping we aim to estimate is conditioned on a fixed wavespeed.

<br />
<p align="center">
<img src = "./saved_figs/dataset_demo/wavebench_rtc_demo.png" width=80%>
</p>

### Inverse source (IS)

In IS, the aim is to predict the initial pressure $p(\cdot, 0)$ based on pressure measurements collected at specific boundary locations over a time interval $[0, T]$. This dataset is insipred by seismic imaging, where pressure field measurements are only feasible at the earth's surface. We offer different versions of this dataset, each with a fixed wavespeed $c$ as depicted in the columns below.

<br />
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