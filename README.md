### Smart rejector for Hate Speech

- name of student: Philippe Lammerts
- names of supervisors: Dr. J. Yang, Dr. Y-C. Hsu, Prof.dr.ir. G.J.P.M. Houben
- academic year: 2021/2022

### Introduction

The amount of hateful content that is spread online on social media platforms remains a serious problem. Manual moderation is still the most reliable solution but is simply infeasible due to the large amount of data generated every second on social media platforms. There exist automated solutions for detecting hate speech, and most of these use Machine Learning models. However, these models tend to be unreliable as they often perform poor on deployment data.

Therefore, in this project, we focus on Machine Learning models with a reject option. The goal of the reject option is to reject a prediction when the model is not confident enough. This Thesis project is about building the first smart rejector for detecting hate speech where the machine assists the human in detecting hate speech automatically and where the human makes the decisions when the machine is not confident enough.

### Research summary

### Installation and Usage

This project is developed with Python and uses the [Conda](https://docs.conda.io/en/latest/) environment management tool for keeping track of the packages (and correct versions) and for creating local environment.

To setup all the required packages and the correct Python version, run the following command:

`conda env create -f environment.yml`

To run the notebooks, you first need to run the following commands so that Jupyter notebook can recognize the Conda kernel:

`conda activate smart-rejector`
`python -m ipykernel install --user`

You can now open a notebook, go to Kernel>change kernel and select smart-rejector there.

### License

_We encourage you to use an open license and by default the repository contains an Apache License 2.0._

## Student project report

Please consider using the official TU Delft Latex Report template:
https://d2k0ddhflgrk1i.cloudfront.net/Websections/TU%20Delft%20Huisstijl/report_style.zip
