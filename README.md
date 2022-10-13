### Value-Sensitive Rejection of Machine Learning Predictions for Hate Speech Detection

- name of student: Philippe Lammerts
- names of supervisors: Dr. J. Yang, Dr. Y-C. Hsu, Prof.dr.ir. G.J.P.M. Houben, P. Lippmann
- academic year: 2021/2022

### Research summary

Hate speech detection on social media platforms remains a challenging task. Manual moderation by humans is the most reliable but infeasible, and machine learning models for detecting hate speech are scalable but unreliable as they often perform poorly on unseen data. Therefore, human-AI collaborative systems, in which we combine the strengths of humans' reliability and the scalability of machine learning, offer great potential for detecting hate speech. While methods for task handover in human-AI collaboration exist that consider the costs of incorrect predictions, insufficient attention has been paid to estimating these costs. In this work, we propose a value-sensitive rejector that automatically rejects machine learning predictions when the prediction's confidence is too low by taking into account the users' perception regarding different types of machine learning predictions. We conducted a crowdsourced survey study with 160 participants to evaluate their perception of correct, incorrect and rejected predictions in the context of hate speech detection. We introduce magnitude estimation, an unbounded scale, as the preferred method for measuring user perception of machine predictions. The results show that we can use magnitude estimation reliably for measuring the users' perception. We integrate the user-perceived values into the value-sensitive rejector and apply the rejector to several state-of-the-art hate speech detection models. The results show that the value-sensitive rejector can help us to determine when to accept or reject predictions to achieve optimal model value. Furthermore, the results show that the best model can be different when optimizing model value compared to optimizing more widely used metrics, such as accuracy.

### Installation and Usage

This project is developed with Python and uses the [Conda](https://docs.conda.io/en/latest/) environment management tool for keeping track of the packages (and correct versions) and for creating local environment.

To setup all the required packages and the correct Python version, run the following command:

`conda env create -f environment.yml`

To run the notebooks, you first need to run the following commands so that Jupyter notebook can recognize the Conda kernel:

`conda activate value-sensitive-rejector`

`python -m ipykernel install --user`

You can now open a notebook, go to Kernel>change kernel and select `value-sensitive-rejector` there.

### License

_We encourage you to use an open license and by default the repository contains an Apache License 2.0._

## Student project report

Please consider using the official TU Delft Latex Report template:
https://d2k0ddhflgrk1i.cloudfront.net/Websections/TU%20Delft%20Huisstijl/report_style.zip
