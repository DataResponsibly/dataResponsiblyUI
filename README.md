# dataResponsiblyUI
A Django project for the Web UIs of the Data, Responsibly, including

- DataSynthesizer (https://github.com/DataResponsibly/DataSynthesizer)
- RankingFacts (https://github.com/DataResponsibly/RankingFacts)

## Run the Web UIs locally

1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
3. Use the following command to initiate the environment.

```bash
git clone https://github.com/DataResponsibly/dataResponsiblyUI.git
cd dataResponsiblyUI
conda env create -f environment.yml
conda activate dataResponsiblyUI
python manage.py migrate
python manage.py runserver
```

Then you can access in the browser

- DataSynthesizer at http://127.0.0.1:8000/synthesizer
- RankingFacts at http://127.0.0.1:8000/rankingfacts

