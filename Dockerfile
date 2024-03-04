FROM python:3.11

WORKDIR /code

COPY requirements.txt nipystats/
RUN pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org -r nipystats/requirements.txt
COPY nipystats nipystats/nipystats
COPY setup.py nipystats/

RUN python -m pip install /code/nipystats/

COPY docker/entrypoint.sh .
ENTRYPOINT ["/code/entrypoint.sh"]
