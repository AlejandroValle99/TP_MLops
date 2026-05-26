FROM apache/airflow:slim-3.2.2rc2
COPY config/requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.2.2rc2 -r /requirements.txt
