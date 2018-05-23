FROM python:3.6.5

WORKDIR /var/app

RUN pip3 install virtualenv
RUN virtualenv /var/app
ADD . /var/app
COPY ./linux-requirements.txt /var/app/requirements.txt
RUN if [ -f /var/app/requirements.txt ]; then pip3 install -r /var/app/requirements.txt; fi

CMD ["python", "foto2vam.py"]
