FROM ubuntu:18.04
RUN  apt update -y
RUN  apt-get install git -y
RUN  apt install python3-pip -y
RUN  apt install python-pip -y
RUN  apt-get install python3.7 -y
RUN  apt-get install vim  -y

RUN  apt update -y
RUN cd ~
RUN git clone https://github.com/Dumiy/licenta
RUN pip install -r /licenta/flask/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN cd licenta/flask
RUN mkdir /licenta/flask/static
RUN mkdir /licenta/flask/static/files
EXPOSE 5500
ENTRYPOINT [ "python" ]
CMD [ "app.py"]
