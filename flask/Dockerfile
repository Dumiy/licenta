FROM ubuntu:18.04
RUN  apt update -y
RUN  apt-get install git -y
RUN  apt install python3-pip -y
RUN  apt install python-pip -y
RUN  apt-get install python3.6 -y


RUN git clone https://github.com/Dumiy/licenta
RUN pip install -r /licenta/flask/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN python /licenta/flask/app.py

EXPOSE 5500