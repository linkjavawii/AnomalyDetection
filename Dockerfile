FROM python:3.6.8
ARG VERSION
LABEL org.label-schema.version=$VERSION
RUN mkdir webapp
COPY ./requirements.txt /webapp/requirements.txt
COPY ./Api_outliers2.py /webapp/Api_outliers2.py
COPY ./form2.py /webapp/form2.py
COPY ./modelDetection2.py /webapp/modelDetection2.py
COPY ./tempates /webapp/tempates
WORKDIR /webapp
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "Api_outliers2.py", "--host", "0.0.0.0" ]
