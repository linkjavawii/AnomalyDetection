import pandas as pd
from flask import url_for, redirect, render_template,Flask,request,flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug import secure_filename
from flask_bootstrap import Bootstrap
from pathlib import Path
from functools import wraps, update_wrapper
from datetime import datetime
from wtforms import FloatField, validators,StringField,IntegerField,FileField,BooleanField,SelectField
from modelDetection2 import changePointDetectModel,subSeqDetectModel,textDetectModel,supervisedOutlier,numericalOutlier,mixedDataOutlier
from form2 import PointDetectForm,SubseqDetectForm,TextDetectForm,SupervisedDetectForm,NumericalDetectForm,MixedDetectForm,Choice,UploadForm

######################################### remove cache ####################################
def nocache(view):
  @wraps(view)
  def no_cache(*args, **kwargs):
    response = make_response(view(*args, **kwargs))
    response.headers['Last-Modified'] = datetime.now()
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response      
  return update_wrapper(no_cache, view)
######################################### remove cache ####################################


class Config(object):
    SECRET_KEY = '78w0o5tuuGex5Ktk8VvVDF9Pw3jv1MVE'

app = Flask(__name__)
app.config.from_object(Config)
Bootstrap(app)
tmp_folder = Path(__file__).parent / 'uploads'
tmp_folder.mkdir(parents=True, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def choice():
    form = Choice()
    if request.method == 'POST':
        if form.point.data:
            return redirect(url_for('changePointApi'))
        
        elif form.subseq.data:
            return redirect(url_for('subSeqApi'))
        
        elif form.text.data:
            return redirect(url_for('textApi'))
        
        elif form.supervised.data:
            return redirect(url_for('supervisedApi'))
        
        #elif form.numerical.data:
            #return redirect(url_for('numericalApi'))
        
        elif form.mixed.data:
            return redirect(url_for('mixedApi'))

    
    return render_template('ChoiceAPI2.html', form=form)

@app.route('/changePointApi', methods=['GET', 'POST'])
def changePointApi():
    form = UploadForm()

    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('processing',filename=filename))

    return render_template('upload_ts_file.html', form=form)

@app.route('/subSeqApi', methods=['GET', 'POST'])
def subSeqApi():
    form = UploadForm()

    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('processing2',filename=filename))

    return render_template('upload_ts_file.html', form=form)

@app.route('/textApi', methods=['GET', 'POST'])
def textApi():
    form = UploadForm()

    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('processing3',filename=filename))

    return render_template('upload_ts_file.html', form=form)

@app.route('/supervisedApi', methods=['GET', 'POST'])
def supervisedApi():
    form = UploadForm()

    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('processing4',filename=filename))

    return render_template('upload_ts_file.html', form=form)


@app.route('/numericalApi', methods=['GET', 'POST'])
def numericalApi():
    form = UploadForm()

    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('processing5',filename=filename))

    return render_template('upload_ts_file.html', form=form)


@app.route('/mixedApi', methods=['GET', 'POST'])
def mixedApi():
    form = UploadForm()

    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('processing6',filename=filename))

    return render_template('upload_ts_file.html', form=form)


@app.route('/processing/<filename>',methods=['GET', 'POST'])
def processing(filename):
    data = pd.read_csv('uploads/{}'.format(filename))
    form = PointDetectForm(csrf_enabled=False)
    
    if request.method == 'GET':
        colnames = data.columns
        date_colname = colnames[colnames.str.match(r'date', case=False) == True]
        feature_list = data.columns.to_list()
        
        if not date_colname.empty:
            feature_list.remove(date_colname[0])

        class CustomForm(PointDetectForm):
            pass
        
        target_selection = []
        for item in feature_list:
            target_selection.append((item,item))
            
        setattr(PointDetectForm, "target_col", SelectField(label="Target column", choices=target_selection))
            
        form = CustomForm(request.form)
        form.LSTM.data = True
        form.gru.data = True
        form.cnn_lstm.data = True
        form.Conv1D.data = True
        form.BiLstm.data = True
        form.ARIMA.data = True
            
        if not date_colname.empty:
            form.datetime_col.data = date_colname[0]
            
        return render_template('ChangePointDetect2.html', form=form, features=feature_list)    
    
    if request.method == 'POST' :
        if form.training.data:
            bar = changePointDetectModel(data, form.datetime_col.data,
                         form.target_col.data,form.lookback.data,form.epochs.data,
                                         form.LSTM.data,form.gru.data,form.Conv1D.data,
                                         form.cnn_lstm.data,form.BiLstm.data,
                                         form.ARIMA.data,form.firstOutliers.data)
            
        else:
            bar = None
        return render_template('ChangePointDetect2.html',form=form,plot=bar)

@app.route('/processing2/<filename>',methods=['GET', 'POST'])
def processing2(filename):    
    data = pd.read_csv('uploads/{}'.format(filename))
    form = SubseqDetectForm(csrf_enabled=False)
    if request.method == 'GET':
        colnames = data.columns
        date_colname = colnames[colnames.str.match(r'date', case=False) == True]
        feature_list = data.columns.to_list()
        
        if not date_colname.empty:
            feature_list.remove(date_colname[0])

        class CustomForm(SubseqDetectForm):
            pass
        
        target_selection = []
        for item in feature_list:
            target_selection.append((item,item))
            
        setattr(SubseqDetectForm, "target_col", SelectField(label="Target column", choices=target_selection))
            
        form = CustomForm(request.form)
        
        form.IF.data = True
        form.knn.data = True
        form.LOF.data = True
        form.AE.data = True
        form.Conv1D_AE.data = True
        form.LSTM_AE.data = True
        form.kmeans.data = True
        form.pca.data = True
                
        if not date_colname.empty:
            form.datetime_col.data = date_colname[0]
            
        return render_template('SubSeqDetect2.html', form=form, features=feature_list)
     
    if request.method == 'POST' :
        if form.training.data:
            result = subSeqDetectModel(data, form.datetime_col.data,
                         form.target_col.data, form.subseqlength.data,form.epochs.data,
                                       form.numberOfNeighbors.data,form.numberOfTrees.data,
                                       form.IF.data,
                                       form.knn.data,form.LOF.data,
                                       form.AE.data,form.Conv1D_AE.data,
                                       form.LSTM_AE.data,form.kmeans.data,
                                       form.pca.data,form.firstOutliers.data)
        else:
            result = None
            
        return render_template('SubSeqDetect2.html', form=form,result=result)


    
@app.route('/processing3/<filename>',methods=['GET', 'POST'])
def processing3(filename):
    data = pd.read_csv('uploads/{}'.format(filename))
    form = TextDetectForm(csrf_enabled=False)
    if request.method == 'GET':
        colnames = data.columns
        feature_list = data.columns.to_list()

        class CustomForm(TextDetectForm):
            pass
        
        target_selection = []
        for item in feature_list:
            target_selection.append((item,item))
            
        setattr(TextDetectForm, "target_col", SelectField(label="Text column", choices=target_selection))
            
        form = CustomForm(request.form)
        
        form.knn.data = True
        form.LOF.data = True
        form.LSA.data = True
        form.AE.data = True
            
        return render_template('TextDetect2.html', form=form, features=feature_list)
    
    if request.method == 'POST' :
        if form.training.data:
            result = textDetectModel(data,
                         form.target_col.data,form.knn.data,
                                       form.LOF.data,form.LSA.data,
                                       form.AE.data,form.firstOutliers.data,
                                        form.numberOfNeighbors.data,form.epochs.data)
        else:
            result = None
            
        return render_template('TextDetect2.html', form=form,tables=[result[0]])


    
    
    
@app.route('/<filename>',methods=['GET', 'POST'])
def processing4(filename):
    data = pd.read_csv('uploads/{}'.format(filename))
    form = SupervisedDetectForm(csrf_enabled=False)
    if request.method == 'GET':
        colnames = data.columns
        num_cols = data._get_numeric_data().columns.to_list()
        cat_cols = list(set(data.columns)-set(num_cols))
        feature_list = data.columns.to_list()


        class CustomForm(SupervisedDetectForm):
            pass
        
        target_selection = []
        for item in feature_list:
            target_selection.append((item,item))
            
        setattr(SupervisedDetectForm, "label_col", SelectField(label="Label column for supervised learning", choices=target_selection))
        for item in feature_list:
            setattr(SupervisedDetectForm, item, BooleanField(label=item, render_kw={'checked': True}))
            
        form = CustomForm(request.form)
        for item in feature_list:
            form.data[item] = True
        
        form.RF_OS.data = True
        form.RF_SM.data = True
        form.RF_US_ENS.data = True
        form.XGB_OS.data = True
        form.XGB_SM.data = True
        form.XGB_US_ENS.data = True
            
        return render_template('SupervisedDetect2.html', form=form, num_features=num_cols, cat_features=cat_cols)
        #render_template('SupervisedDetect2.html', form=form, features=feature_list)

    
    if request.method == 'POST' :
        if form.training.data:
            result = supervisedOutlier(data,form)
        else:
            result = None
            
        return render_template('SupervisedDetect2.html', form=form,result=result[0],cm_list=result[1])

@app.route('/processing5/<filename>',methods=['GET', 'POST'])
def processing5(filename):
    data = pd.read_csv('uploads/{}'.format(filename))
    form = NumericalDetectForm(csrf_enabled=False)
    if request.method == 'GET':
        colnames = data.columns
        feature_list = data.columns.to_list()

        class CustomForm(NumericalDetectForm):
            pass
        
        for item in feature_list:
            setattr(NumericalDetectForm, item, BooleanField(label=item, render_kw={'checked': True}))
            
        form = CustomForm(request.form)
        for item in feature_list:
            form.data[item] = True

        form.IF.data = True
        form.knn.data = True
        form.LOF.data = True
        form.AE.data = True
        form.kmeans.data = True
            
        return render_template('NumericalDetect2.html', form=form, features=feature_list)

    
    if request.method == 'POST' :
        if form.training.data:
            result = numericalOutlier(data,form)
        else:
            result = None
            
        return render_template('NumericalDetect2.html', form=form,tables=[result[0]],titles=result[1])    

    
@app.route('/processing6/<filename>',methods=['GET', 'POST'])
def processing6(filename):
    data = pd.read_csv('uploads/{}'.format(filename))
    form = MixedDetectForm(csrf_enabled=False)
    if request.method == 'GET':
        colnames = data.columns
        num_cols = data._get_numeric_data().columns.to_list()
        cat_cols = list(set(data.columns)-set(num_cols))
        feature_list = data.columns.to_list()

        class CustomForm(MixedDetectForm):
            pass
        
        for item in feature_list:
            setattr(MixedDetectForm, item, BooleanField(label=item, render_kw={'checked': True}))
            
        form = CustomForm(request.form)
        for item in feature_list:
            form.data[item] = True

        form.knn.data = True
        form.LOF.data = True
        form.pca.data = True
        form.AE.data = True
        form.IF.data = True
        #form.kmeans.data = True
            
        return render_template('MixedDetect2.html', form=form, num_features=num_cols, cat_features=cat_cols)

    
    if request.method == 'POST' :
        if form.training.data:
            result = mixedDataOutlier(data,form)
        else:
            result = None
            
        return render_template('MixedDetect2.html', form=form,tables=[result[0]],titles=result[1])    

    
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
